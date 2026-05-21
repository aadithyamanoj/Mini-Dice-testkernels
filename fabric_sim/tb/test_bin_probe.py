"""Cocotb probe: load one .bin into dice_top, drive boundary, sample outputs.

Lifted from dora's test_dice_gemm._program_from_bin scanchain-load pattern
(same fabric, same scanchain), with the kernel-specific golden checks removed.
This test purely OBSERVES: it prints mem_addr_o_* / mem_data_o_* / ext_data_o_*
for each value of regS_i_0 in the sweep range, so we can see directly whether
the CGRA fabric (without the FPGA controller or DPI in the loop) emits the
expected affine addresses for our bitstreams.

Env-driven:
  BIN_PATH       path to .bin (required)
  BITSTREAM_BITS bitstream payload bit count (required)
  CSR_X0..7      hex/dec strings (default 0)
  R0..R7         hex/dec value to drive on ext_data_i_0..7 (default 0)
                 (these are the per-thread GP regs from the source_map)
  C0..C7         hex/dec value to drive on ext_data_i_8..15 (default 0)
                 (constant registers)
  REGS0_SWEEP    "lo..hi" inclusive (default "0..0")
  SETTLE_CYCLES  cycles after each input change (default 8)
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, RisingEdge


# Same constants as dora's test_dice_gemm — taken from the static-build chip:
# mini_dice array = 25 SBs + 16 PEs = 41 instances, each with head/tail
# delimiters, plus 1 head + 1 tail for the array = 84 delimiter stages.
SCANCHAIN_DELIM_DEPTH = 84

NUM_EXT_DATA_INPUTS  = 16
NUM_EXT_DATA_OUTPUTS = 16
NUM_CSRX             = 8
NUM_MEM_PORTS        = 4
NUM_EXT_PRED_INPUTS  = 2


def _parse_int(s: str, default: int = 0) -> int:
    if s is None or s == "":
        return default
    return int(s, 0)


def _parse_range(spec: Optional[str]) -> list[int]:
    if not spec:
        return [0]
    if ".." in spec:
        lo, hi = spec.split("..", 1)
        return list(range(int(lo, 0), int(hi, 0) + 1))
    return [int(spec, 0)]


def _bin_bits_lsb(data: bytes, bit_count: int) -> list[int]:
    """Decode .bin bytes -> serial bit stream, LSB-first per byte. Matches
    dora's `lsb` decode mode (the working mode for static-build)."""
    bits: list[int] = []
    for byte in data:
        for i in range(8):
            bits.append((byte >> i) & 1)
    if bit_count > len(bits):
        raise AssertionError(
            f"Requested bit_count={bit_count} exceeds packed bits={len(bits)}")
    return bits[:bit_count]


async def setup_clocks(dut) -> None:
    cocotb.start_soon(Clock(dut.clk_i, 10, unit="ns").start())
    cocotb.start_soon(Clock(dut.prog_clk_i, 12, unit="ns").start())
    await ClockCycles(dut.clk_i, 1)


async def reset_dut(dut) -> None:
    dut.reset_i.value     = 1
    dut.prog_rst_i.value  = 1
    dut.en_i.value        = 1
    dut.prog_we_i.value   = 0
    dut.prog_done_i.value = 0
    dut.prog_din_i.value  = 0
    await ClockCycles(dut.clk_i, 4)
    await ClockCycles(dut.prog_clk_i, 4)
    dut.reset_i.value    = 0
    dut.prog_rst_i.value = 0
    await ClockCycles(dut.clk_i, 3)
    await ClockCycles(dut.prog_clk_i, 3)


async def _shift_one(dut, bit: int) -> None:
    dut.prog_din_i.value = bit
    await RisingEdge(dut.prog_clk_i)


async def program_from_bin(dut, bin_path: Path, bit_count: int,
                            extra_flush: int = SCANCHAIN_DELIM_DEPTH) -> None:
    data = bin_path.read_bytes()
    bits = _bin_bits_lsb(data, bit_count)
    dut.prog_done_i.value = 0
    dut.prog_we_i.value   = 1
    for b in bits:
        await _shift_one(dut, b)
    dut.prog_we_i.value = 0
    for _ in range(max(extra_flush, SCANCHAIN_DELIM_DEPTH)):
        await RisingEdge(dut.prog_clk_i)
    dut.prog_din_i.value  = 0
    dut.prog_done_i.value = 1
    await ClockCycles(dut.prog_clk_i, 16)
    await ClockCycles(dut.clk_i, 8)


def drive_boundary_zero(dut) -> None:
    for i in range(NUM_EXT_DATA_INPUTS):
        getattr(dut, f"ext_data_i_{i}").value = 0
    for i in range(NUM_EXT_PRED_INPUTS):
        getattr(dut, f"ext_pred_i_{i}").value = 0


def drive_data_inputs(dut, r_vals: dict[int, int], c_vals: dict[int, int]) -> None:
    """Drive ext_data_i_0..7 (= R0..R7) and ext_data_i_8..15 (= C0..C7).

    source_map.json maps R<n> -> ext_data_i_<n> and C<n> -> ext_data_i_<n+8>,
    so when a compute DFG declares `R0 [opcode=input]` the fabric reads the
    value from ext_data_i_0. This lets us inject known test values into the
    register-file-equivalent inputs without needing to run a prior load p-graph.
    """
    for i in range(8):
        getattr(dut, f"ext_data_i_{i}").value     = r_vals.get(i, 0) & 0xFFFF
        getattr(dut, f"ext_data_i_{i + 8}").value = c_vals.get(i, 0) & 0xFFFF


def drive_csrs(dut, csrs: dict[int, int]) -> None:
    for idx in range(NUM_CSRX):
        getattr(dut, f"csrX_i_{idx}").value = csrs.get(idx, 0) & 0xFFFF


def snapshot(dut) -> dict:
    return {
        "mem_addr": [int(getattr(dut, f"mem_addr_o_{i}").value)
                     for i in range(NUM_MEM_PORTS)],
        "mem_data": [int(getattr(dut, f"mem_data_o_{i}").value)
                     for i in range(NUM_MEM_PORTS)],
        "ext_data": [int(getattr(dut, f"ext_data_o_{i}").value)
                     for i in range(NUM_EXT_DATA_OUTPUTS)],
        "ext_pred": [int(getattr(dut, f"ext_pred_o_{i}").value)
                     for i in range(NUM_EXT_PRED_INPUTS)],
    }


@cocotb.test()
async def probe_bin(dut):
    bin_path_str   = os.environ.get("BIN_PATH")
    bit_count_str  = os.environ.get("BITSTREAM_BITS")
    if not bin_path_str or not bit_count_str:
        raise RuntimeError("Set BIN_PATH and BITSTREAM_BITS env vars")
    bin_path  = Path(bin_path_str).resolve()
    bit_count = int(bit_count_str, 0)
    if not bin_path.exists():
        raise FileNotFoundError(bin_path)

    csrs   = {i: _parse_int(os.environ.get(f"CSR_X{i}"), 0) for i in range(NUM_CSRX)}
    r_vals = {i: _parse_int(os.environ.get(f"R{i}"), 0) for i in range(8)}
    c_vals = {i: _parse_int(os.environ.get(f"C{i}"), 0) for i in range(8)}
    regs0_values  = _parse_range(os.environ.get("REGS0_SWEEP", "0"))
    settle_cycles = _parse_int(os.environ.get("SETTLE_CYCLES"), 8)

    log = dut._log
    log.info("[PROBE] bin=%s bits=%d", bin_path, bit_count)
    log.info("[PROBE] CSRs: " + " ".join(f"csrX{i}=0x{csrs[i]:04x}" for i in range(NUM_CSRX)))
    log.info("[PROBE] R: " + " ".join(f"R{i}=0x{r_vals[i]:04x}" for i in range(8)))
    log.info("[PROBE] C: " + " ".join(f"C{i}=0x{c_vals[i]:04x}" for i in range(8)))
    log.info("[PROBE] regS sweep: %s", regs0_values)

    await setup_clocks(dut)
    await reset_dut(dut)

    drive_boundary_zero(dut)
    drive_data_inputs(dut, r_vals, c_vals)
    drive_csrs(dut, csrs)
    dut.regS_i_0.value = 0

    await program_from_bin(dut, bin_path, bit_count)
    log.info("[PROBE] scanchain load complete")

    # Settle once so the fabric's pipelined CSR/REGS reads land.
    drive_csrs(dut, csrs)
    drive_data_inputs(dut, r_vals, c_vals)
    await ClockCycles(dut.clk_i, settle_cycles)

    log.info("[PROBE] tid | "
             "mem_addr0  mem_addr1  mem_addr2  mem_addr3 | "
             "mem_data0  mem_data1  mem_data2  mem_data3 | "
             "ext_data_o[0..7]")
    for tid in regs0_values:
        dut.regS_i_0.value = tid
        await ClockCycles(dut.clk_i, settle_cycles)
        snap = snapshot(dut)
        log.info(
            "[PROBE] %3d | %04x %04x %04x %04x | %04x %04x %04x %04x | %s",
            tid,
            snap["mem_addr"][0], snap["mem_addr"][1], snap["mem_addr"][2], snap["mem_addr"][3],
            snap["mem_data"][0], snap["mem_data"][1], snap["mem_data"][2], snap["mem_data"][3],
            " ".join(f"{v:04x}" for v in snap["ext_data"][:8]),
        )

    # Pass unconditionally — this test is a probe, not a checker.
    log.info("[PROBE] done")
