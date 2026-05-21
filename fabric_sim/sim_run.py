"""VCS + cocotb runner for mini_dice fabric isolation tests.

Compiles dora's static-build RTL (dice_top + the whole fabric, no FPGA-side IO
or bsg_link), loads a .bin bitstream of our choice into the scanchain, drives
boundary inputs, and samples mem_addr_o / ext_data_o to see what the fabric
actually emits. Lets us isolate "is the bitstream wrong?" from "is the chip
TB / DPI / dispatcher broken?" without needing the full chip_top sim.

Env vars consumed by tb/test_bin_probe.py:
  BIN_PATH           absolute path to the .bin to load (required)
  BITSTREAM_BITS     bit count from compiler_arch.bitstream_size (required)
  CSR_X0..CSR_X7     hex/dec values to drive on csrX_i_*  (default 0)
  REGS0_SWEEP        "lo..hi" inclusive (default "0..0")
  SETTLE_CYCLES      cycles after each input change (default 8)
"""
from __future__ import annotations

import os
from pathlib import Path

from cocotb_tools.runner import get_runner


STATIC_BUILD_RTL = Path(
    "/data/amanoj3/dora/examples/devices/dice-isca/mini_dice/static-build/rtl"
)
# bsg_defines.sv lives in BaseJump STL inside dora's external/.
BSG_MISC     = Path("/data/amanoj3/dora/dora.py/external/basejump_stl/bsg_misc")
BSG_MEM      = Path("/data/amanoj3/dora/dora.py/external/basejump_stl/bsg_mem")
BSG_DATAFLOW = Path("/data/amanoj3/dora/dora.py/external/basejump_stl/bsg_dataflow")


def _gather_sources(rtl_dir: Path) -> list[str]:
    files = sorted(rtl_dir.glob("*.sv"))
    files += sorted((rtl_dir / "alu").glob("*.sv"))
    return [str(p) for p in files]


def main() -> None:
    tests_dir = Path(__file__).parent.resolve()
    rtl_dir   = STATIC_BUILD_RTL
    if not rtl_dir.exists():
        raise SystemExit(f"static-build RTL not found at {rtl_dir}")

    verilog_sources = _gather_sources(rtl_dir)
    if not verilog_sources:
        raise SystemExit(f"No .sv files under {rtl_dir}")

    runner = get_runner("vcs")

    build_args = [
        "-sverilog",
        "+lint=none",
        "-timescale=1ns/1ps",
        "+libext+.sv+.v",
        f"+incdir+{rtl_dir}",
        f"+incdir+{rtl_dir}/alu",
        # BaseJump STL include + library search (resolves `include "bsg_defines.sv"`
        # and on-demand bsg module compilation the same way dora's own sim_build does).
        f"+incdir+{BSG_MISC}",
        f"-y", str(BSG_MISC),
        f"-y", str(BSG_MEM),
        f"-y", str(BSG_DATAFLOW),
    ]

    build_dir = tests_dir / "sim_build_vcs"
    runner.build(
        verilog_sources=verilog_sources,
        hdl_toplevel="dice_top",
        build_dir=str(build_dir),
        build_args=build_args,
        always=True,
    )

    extra_env = {
        "PYTHONPATH": (
            f"{tests_dir}:{tests_dir / 'tb'}:{os.environ.get('PYTHONPATH', '')}"
        ),
    }
    for k in ("BIN_PATH", "BITSTREAM_BITS",
              "CSR_X0", "CSR_X1", "CSR_X2", "CSR_X3",
              "CSR_X4", "CSR_X5", "CSR_X6", "CSR_X7",
              "REGS0_SWEEP", "SETTLE_CYCLES"):
        if k in os.environ:
            extra_env[k] = os.environ[k]

    runner.test(
        hdl_toplevel="dice_top",
        test_module="tb.test_bin_probe",
        build_dir=str(build_dir),
        test_dir=str(tests_dir),
        extra_env=extra_env,
    )


if __name__ == "__main__":
    main()
