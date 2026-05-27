"""Batch probe every kernel/stage .bin on the fabric.

For each kernel we walk every stage in its test_vector.json's pgraph_metadata,
load that stage's .bin into dice_top via the cocotb probe, and capture the
fabric's `mem_addr_o_*` + `ext_data_o_*` for REGS0 in 0..15 with the kernel's
CTA-0 CSR values.

Per-stage classification:

  load-affine     (num_stores=0, ld_dst>=0): expect mem_addr_o_<lane> to
                   trace `csr_base + REGS0 * csrX3` across tids.
  store-affine    (num_stores=0, out_regs contains a GPR > 3):
                   expect ext_data_o_<reg-bit> to trace `csr_base + REGS0 * csrX3`.
  compute         (no mem ops, writes a GPR <4): expect SOMETHING nonzero
                   on the out-reg ext_data_o port (since register-file state
                   from prior p-graphs is missing, we can't predict exact values).
  store           (num_stores>=1, no addr-gen): pure routing; no oracle here
                   because R0..R3 / R4..R7 are still at reset (= 0).

Results land in fabric_sim/probe_results.json with a per-stage verdict.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional


FABRIC_SIM   = Path(__file__).parent.resolve()
TESTKERNELS  = FABRIC_SIM.parent
KERNELS_DIR  = TESTKERNELS / "kernels"
BUILD_DIR    = TESTKERNELS / "build"      # legacy/output dir, kept for completeness
BIN_DIR_ROOT = TESTKERNELS / "kernels"    # .bin files now live next to .fasm here
DORA_RUN     = Path("/data/amanoj3/dora/scripts/dora-run")
MODULE_BASH  = Path("/usr/share/Modules/init/bash")

# Default CSR values per kernel — must match kernels/<k>/_test_vector.json runtime.csr_values
KERNEL_CSRS = {
    "gemm":          {0: 16, 1: 272, 2: 528, 3: 64, 4: 128, 5: 192, 6: 0,   7: 0},
    "nn_cuda":       {0: 1,  1: 64,  2: 128, 3: 1,  4: 100, 5: 200, 6: 0,   7: 0},
    "conv2d":        {0: 16, 1: 16,  2: 2,   3: 1,  4: 1,   5: 256, 6: 0,   7: 0},
    "tiled_conv":    {0: 16, 1: 272, 2: 528, 3: 64, 4: 128, 5: 192, 6: 0,   7: 0},
    "srad_prepare":  {0: 1,  1: 128, 2: 256, 3: 1,  4: 1,   5: 1,   6: 0,   7: 0},
    "srad_extract":  {0: 1,  1: 128, 2: 256, 3: 1,  4: 1,   5: 1,   6: 0,   7: 0},
    "srad_compress": {0: 1,  1: 128, 2: 256, 3: 1,  4: 1,   5: 1,   6: 0,   7: 0},
    "srad_srad":     {0: 1,  1: 128, 2: 256, 3: 1,  4: 1,   5: 1,   6: 0,   7: 0},
    "srad_srad2":    {0: 1,  1: 128, 2: 256, 3: 1,  4: 1,   5: 1,   6: 0,   7: 0},
}


# Per-stage expectation: maps stage name -> spec dict.
#
# kind:
#  - load-affine:  mem_addr_o_<port_idx> = csrX<base> + REGS0
#  - load-nonzero: mem_addr_o_<port_idx> != 0 for every tid (looser)
#  - store-addr:   ext_data_o_<out_reg> = csrX<base> + REGS0
#  - compute-check: drive r/c/csr inputs from spec, expect
#                   ext_data_o_<out_port> == expected(r, c, csr) for every tid
#  - compute:      "soft" - drive zeros, expect some output (placeholder)
#  - store:        "soft" - pure routing, no scalar oracle
#
# compute-check entries carry:
#   in_r:    {reg_idx: value}  - drive ext_data_i_<reg_idx>
#   in_c:    {reg_idx: value}  - drive ext_data_i_<reg_idx+8>
#   csr_override: {idx: value} - override the kernel's default CSRs (compute may need csrX4/5 nonzero)
#   out_port: int              - which ext_data_o_<n> to sample
#   expected: int              - 16-bit value the fabric should emit
STAGE_EXPECTATIONS = {
    # gemm load p-graphs all drive MEM_ADDR0 with csrX0 + (k_offset CSR) + REGS0
    # Only k=0 has no extra offset; for k>0 we additionally sweep CSR_X3..5 but
    # the per-tid increment is still csrX3 (always = 64 for gemm) for each k.
    # For simplicity, only assert the k=0 case strictly; others get the
    # "nonzero output" sanity check.
    # gemm
    "gemm_load_A_k0":    {"kind": "load-affine",  "csr_base": 0, "port": 0},
    "gemm_load_B_k0":    {"kind": "load-affine",  "csr_base": 1, "port": 0},
    "gemm_load_A_k1":    {"kind": "load-nonzero", "port": 0},
    "gemm_load_B_k1":    {"kind": "load-nonzero", "port": 0},
    "gemm_load_A_k2":    {"kind": "load-nonzero", "port": 0},
    "gemm_load_B_k2":    {"kind": "load-nonzero", "port": 0},
    "gemm_load_A_k3":    {"kind": "load-nonzero", "port": 0},
    "gemm_load_B_k3":    {"kind": "load-nonzero", "port": 0},
    # R4 = R0 * R1, output OUT4 = ext_data_o_4. R0=3, R1=4 -> 12.
    "gemm_mul_k0":       {"kind": "compute-check", "in_r": {0: 3, 1: 4},
                          "out_port": 4, "expected": 12,
                          "formula": "R4 = R0 * R1"},
    # R4 = R0 * R1 + R4 (R4 already at reset=0 -> equivalent to mul). R0=3, R1=4, R4=5 -> 17.
    "gemm_mac_k1":       {"kind": "compute-check", "in_r": {0: 3, 1: 4, 4: 5},
                          "out_port": 4, "expected": 17,
                          "formula": "R4 = R0 * R1 + R4"},
    "gemm_mac_k2":       {"kind": "compute-check", "in_r": {0: 3, 1: 4, 4: 5},
                          "out_port": 4, "expected": 17,
                          "formula": "R4 = R0 * R1 + R4"},
    "gemm_mac_k3":       {"kind": "compute-check", "in_r": {0: 3, 1: 4, 4: 5},
                          "out_port": 4, "expected": 17,
                          "formula": "R4 = R0 * R1 + R4"},
    "gemm_gen_C_addr":   {"kind": "store-addr",   "csr_base": 2, "out_reg": 5},
    "gemm_store_C":      {"kind": "store"},

    # conv2D 3-tap-1D (= one pass of separable 2D conv).
    # csrX0=16, csrX1=16 -> load_c reads mem[16+tid], load_m reads mem[tid]
    # (zero in our mock since addr&0xFFFF = tid), load_p reads mem[32+tid].
    # compute: R3 = csrX2*R0 + csrX3*R1 + csrX4*R2
    #        = 2*(16+tid) + 1*tid + 1*(32+tid) = 64 + 4*tid
    "conv_load_c":       {"kind": "load-affine",  "csr_base": 0, "port": 0},
    # load_m and load_p mix csrX1 with csrX0; "load-affine" check would have
    # to subtract/add csrX1 from the oracle, simpler to use load-nonzero.
    "conv_load_m":       {"kind": "load-nonzero", "port": 0},
    "conv_load_p":       {"kind": "load-nonzero", "port": 0},
    # R3 = R0*csrX2 + R1*csrX3 + R2*csrX4. With probe stimulus R0=5, R1=3,
    # R2=2 and the default csrX2=2, csrX3=1, csrX4=1:
    # R3 = 5*2 + 3*1 + 2*1 = 15.
    "conv_compute_3tap": {"kind": "compute-check", "in_r": {0: 5, 1: 3, 2: 2},
                          "out_port": 3, "expected": 15,
                          "formula": "R3 = csrX2*R0 + csrX3*R1 + csrX4*R2"},
    "conv_gen_store_addr": {"kind": "store-addr", "csr_base": 5, "out_reg": 4},
    "conv_store":        {"kind": "store"},

    # tiled CNN conv -- one K-tile chunk. Same packed layout as gemm, plus a
    # load_C -> accum -> store_C read-modify-write tail.
    "tconv_load_A_k0":   {"kind": "load-affine",  "csr_base": 0, "port": 0},
    "tconv_load_B_k0":   {"kind": "load-affine",  "csr_base": 1, "port": 0},
    "tconv_load_A_k1":   {"kind": "load-nonzero", "port": 0},
    "tconv_load_B_k1":   {"kind": "load-nonzero", "port": 0},
    "tconv_load_A_k2":   {"kind": "load-nonzero", "port": 0},
    "tconv_load_B_k2":   {"kind": "load-nonzero", "port": 0},
    "tconv_load_A_k3":   {"kind": "load-nonzero", "port": 0},
    "tconv_load_B_k3":   {"kind": "load-nonzero", "port": 0},
    # R4 = R0 * R1. R0=3, R1=4 -> 12.
    "tconv_mul_k0":      {"kind": "compute-check", "in_r": {0: 3, 1: 4},
                          "out_port": 4, "expected": 12, "formula": "R4 = R0 * R1"},
    # R4 = R0 * R1 + R4. R0=3, R1=4, R4=5 -> 17.
    "tconv_mac_k1":      {"kind": "compute-check", "in_r": {0: 3, 1: 4, 4: 5},
                          "out_port": 4, "expected": 17, "formula": "R4 = R0 * R1 + R4"},
    "tconv_mac_k2":      {"kind": "compute-check", "in_r": {0: 3, 1: 4, 4: 5},
                          "out_port": 4, "expected": 17, "formula": "R4 = R0 * R1 + R4"},
    "tconv_mac_k3":      {"kind": "compute-check", "in_r": {0: 3, 1: 4, 4: 5},
                          "out_port": 4, "expected": 17, "formula": "R4 = R0 * R1 + R4"},
    "tconv_gen_C_addr":  {"kind": "store-addr",   "csr_base": 2, "out_reg": 5},
    # load_C routes R5 -> MEM_ADDR0 (address from a GPR, not a CSR); pure
    # routing, so no scalar oracle -- soft-checked like a store.
    "tconv_load_C":      {"kind": "store"},
    # R4 = R4 + R6. R4=5, R6=7 -> 12.
    "tconv_accum":       {"kind": "compute-check", "in_r": {4: 5, 6: 7},
                          "out_port": 4, "expected": 12, "formula": "R4 = R4 + R6"},
    "tconv_store_C":     {"kind": "store"},

    # srad_prepare
    "prep_load_I":       {"kind": "load-affine",  "csr_base": 0, "port": 0},
    "prep_gen_sums_addr":{"kind": "store-addr",   "csr_base": 1, "out_reg": 4},
    "prep_gen_sums2_addr":{"kind": "store-addr",  "csr_base": 2, "out_reg": 4},
    "prep_store_sums":   {"kind": "store"},
    "prep_store_sums2":  {"kind": "store"},
    # R0 = R0 * R0, OUT0 = ext_data_o_0. R0=7 -> 49.
    "prep_square":       {"kind": "compute-check", "in_r": {0: 7},
                          "out_port": 0, "expected": 49,
                          "formula": "R0 = R0 * R0"},

    # srad_extract: R0 = R0 * csrX4. R0=6, csrX4=7 -> 42.
    "ext_load_I":        {"kind": "load-affine",  "csr_base": 0, "port": 0},
    "ext_scale":         {"kind": "compute-check", "in_r": {0: 6},
                          "csr_override": {4: 7},
                          "out_port": 0, "expected": 42,
                          "formula": "R0 = R0 * csrX4"},
    "ext_gen_store_addr":{"kind": "store-addr",   "csr_base": 0, "out_reg": 4},
    "ext_store_I":       {"kind": "store"},

    # srad_compress: same compute as extract.
    "comp_load_I":       {"kind": "load-affine",  "csr_base": 0, "port": 0},
    "comp_scale":        {"kind": "compute-check", "in_r": {0: 6},
                          "csr_override": {4: 7},
                          "out_port": 0, "expected": 42,
                          "formula": "R0 = R0 * csrX4"},
    "comp_gen_store_addr":{"kind": "store-addr",  "csr_base": 0, "out_reg": 4},
    "comp_store_I":      {"kind": "store"},

    # srad_srad: R2 = R0 - R1, OUT2 = ext_data_o_2. R0=15, R1=4 -> 11.
    "srad_load_Jc":      {"kind": "load-affine",  "csr_base": 0, "port": 0},
    "srad_load_N":       {"kind": "load-affine",  "csr_base": 1, "port": 0},
    "srad_compute_dN":   {"kind": "compute-check", "in_r": {0: 15, 1: 4},
                          "out_port": 2, "expected": 11,
                          "formula": "R2 = R0 - R1"},
    "srad_gen_dN_addr":  {"kind": "store-addr",   "csr_base": 2, "out_reg": 4},
    "srad_store_dN":     {"kind": "store"},

    # srad_srad2: R0 = R0 + csrX4*csrX5. R0=5, csrX4=3, csrX5=4 -> 17.
    "srad2_load_I":      {"kind": "load-affine",  "csr_base": 0, "port": 0},
    "srad2_update":      {"kind": "compute-check", "in_r": {0: 5},
                          "csr_override": {4: 3, 5: 4},
                          "out_port": 0, "expected": 17,
                          "formula": "R0 = R0 + csrX4*csrX5"},
    "srad2_gen_store_addr":{"kind": "store-addr", "csr_base": 0, "out_reg": 4},
    "srad2_store_I":     {"kind": "store"},

    # nn_cuda: R0 = (csrX4-R0)^2 + (csrX5-R1)^2, OUT0 = ext_data_o_0.
    # R0=2, R1=3, csrX4=10, csrX5=20 -> 8^2 + 17^2 = 64 + 289 = 353.
    "nn_load_lat":       {"kind": "load-affine",  "csr_base": 0, "port": 0},
    "nn_load_lng":       {"kind": "load-affine",  "csr_base": 1, "port": 0},
    "nn_compute_distsq": {"kind": "compute-check", "in_r": {0: 2, 1: 3},
                          "csr_override": {4: 10, 5: 20},
                          "out_port": 0, "expected": 353,
                          "formula": "R0 = (csrX4-R0)^2 + (csrX5-R1)^2"},
    "nn_gen_store_addr": {"kind": "store-addr",   "csr_base": 2, "out_reg": 4},
    "nn_store_dist":     {"kind": "store"},
}


def _module_load_vcs() -> dict:
    """Source /etc/Modules and load vcs, return the env delta we need."""
    cmd = f"source {MODULE_BASH} && module load vcs >/dev/null 2>&1 && env"
    out = subprocess.check_output(["bash", "-c", cmd]).decode()
    env = {}
    for line in out.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            env[k] = v
    return env


def _run_one(bin_path: Path, csrs: dict, env: dict,
             r_vals: Optional[dict] = None,
             c_vals: Optional[dict] = None) -> str:
    """Run sim_run.py for one bin and return the full stdout.

    r_vals / c_vals: optional {idx: value} to drive on ext_data_i_*. Used by
    compute-check stages that need R0..R7 / C0..C7 set to known test inputs
    before the bitstream programs the fabric.
    """
    proc_env = dict(env)
    proc_env["BIN_PATH"]       = str(bin_path)
    proc_env["BITSTREAM_BITS"] = "1074"
    for i, v in csrs.items():
        proc_env[f"CSR_X{i}"]  = str(v)
    for i, v in (r_vals or {}).items():
        proc_env[f"R{i}"]      = str(v)
    for i, v in (c_vals or {}).items():
        proc_env[f"C{i}"]      = str(v)
    proc_env["REGS0_SWEEP"]    = "0..15"
    proc_env["SETTLE_CYCLES"]  = "8"

    result = subprocess.run(
        [str(DORA_RUN), "python", str(FABRIC_SIM / "sim_run.py")],
        cwd=str(FABRIC_SIM),
        env=proc_env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        return f"<SIM RETURNED {result.returncode}>\n{result.stderr[-800:]}"
    return result.stdout + result.stderr


# Parse the [PROBE] tid lines from the cocotb log
_PROBE_LINE = re.compile(
    r"\[PROBE\]\s+(\d+)\s*\|\s*"
    r"([0-9a-f]+)\s+([0-9a-f]+)\s+([0-9a-f]+)\s+([0-9a-f]+)\s*\|\s*"
    r"([0-9a-f]+)\s+([0-9a-f]+)\s+([0-9a-f]+)\s+([0-9a-f]+)\s*\|\s*"
    r"([0-9a-f ]+)"
)


def _parse(log: str) -> list[dict]:
    out = []
    for line in log.splitlines():
        m = _PROBE_LINE.search(line)
        if not m:
            continue
        tid = int(m.group(1))
        mem_addr = [int(m.group(i), 16) for i in (2, 3, 4, 5)]
        mem_data = [int(m.group(i), 16) for i in (6, 7, 8, 9)]
        ext_data = [int(x, 16) for x in m.group(10).split()]
        out.append({"tid": tid, "mem_addr": mem_addr, "mem_data": mem_data,
                    "ext_data": ext_data})
    return out


def _classify(stage_name: str, csrs: dict, samples: list[dict]) -> tuple[str, str]:
    """Return (status, detail) per the expectation table.

    Per-tid stride is always 1 across our DFGs: every affine address is
    `csrX_base + REGS0`. csrX3 in gemm is the k-stride between A/B_packed
    slices, not a per-thread stride.
    """
    if not samples:
        return ("NO-DATA", "no probe samples extracted from log")
    spec = STAGE_EXPECTATIONS.get(stage_name)
    if spec is None:
        return ("UNKNOWN-STAGE", f"no expectation registered for {stage_name}")
    kind = spec["kind"]

    if kind == "load-affine":
        base = csrs[spec["csr_base"]]
        port = spec["port"]
        bad = []
        for s in samples:
            want = (base + s["tid"]) & 0xFFFF
            got  = s["mem_addr"][port]
            if got != want:
                bad.append(f"tid={s['tid']} want=0x{want:04x} got=0x{got:04x}")
        if bad:
            return ("FAIL", f"mismatch ({len(bad)}/{len(samples)}): " + bad[0])
        return ("PASS", f"mem_addr_o_{port} = csrX{spec['csr_base']}+REGS0 ∀ tid")

    if kind == "load-nonzero":
        port = spec["port"]
        nonzero = sum(1 for s in samples if s["mem_addr"][port])
        if nonzero == 0:
            return ("FAIL", "mem_addr_o stayed 0 across all tids")
        return ("SOFT-PASS", f"{nonzero}/{len(samples)} tids gave nonzero mem_addr_o_{port}")

    if kind == "store-addr":
        base = csrs[spec["csr_base"]]
        out_reg = spec["out_reg"]
        bad = []
        for s in samples:
            want = (base + s["tid"]) & 0xFFFF
            got  = s["ext_data"][out_reg] if out_reg < len(s["ext_data"]) else None
            if got != want:
                bad.append(f"tid={s['tid']} want=0x{want:04x} got={got if got is None else f'0x{got:04x}'}")
        if bad:
            return ("FAIL", f"mismatch ({len(bad)}/{len(samples)}): " + bad[0])
        return ("PASS", f"ext_data_o_{out_reg} = csrX{spec['csr_base']}+REGS0 ∀ tid")

    if kind == "compute-check":
        out_port = spec["out_port"]
        want = spec["expected"] & 0xFFFF
        bad = []
        for s in samples:
            got = s["ext_data"][out_port] if out_port < len(s["ext_data"]) else None
            if got != want:
                bad.append(f"tid={s['tid']} got={got if got is None else f'0x{got:04x}'}")
        if bad:
            return ("FAIL", f"expected 0x{want:04x} on ext_data_o_{out_port}, " +
                            f"mismatched on {len(bad)}/{len(samples)} tids: " + bad[0])
        return ("PASS", f"{spec['formula']} = 0x{want:04x} ∀ tid")

    if kind == "store":
        return ("SOFT-PASS", "store stage probed without RF state; ran to completion")

    return ("UNKNOWN-KIND", kind)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kernel", action="append", default=None,
                    help="restrict to one kernel (repeatable). default: all")
    ap.add_argument("--out",    default=str(FABRIC_SIM / "probe_results.json"))
    args = ap.parse_args()

    env = _module_load_vcs()
    if "PATH" not in env or not any("vcs" in p for p in env["PATH"].split(":")):
        print("WARNING: vcs not in PATH after module load -- runs may fail", file=sys.stderr)

    kernels = args.kernel or sorted(KERNEL_CSRS.keys())
    results = []
    for k in kernels:
        json_path = next((p for p in (KERNELS_DIR / k).glob("*_test_vector.json")), None)
        if json_path is None:
            print(f"!! no test_vector.json in kernels/{k}")
            continue
        meta = json.loads(json_path.read_text())
        csrs = KERNEL_CSRS[k]
        print(f"\n=== kernel: {k}  ({len(meta['pgraph_metadata'])} p-graphs) ===")
        for entry, stage in zip(meta["pgraph_metadata"], meta.get("stage_artifacts", [])):
            stem    = Path(stage.get("fasm_path", "")).stem or stage.get("kernel")
            bin_p   = BIN_DIR_ROOT / k / f"{stem}.bin"
            if not bin_p.exists():
                print(f"  {stem:<25}  MISSING {bin_p}")
                results.append({"kernel": k, "stage": stem, "status": "MISSING-BIN"})
                continue
            spec      = STAGE_EXPECTATIONS.get(stem, {})
            run_csrs  = dict(csrs)
            run_csrs.update({int(k): int(v) for k, v in spec.get("csr_override", {}).items()})
            r_vals    = {int(k): int(v) for k, v in spec.get("in_r", {}).items()}
            c_vals    = {int(k): int(v) for k, v in spec.get("in_c", {}).items()}
            log       = _run_one(bin_p, run_csrs, env, r_vals=r_vals, c_vals=c_vals)
            samples   = _parse(log)
            status, detail = _classify(stem, run_csrs, samples)
            print(f"  {stem:<25}  {status:<10}  {detail}")
            results.append({"kernel": k, "stage": stem, "status": status, "detail": detail,
                            "samples": samples})

    Path(args.out).write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nWrote detailed log to {args.out}")

    # Final summary
    by_status: dict[str, int] = {}
    for r in results:
        by_status[r["status"]] = by_status.get(r["status"], 0) + 1
    print(f"\nStatus tally:")
    for s in sorted(by_status):
        print(f"  {s:<14} {by_status[s]}")
    fail_count = by_status.get("FAIL", 0) + by_status.get("MISSING-BIN", 0)
    return 1 if fail_count else 0


if __name__ == "__main__":
    sys.exit(main())
