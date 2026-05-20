"""fill_expected_writes.py -- compute AXI expected_writes for each kernel.

Mini_Dice's TB drives memory reads through an AXI read mock that returns
`addr & 0xFFFF` (mirrors `_axi_read_mock` in
Mini_Dice/tb/test_vectors/gen_fdr_metadata.py). Given that model + each
kernel's CSR layout + per-CTA overrides, we can deterministically simulate
each thread's writes and populate `runtime.axi.expected_writes` so the TB has
a golden reference to check against.

Updates each kernel's test_vector.json IN PLACE.

Usage:
    python3 scripts/fill_expected_writes.py --all
    python3 scripts/fill_expected_writes.py --kernel srad_prepare
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

DATA_MASK = 0xFFFF
NUM_THREADS = 16
STRB_FULL = 3  # 2 bytes = b11

KERNELS_DIR = Path(__file__).resolve().parent.parent / "kernels"


def axi_read(addr: int) -> int:
    """Mock memory: byte address -> 16-bit value = addr & 0xFFFF."""
    return addr & DATA_MASK


def effective_csrs(initial: dict[str, int],
                   overrides: list[dict],
                   cta_idx: int) -> dict[str, int]:
    """Merge per-CTA overrides on top of the initial CSR launch values."""
    csr = {k: int(v) for k, v in initial.items()}
    if cta_idx < len(overrides):
        for k, v in overrides[cta_idx]["csr_values"].items():
            csr[k] = int(v)
    return csr


# ---------------------------------------------------------------------------
# Per-kernel models
# ---------------------------------------------------------------------------
# Each model takes (csr_initial, overrides, num_ctas) and returns a list of
# {"addr", "data", "strb"} dicts representing the expected AXI writes for the
# whole kernel grid in dispatch order.
# ---------------------------------------------------------------------------

def model_srad_prepare(csr0, ov, num_ctas):
    """
    d_sums[i]  = d_I[i]
    d_sums2[i] = d_I[i]^2
    addr_load = csrX0 + tid*csrX3       (d_I)
    addr_s    = csrX1 + tid*csrX3       (d_sums)
    addr_s2   = csrX2 + tid*csrX3       (d_sums2)
    """
    writes = []
    for cta in range(num_ctas):
        c = effective_csrs(csr0, ov, cta)
        for tid in range(NUM_THREADS):
            load_a = c["csrX0"] + tid * c["csrX3"]
            v = axi_read(load_a)
            writes.append({"addr": c["csrX1"] + tid * c["csrX3"],
                           "data": v, "strb": STRB_FULL})
            writes.append({"addr": c["csrX2"] + tid * c["csrX3"],
                           "data": (v * v) & DATA_MASK, "strb": STRB_FULL})
    return writes


def model_srad_extract(csr0, ov, num_ctas):
    """d_I[i] = (d_I[i] * csrX4) & 0xFFFF   (INT16 stand-in for exp(x/255))"""
    writes = []
    for cta in range(num_ctas):
        c = effective_csrs(csr0, ov, cta)
        for tid in range(NUM_THREADS):
            a = c["csrX0"] + tid * c["csrX3"]
            v = axi_read(a)
            new = (v * c["csrX4"]) & DATA_MASK
            writes.append({"addr": a, "data": new, "strb": STRB_FULL})
    return writes


def model_srad_compress(csr0, ov, num_ctas):
    """Same compute shape as extract (multiply by csrX4 in-place)."""
    return model_srad_extract(csr0, ov, num_ctas)


def model_srad_srad(csr0, ov, num_ctas):
    """dN_loc[i] = Jc[i] - north[i];   d_dN[i] = dN_loc[i]
    Jc:    mem[csrX0 + tid*csrX3]
    north: mem[csrX1 + tid*csrX3]   (host-staged north-shifted view)
    store: mem[csrX2 + tid*csrX3]
    """
    writes = []
    for cta in range(num_ctas):
        c = effective_csrs(csr0, ov, cta)
        for tid in range(NUM_THREADS):
            jc    = axi_read(c["csrX0"] + tid * c["csrX3"])
            north = axi_read(c["csrX1"] + tid * c["csrX3"])
            dn = (jc - north) & DATA_MASK
            writes.append({"addr": c["csrX2"] + tid * c["csrX3"],
                           "data": dn, "strb": STRB_FULL})
    return writes


def model_srad_srad2(csr0, ov, num_ctas):
    """d_I[i] = d_I[i] + csrX4 * csrX5  (lambda*0.25 * D, host pre-scalarized)."""
    writes = []
    for cta in range(num_ctas):
        c = effective_csrs(csr0, ov, cta)
        product = (c["csrX4"] * c["csrX5"]) & DATA_MASK
        for tid in range(NUM_THREADS):
            a = c["csrX0"] + tid * c["csrX3"]
            v = axi_read(a)
            new = (v + product) & DATA_MASK
            writes.append({"addr": a, "data": new, "strb": STRB_FULL})
    return writes


def model_nn_cuda(csr0, ov, num_ctas):
    """dist[i] = (csrX4 - lat[i])^2 + (csrX5 - lng[i])^2 (squared Euclidean)."""
    writes = []
    for cta in range(num_ctas):
        c = effective_csrs(csr0, ov, cta)
        for tid in range(NUM_THREADS):
            lat = axi_read(c["csrX0"] + tid * c["csrX3"])
            lng = axi_read(c["csrX1"] + tid * c["csrX3"])
            dlat = (c["csrX4"] - lat) & DATA_MASK
            dlng = (c["csrX5"] - lng) & DATA_MASK
            dist = (dlat * dlat + dlng * dlng) & DATA_MASK
            writes.append({"addr": c["csrX2"] + tid * c["csrX3"],
                           "data": dist, "strb": STRB_FULL})
    return writes


def model_gemm(csr0, ov, num_ctas):
    """
    C[g] = sum_{k=0..3} A_packed[k,g] * B_packed[k,g]
    A_packed[k,g] = mem[csrX0_cta + k_offset(k) + tid]  (k_offset(0)=0, (1)=csrX3, (2)=csrX4, (3)=csrX5)
    B_packed[k,g] = mem[csrX1_cta + k_offset(k) + tid]
    store C[g]    = mem[csrX2_cta + tid]
    """
    writes = []
    for cta in range(num_ctas):
        c = effective_csrs(csr0, ov, cta)
        k_offset = [0, c["csrX3"], c["csrX4"], c["csrX5"]]
        for tid in range(NUM_THREADS):
            acc = 0
            for k in range(4):
                a_addr = c["csrX0"] + k_offset[k] + tid
                b_addr = c["csrX1"] + k_offset[k] + tid
                a_val = axi_read(a_addr)
                b_val = axi_read(b_addr)
                acc = (acc + a_val * b_val) & DATA_MASK
            writes.append({"addr": c["csrX2"] + tid, "data": acc, "strb": STRB_FULL})
    return writes


KERNEL_MODELS: dict[str, Callable] = {
    "srad_prepare":  model_srad_prepare,
    "srad_extract":  model_srad_extract,
    "srad_compress": model_srad_compress,
    "srad_srad":     model_srad_srad,
    "srad_srad2":    model_srad_srad2,
    "nn_cuda":       model_nn_cuda,
    "gemm":          model_gemm,
}


def _kernel_json(kdir: Path) -> Path:
    matches = sorted(kdir.glob("*_test_vector.json"))
    if not matches:
        raise SystemExit(f"no *_test_vector.json under {kdir}")
    return matches[0]


def fill_one(kernel_name: str) -> None:
    kdir = KERNELS_DIR / kernel_name
    if not kdir.exists():
        raise SystemExit(f"unknown kernel: {kernel_name} (no dir {kdir})")
    if kernel_name not in KERNEL_MODELS:
        raise SystemExit(f"no simulator registered for {kernel_name}")
    jp = _kernel_json(kdir)
    data = json.loads(jp.read_text())
    runtime = data.setdefault("runtime", {})
    csr0 = runtime.get("csr_values") or {}
    overrides = runtime.get("per_cta_csr_overrides") or []
    grid_x = int(data["dice_cta_desc"]["kernel_desc"]["grid_size"]["x"])
    num_ctas = max(grid_x, len(overrides), 1)
    writes = KERNEL_MODELS[kernel_name](csr0, overrides, num_ctas)
    runtime.setdefault("axi", {})["expected_writes"] = writes
    jp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    print(f"  {kernel_name:<14} -> {len(writes)} expected writes  ({jp})")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--all", action="store_true", help="process every known kernel")
    g.add_argument("--kernel", choices=sorted(KERNEL_MODELS.keys()),
                   help="process just one kernel")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if args.all:
        for k in sorted(KERNEL_MODELS.keys()):
            fill_one(k)
    else:
        fill_one(args.kernel)


if __name__ == "__main__":
    main()
