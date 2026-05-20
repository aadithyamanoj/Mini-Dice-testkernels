# Mini-Dice-testkernels

Self-contained test kernels for the Mini_Dice testbench. Each kernel is a
chain of p-graphs produced by CGRA-Solve targeting the dora
`mini_dice/static-build` arch; the Makefile turns the FASMs + JSON metadata
into the `.mem` files Mini_Dice's TB consumes.

## Layout

```
Mini-Dice-testkernels/
├── Makefile
├── README.md
├── kernels/
│   ├── srad_prepare/   srad_extract/  srad_compress/  srad_srad/  srad_srad2/
│   ├── nn_cuda/
│   └── gemm/
│       ├── *.fasm                       (per-p-graph CGRA-Solve output)
│       └── *_test_vector.json           (kernel descriptor + pgraph metadata)
├── scripts/
│   ├── fasm_to_bin.py            (dora bitgen wrapper: fasm -> .bin)
│   ├── bins_to_bitstream_mem.py  (combine per-p-graph .bin -> bitstream.mem)
│   ├── json_to_meta_mem.py       (JSON -> meta.mem + cta_desc.mem + runtime.json)
│   └── fill_expected_writes.py   (per-kernel sim -> populate axi.expected_writes)
├── patches/                      (ready-to-copy multi-CTA support for Mini_Dice TB)
│   ├── dpi_dice_core_runtime.cpp  -> Mini_Dice/tb/cgra_core/dice_core/
│   └── tb_chip_top.sv             -> Mini_Dice/tb/mini_dice/
└── build/                        (created by `make`)
    └── <kernel>/
        ├── *.bin                          (one per p-graph FASM)
        ├── <kernel>_bitstream.mem         (sequential pgraph bitstreams)
        ├── <kernel>_meta.mem              (sequential pgraph_meta_t entries)
        ├── <kernel>_cta_desc.mem          (single dice_cta_desc_t @0)
        └── <kernel>_runtime.json          (csr_values + per_cta overrides + axi)
```

## Kernels

| Folder | P-graphs | Description |
|---|---|---|
| `srad_prepare`  | 6  | rodinia SRAD prepare: `d_sums[i]=d_I[i]; d_sums2[i]=d_I[i]²` |
| `srad_extract`  | 4  | SRAD extract — `exp(x/255)` stubbed as `x * csrX4` (INT16) |
| `srad_compress` | 4  | SRAD compress — `log(x)*255` stubbed as `x * csrX4` (INT16) |
| `srad_srad`     | 5  | SRAD main step (one direction; full version needs indirect loads) |
| `srad_srad2`    | 4  | SRAD2 update — divergence pre-scalarized into CSRs by host |
| `nn_cuda`       | 5  | rodinia kNN per-thread Euclidean distance (sqrt dropped) |
| `gemm`          | 14 | INT16 GEMM with K=4 unrolled (`C[g] = sum_k A_packed[k,g]*B_packed[k,g]`) |

All kernels use the **one-thread-per-element** convention (no implicit lane
unrolling, at most one memory port active per p-graph) and the
**single-resident-CTA + per-CTA CSR re-staging** pattern documented in each
JSON's `runtime._notes`.

## Build

```bash
make            # build every kernel
make <kernel>   # build one (e.g. make gemm)
make clean      # rm -rf build/
make list       # show discovered kernels
make fill-writes # re-simulate every kernel and refresh axi.expected_writes in
                 # each kernels/<k>/*_test_vector.json (golden reference)
```

Overrides:

```bash
make DORA_REPO=/some/other/dora           # alternate dora checkout
make WORKSPACE_PKL=/path/to/workspace.pkl # alternate dora workspace pickle
make PYTHON=python3.12                    # alternate interpreter
```

## What each script does

### `scripts/fasm_to_bin.py`

Wraps dora's `GenerateBitstreamFromFasmStep`. One invocation per FASM:

```bash
python3 scripts/fasm_to_bin.py \
    --fasm   kernels/gemm/gemm_mul_k0.fasm \
    --out    build/gemm/gemm_mul_k0.bin \
    --workspace /data/amanoj3/dora/.../static-build/workspace.pkl
```

Outputs a raw binary bitstream (the dora bitgen native format). Bitstream
byte count matches dora's reference for the same arch.

### `scripts/bins_to_bitstream_mem.py`

Combines per-p-graph `.bin` files into one `$readmemh` file matching
`Mini_Dice/tb/test_vectors/*_bitstream.mem` format (32-bit words at
sequential addresses, p-graph base addresses from JSON `bitstream_addr`).

```bash
python3 scripts/bins_to_bitstream_mem.py \
    --json     kernels/gemm/gemm_test_vector.json \
    --bins-dir build/gemm \
    --out      build/gemm/gemm_bitstream.mem
```

### `scripts/fill_expected_writes.py`

Per-kernel simulator that mirrors dora's `_axi_read_mock` (memory reads
return `addr & 0xFFFF`) and walks every `(cta_idx, tid)` pair under the
kernel's per-CTA CSR-override table. Writes the predicted AXI traffic into
`runtime.axi.expected_writes` of each `kernels/<k>/*_test_vector.json` so
the testbench has a golden reference per the same format used by
`Mini_Dice/tb/test_vectors/full_mul_array_test_vector.json`.

```bash
python3 scripts/fill_expected_writes.py --all
python3 scripts/fill_expected_writes.py --kernel gemm
```

Write counts (4 CTAs × 16 threads = 64 elements per kernel):

| Kernel        | Writes | What's verified |
|---|---|---|
| `srad_prepare`  | 128 | `d_sums[i] = d_I[i]`, `d_sums2[i] = d_I[i]²` |
| `srad_extract`  |  64 | `d_I[i] = d_I[i] * csrX4` |
| `srad_compress` |  64 | `d_I[i] = d_I[i] * csrX4` |
| `srad_srad`     |  64 | `d_dN[i] = Jc[i] - north[i]` |
| `srad_srad2`    |  64 | `d_I[i] += csrX4 * csrX5` |
| `nn_cuda`       |  64 | `dist[i] = (csrX4-lat[i])² + (csrX5-lng[i])²` |
| `gemm`          |  64 | `C[g] = Σ_k A_packed[k,g] * B_packed[k,g]` |

All arithmetic uses INT16 modular semantics (`& 0xFFFF`). Re-run after any
CSR layout, p-graph semantics, or grid_size change.

### `scripts/json_to_meta_mem.py`

Packs the test_vector.json into three Mini_Dice TB inputs:
- `<stem>_meta.mem` — sequential `pgraph_meta_t` entries (one per `pc`)
- `<stem>_cta_desc.mem` — single `dice_cta_desc_t` at address 0
- `<stem>_runtime.json` — CSR launch values + per-CTA overrides + AXI expectations

Widths are auto-loaded from `/data/amanoj3/Mini_Dice/rtl/includes/dice_config.vh`
when present, else from the hardcoded defaults that match that file as of
2026-05-20. Output widths verified against the upstream reference: 102-bit
`pgraph_meta_t`, 120-bit `dice_cta_desc_t`, 2048-bit metadata memory width.

## Multi-CTA serialization

Every kernel JSON declares `grid_size.x = 4` (4 serial CTAs × 16 threads). The
core does not need cta_id at runtime — instead the FPGA controller reprograms
array-base CSRs between dispatches via `runtime.per_cta_csr_overrides` in the
runtime JSON. The kernel reads `REGS0 = local_tid (0..15)` every CTA.

## Provenance

Generated by CGRA-Solve mappers (`scripts/srad/`, `scripts/nn_cuda/`,
`scripts/gemm/` in the parent repo) targeting
`benchmarks/architectures/mini_dice_static/`. The FASMs' `dora_layout_hash`
header matches dora's reference (`ca30464…7897ab`), confirming they target
the same arch binary that the static-build's reference FASMs do.
