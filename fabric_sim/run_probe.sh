#!/usr/bin/env bash
# Wrapper: run the fabric probe on one of our generated bins.
#
# Usage:
#   ./run_probe.sh gemm gemm_load_A_k0
#       -> probes ../build/gemm/gemm_load_A_k0.bin
#          with the kernel-specific CSR defaults below
#
# Override any CSR / sweep on the command line:
#   CSR_X0=16 CSR_X3=64 REGS0_SWEEP=0..15 ./run_probe.sh gemm gemm_load_A_k0
set -euo pipefail

KERNEL="${1:?first arg = kernel name (e.g. gemm)}"
STAGE="${2:?second arg = stage name (e.g. gemm_load_A_k0)}"

here="$(cd "$(dirname "$0")" && pwd)"
repo="$(cd "$here/.." && pwd)"
bin="$repo/build/$KERNEL/$STAGE.bin"

if [[ ! -f "$bin" ]]; then
  echo "missing $bin -- did you run 'make $KERNEL'?" >&2
  exit 1
fi

# cocotb 2.0+ lives in dora's poetry venv. Use the dora-run wrapper, which
# activates the poetry env + local-tool overrides for us.
DORA_REPO="${DORA_REPO:-/data/amanoj3/dora}"
DORA_RUN="${DORA_RUN:-$DORA_REPO/scripts/dora-run}"
if [[ ! -x "$DORA_RUN" ]]; then
  echo "missing $DORA_RUN -- set DORA_REPO=... or DORA_RUN=..." >&2
  exit 1
fi

# The Modules `module` command is a shell function -- source it if needed
# before loading vcs, so we work whether or not the parent shell has it.
if ! command -v module >/dev/null 2>&1; then
  if [[ -f /usr/share/Modules/init/bash ]]; then
    # shellcheck disable=SC1091
    source /usr/share/Modules/init/bash
  fi
fi
if ! command -v vcs >/dev/null 2>&1; then
  module load vcs >/dev/null 2>&1 || true
fi
if ! command -v vcs >/dev/null 2>&1; then
  echo "vcs not on PATH after 'module load vcs'" >&2
  exit 1
fi

# bitstream_size comes from the dora compiler_arch.json. We hardcode 1074 here
# (matches dora static-build mini_dice/build_nopred); override via env if a
# different arch is in use.
export BITSTREAM_BITS="${BITSTREAM_BITS:-1074}"
export BIN_PATH="$bin"

# Per-kernel CSR defaults that match what the kernel JSON expects for CTA 0.
# Override any of these inline on the command line. See the README for more.
case "$KERNEL" in
  gemm)
    export CSR_X0="${CSR_X0:-16}"   # A_packed base
    export CSR_X1="${CSR_X1:-272}"  # B_packed base
    export CSR_X2="${CSR_X2:-528}"  # C base
    export CSR_X3="${CSR_X3:-64}"   # k=1 stride
    export CSR_X4="${CSR_X4:-128}"  # k=2 stride
    export CSR_X5="${CSR_X5:-192}"  # k=3 stride
    ;;
  nn_cuda)
    export CSR_X0="${CSR_X0:-1}"
    export CSR_X1="${CSR_X1:-64}"
    export CSR_X2="${CSR_X2:-128}"
    export CSR_X3="${CSR_X3:-1}"
    export CSR_X4="${CSR_X4:-100}"
    export CSR_X5="${CSR_X5:-200}"
    ;;
  srad_*)
    export CSR_X0="${CSR_X0:-1}"
    export CSR_X1="${CSR_X1:-128}"
    export CSR_X2="${CSR_X2:-256}"
    export CSR_X3="${CSR_X3:-1}"
    export CSR_X4="${CSR_X4:-1}"
    export CSR_X5="${CSR_X5:-1}"
    ;;
esac

export REGS0_SWEEP="${REGS0_SWEEP:-0..15}"
export SETTLE_CYCLES="${SETTLE_CYCLES:-8}"

echo "[run_probe] vcs:      $(command -v vcs)"
echo "[run_probe] dora-run: $DORA_RUN"
echo "[run_probe] kernel:   $KERNEL / $STAGE"
echo "[run_probe] bin:      $bin ($(stat -c%s "$bin") bytes)"
echo "[run_probe] bits:     $BITSTREAM_BITS"
echo "[run_probe] csrs:     " \
  "csrX0=$CSR_X0 csrX1=${CSR_X1:-0} csrX2=${CSR_X2:-0}" \
  "csrX3=${CSR_X3:-0} csrX4=${CSR_X4:-0} csrX5=${CSR_X5:-0}"
echo "[run_probe] sweep:    REGS0=$REGS0_SWEEP, settle=$SETTLE_CYCLES"

cd "$here"
exec "$DORA_RUN" python "$here/sim_run.py"
