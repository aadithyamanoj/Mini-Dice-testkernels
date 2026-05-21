# Mini-Dice-testkernels build
#
# Layout:
#   kernels/<k>/*.fasm                       human-readable source (CGRA-Solve output)
#   kernels/<k>/*.bin                        prebuilt bitstreams (committed alongside)
#   kernels/<k>/*_test_vector.json           kernel descriptor + pgraph metadata
#
# Build outputs (under build/<k>/):
#   <k>_bitstream.mem    concatenated per-pgraph bins ($readmemh, 32-bit words)
#   <k>_meta.mem         sequential pgraph_meta_t entries
#   <k>_cta_desc.mem     single dice_cta_desc_t at @0
#   <k>_runtime.json     csr_values + per_cta_csr_overrides + axi.expected_writes
#
# Default `make all` only needs stdlib Python -- no dora, no VCS. It consumes
# the committed .bin files directly. If you want to regenerate the .bin from
# the .fasm (e.g. after editing the FASM), run `make regen-bins`; that target
# DOES require dora (via scripts/dora-run python) and a workspace.pkl.
#
# Targets:
#   make / make all       build all .mem files (no dora needed)
#   make <kernel>         build only that kernel
#   make list             list discoverable kernels
#   make fill-writes      re-simulate every kernel and refresh axi.expected_writes
#   make regen-bins       regenerate kernels/<k>/*.bin from *.fasm via dora bitgen
#   make clean            rm -rf build/

PYTHON          ?= python3
DORA_REPO       ?= /data/amanoj3/dora
DORA_STATIC_DIR ?= $(DORA_REPO)/examples/devices/dice-isca/mini_dice/static-build
WORKSPACE_PKL   ?= $(DORA_STATIC_DIR)/workspace.pkl
DORA_RUN        ?= $(DORA_REPO)/scripts/dora-run

BUILD_DIR     := build
SCRIPTS_DIR   := scripts
KERNELS_DIR   := kernels

# Default-flow scripts: plain stdlib Python, run via $(PYTHON).
BINS_TO_MEM     := $(PYTHON) $(SCRIPTS_DIR)/bins_to_bitstream_mem.py
JSON_TO_META    := $(PYTHON) $(SCRIPTS_DIR)/json_to_meta_mem.py
FILL_WRITES     := $(PYTHON) $(SCRIPTS_DIR)/fill_expected_writes.py

# Bitgen needs dora's poetry env (cocotb/dora deps). Only used by `regen-bins`.
FASM_TO_BIN_DORA := $(DORA_RUN) python $(CURDIR)/$(SCRIPTS_DIR)/fasm_to_bin.py

# Discover every subdirectory under kernels/ that has at least one *_test_vector.json
KERNELS := $(notdir $(patsubst %/,%, $(sort $(dir $(wildcard $(KERNELS_DIR)/*/*_test_vector.json)))))

# Per-kernel output filenames (stem == kernel folder name)
define KERNEL_OUTPUTS
$(BUILD_DIR)/$(1)/$(1)_bitstream.mem \
$(BUILD_DIR)/$(1)/$(1)_meta.mem \
$(BUILD_DIR)/$(1)/$(1)_cta_desc.mem \
$(BUILD_DIR)/$(1)/$(1)_runtime.json
endef

ALL_OUTPUTS := $(foreach k,$(KERNELS),$(call KERNEL_OUTPUTS,$(k)))

.PHONY: all clean list fill-writes regen-bins $(KERNELS)
.DELETE_ON_ERROR:

# `make all` repopulates axi.expected_writes (for ALL CTAs in the grid)
# before building any .mem. This guards against a stale or empty
# axi.expected_writes -- e.g. after a CGRA-Solve mapper regen wiped the
# JSON's runtime block -- silently producing a TB that "passes" after
# only CTA 0's writes. fill-writes is pure-stdlib Python and idempotent;
# if kernel semantics haven't changed the JSON content stays byte-identical.
all: fill-writes $(ALL_OUTPUTS)

list:
	@echo "Discovered kernels:"
	@for k in $(KERNELS); do echo "  - $$k"; done

# Re-simulate every kernel and overwrite axi.expected_writes in each
# kernels/<k>/*_test_vector.json.
fill-writes:
	$(FILL_WRITES) --all

clean:
	rm -rf $(BUILD_DIR)

# ---------------------------------------------------------------------------
# Optional: regenerate the committed .bin files from .fasm via dora bitgen.
# Requires DORA_REPO to be checked out and the workspace pickle to exist.
# Only need to run this after editing FASMs (e.g. re-mapping in CGRA-Solve).
# ---------------------------------------------------------------------------
ALL_FASMS := $(foreach k,$(KERNELS),$(wildcard $(KERNELS_DIR)/$(k)/*.fasm))
ALL_BINS  := $(patsubst $(KERNELS_DIR)/%.fasm,$(KERNELS_DIR)/%.bin,$(ALL_FASMS))

regen-bins: $(ALL_BINS)

# Pattern rule: bin sits next to its fasm in kernels/<k>/. Requires dora.
$(KERNELS_DIR)/%.bin: $(KERNELS_DIR)/%.fasm
	@test -x $(DORA_RUN) || { echo "ERROR: $(DORA_RUN) not found -- set DORA_REPO=..." >&2; exit 1; }
	@test -f $(WORKSPACE_PKL) || { echo "ERROR: workspace.pkl not found at $(WORKSPACE_PKL)" >&2; exit 1; }
	$(FASM_TO_BIN_DORA) --fasm $(CURDIR)/$< --workspace $(WORKSPACE_PKL) --out $(CURDIR)/$@

# ---------------------------------------------------------------------------
# Per-kernel build rules (default flow, no dora needed)
# ---------------------------------------------------------------------------
define KERNEL_template

$(1): $(call KERNEL_OUTPUTS,$(1))

BINS_$(1)  := $$(wildcard $(KERNELS_DIR)/$(1)/*.bin)
JSON_$(1)  := $$(firstword $$(wildcard $(KERNELS_DIR)/$(1)/*_test_vector.json))

$(BUILD_DIR)/$(1)/$(1)_bitstream.mem: $$(BINS_$(1)) $$(JSON_$(1))
	@mkdir -p $$(@D)
	$(BINS_TO_MEM) --json $$(JSON_$(1)) --bins-dir $(KERNELS_DIR)/$(1) --out $$@

$(BUILD_DIR)/$(1)/$(1)_meta.mem $(BUILD_DIR)/$(1)/$(1)_cta_desc.mem $(BUILD_DIR)/$(1)/$(1)_runtime.json: $$(JSON_$(1))
	@mkdir -p $(BUILD_DIR)/$(1)
	$(JSON_TO_META) --json $$(JSON_$(1)) --out-dir $(BUILD_DIR)/$(1) --stem $(1)

endef

$(foreach k,$(KERNELS),$(eval $(call KERNEL_template,$(k))))
