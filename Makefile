# Mini-Dice-testkernels build
#
# For each kernel under kernels/<k>/, this Makefile:
#   1. Runs dora bitgen on every <stage>.fasm -> build/<k>/<stage>.bin
#   2. Combines the bins into build/<k>/<k>_bitstream.mem ($readmemh, 32-bit)
#   3. Packs the test_vector.json into:
#        build/<k>/<k>_meta.mem      (sequential pgraph_meta_t entries)
#        build/<k>/<k>_cta_desc.mem  (single dice_cta_desc_t at @0)
#        build/<k>/<k>_runtime.json  (csr_values + axi.expected_writes sidecar)
#
# Targets:
#   make            -- build everything
#   make <kernel>   -- build only that kernel  (e.g. make srad_prepare)
#   make clean      -- rm -rf build/
#   make list       -- list discoverable kernels

PYTHON          ?= python3
DORA_REPO       ?= /data/amanoj3/dora
DORA_STATIC_DIR ?= $(DORA_REPO)/examples/devices/dice-isca/mini_dice/static-build
WORKSPACE_PKL   ?= $(DORA_STATIC_DIR)/workspace.pkl

BUILD_DIR     := build
SCRIPTS_DIR   := scripts
KERNELS_DIR   := kernels

FASM_TO_BIN     := $(PYTHON) $(SCRIPTS_DIR)/fasm_to_bin.py
BINS_TO_MEM     := $(PYTHON) $(SCRIPTS_DIR)/bins_to_bitstream_mem.py
JSON_TO_META    := $(PYTHON) $(SCRIPTS_DIR)/json_to_meta_mem.py
FILL_WRITES     := $(PYTHON) $(SCRIPTS_DIR)/fill_expected_writes.py

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

.PHONY: all clean list fill-writes $(KERNELS)
.DELETE_ON_ERROR:

all: $(ALL_OUTPUTS)

list:
	@echo "Discovered kernels:"
	@for k in $(KERNELS); do echo "  - $$k"; done

# Re-simulate every kernel and overwrite axi.expected_writes in each
# kernels/<k>/*_test_vector.json. Re-run this whenever the CSR layout, p-graph
# semantics, or grid size change so the TB has an accurate golden reference.
fill-writes:
	$(FILL_WRITES) --all

clean:
	rm -rf $(BUILD_DIR)

# ---------------------------------------------------------------------------
# Per-kernel rule expansion
# ---------------------------------------------------------------------------
# For each kernel `k`:
#   FASMS_$k    = list of kernels/$k/<stage>.fasm
#   BINS_$k     = list of build/$k/<stage>.bin
#   JSON_$k     = the test_vector.json
#
# Recipes (pattern):
#   build/$k/<stage>.bin           : kernels/$k/<stage>.fasm  -> fasm_to_bin
#   build/$k/$k_bitstream.mem      : all bins + JSON          -> bins_to_bitstream_mem
#   build/$k/$k_{meta,cta_desc,runtime}: JSON                 -> json_to_meta_mem
#
# We use $(eval $(call ...)) to instantiate these per kernel without losing
# Make's automatic dependency tracking.

define KERNEL_template

$(1): $(call KERNEL_OUTPUTS,$(1))

FASMS_$(1) := $$(wildcard $(KERNELS_DIR)/$(1)/*.fasm)
BINS_$(1)  := $$(patsubst $(KERNELS_DIR)/$(1)/%.fasm,$(BUILD_DIR)/$(1)/%.bin,$$(FASMS_$(1)))
JSON_$(1)  := $$(firstword $$(wildcard $(KERNELS_DIR)/$(1)/*_test_vector.json))

# dora-run cd's into dora.py/ before exec, so all script paths and --out
# targets must be absolute. $(CURDIR) anchors them to the testkernels root.
$(BUILD_DIR)/$(1)/%.bin: $(KERNELS_DIR)/$(1)/%.fasm $(WORKSPACE_PKL)
	@mkdir -p $$(@D)
	$(FASM_TO_BIN) --fasm $$(CURDIR)/$$< --workspace $(WORKSPACE_PKL) --out $$(CURDIR)/$$@

$(BUILD_DIR)/$(1)/$(1)_bitstream.mem: $$(BINS_$(1)) $$(JSON_$(1))
	@mkdir -p $$(@D)
	$(BINS_TO_MEM) --json $$(CURDIR)/$$(JSON_$(1)) --bins-dir $$(CURDIR)/$(BUILD_DIR)/$(1) --out $$(CURDIR)/$$@

$(BUILD_DIR)/$(1)/$(1)_meta.mem $(BUILD_DIR)/$(1)/$(1)_cta_desc.mem $(BUILD_DIR)/$(1)/$(1)_runtime.json: $$(JSON_$(1))
	@mkdir -p $(BUILD_DIR)/$(1)
	$(JSON_TO_META) --json $$(CURDIR)/$$(JSON_$(1)) --out-dir $$(CURDIR)/$(BUILD_DIR)/$(1) --stem $(1)

endef

$(foreach k,$(KERNELS),$(eval $(call KERNEL_template,$(k))))
