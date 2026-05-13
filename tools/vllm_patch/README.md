# vLLM Patch for AngelSlim Calibration

This directory contains patch files that need to be applied to an installed
vLLM package to enable AngelSlim's PTQ calibration features (especially MoE
expert statistics collection on `FusedMoE` layers).

## What's in this directory

| File                | Purpose                                                                                                |
| ------------------- | ------------------------------------------------------------------------------------------------------ |
| `fused_moe.py`      | Patched version of `vllm/model_executor/layers/fused_moe/fused_moe.py` with AngelSlim hooks injected.  |
| `envs.py`           | Patched version of `vllm/envs.py` that adds `VLLM_MOE_COLLECT_STATS*` environment variables.           |
| `README.md`         | This file.                                                                                             |

These patches are aligned with the **current** vLLM version installed in the
calibration environment. If your vLLM version differs, the patch files may
need to be regenerated against your specific vLLM source.

## Required companion file: `vllm_calibrate_utils.py`

`fused_moe.py` imports `collect_fused_moe_internal_stats` from a module named
`vllm_calibrate_utils`. The lookup logic walks up from the patched
`fused_moe.py` location and appends `vllm/tools/`, so the calibration utils
file **must** be placed inside the installed vLLM package as:

```
<vllm_install_dir>/tools/vllm_calibrate_utils.py
```

The single source of truth for this file lives at:

```
angelslim/compressor/quant/core/vllm_calibrate_utils.py
```

## Deployment

Assuming `VLLM_DIR` points to your installed vLLM package directory (e.g.
`/usr/local/lib/python3.12/dist-packages/vllm` or your editable-install
checkout), run:

```bash
# 1. Locate the vLLM package directory
VLLM_DIR=$(python3 -c "import vllm, os; print(os.path.dirname(vllm.__file__))")
echo "vLLM is installed at: $VLLM_DIR"

# 2. (Optional) Back up the original files
cp "$VLLM_DIR/envs.py" "$VLLM_DIR/envs.py.bak"
cp "$VLLM_DIR/model_executor/layers/fused_moe/fused_moe.py" \
   "$VLLM_DIR/model_executor/layers/fused_moe/fused_moe.py.bak"

# 3. Apply the patches
ANGELSLIM_ROOT=$(git rev-parse --show-toplevel)   # or set manually
cp "$ANGELSLIM_ROOT/tools/vllm_patch/envs.py"      "$VLLM_DIR/envs.py"
cp "$ANGELSLIM_ROOT/tools/vllm_patch/fused_moe.py" \
   "$VLLM_DIR/model_executor/layers/fused_moe/fused_moe.py"

# 4. Place the calibration utils inside the vLLM package
mkdir -p "$VLLM_DIR/tools"
cp "$ANGELSLIM_ROOT/angelslim/compressor/quant/core/vllm_calibrate_utils.py" \
   "$VLLM_DIR/tools/vllm_calibrate_utils.py"
```

After that, set the environment variables before launching calibration:

```bash
export VLLM_MOE_COLLECT_STATS=1
export VLLM_MOE_COLLECT_STATS_VERBOSE=0
export VLLM_MOE_COLLECT_PER_EXPERT_STATS=1
```

Then run the calibration scripts under `scripts/ptq/`, e.g.
`run_vllm_calibrate_for_HY3_0_622post4.sh`.

## Reverting

To restore the original vLLM files:

```bash
mv "$VLLM_DIR/envs.py.bak" "$VLLM_DIR/envs.py"
mv "$VLLM_DIR/model_executor/layers/fused_moe/fused_moe.py.bak" \
   "$VLLM_DIR/model_executor/layers/fused_moe/fused_moe.py"
rm  "$VLLM_DIR/tools/vllm_calibrate_utils.py"
```
