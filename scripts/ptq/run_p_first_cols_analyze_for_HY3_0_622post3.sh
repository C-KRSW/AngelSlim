export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export RAY_DEDUP_LOGS=0
export PYTHONDONTWRITEBYTECODE=1

export VLLM_ENABLE_CHUNKED_PREFILL=1
export VLLM_DISTRIBUTED_EXECUTOR_BACKEND=mp
export MOE_MODE=fused
export VLLM_ATTENTION_BACKEND=FLASHINFER
export ASYNC_SCHEDULING=1
export VLLM_ENABLE_PREFIX_CACHING=1
export PRECISIONMODE=HF

# This analysis is much cheaper than the full scale search
# (no MSE loop, no histogram), but we still bump the timeout to be safe.
export RAY_CGRAPH_get_timeout=3600

run_name=hy3.0_vllm_p_analysis_codeagent-32k-e92-s60-all
model_path=/dockerdata/e92-s60-bf16
ptq_data_path=/apdcephfs_gy8/share_300532381/krizaltang/datasets/512applied_codeagent_1k.json
output_dir=/apdcephfs_gy8/share_300532381/krizaltang/e92-s60-bf16-codeagent-p-analysis-per-head-32k

tp_size=16
batch_size=4
# Only a handful of prompts is needed - per-column means stabilise fast.
num_samples=4
max_length=32768

distributed_executor_backend="ray"

skip_weight_loading=""  # set to "--skip-weight-loading" for fast debug

# Number of leading P columns to analyse.
num_cols=64

# Q-token block size for memory-bounded P computation.
q_block_size=2048

# JSON size knob: set to "--no-per-head-detail" to skip the per-Q-head
# per-column tensors in the output (file shrinks from O(H_q*N*L) to O(N*L)).
# 80 layers x 64 heads x 64 cols x 3 fields ~ 20MB; layer-only summary ~250KB.
no_per_head_detail=""

mkdir -p logs

python3 tools/run_p_first_cols_analyze.py \
    --model-path $model_path \
    --ptq-data-path $ptq_data_path \
    --output-dir $output_dir \
    --tp-size $tp_size \
    --batch-size $batch_size \
    --num-samples $num_samples \
    --max-length $max_length \
    --distributed-executor-backend $distributed_executor_backend \
    --num-cols $num_cols \
    --q-block-size $q_block_size \
    $no_per_head_detail \
    $skip_weight_loading \
    2>&1 | tee logs/${run_name}.log
