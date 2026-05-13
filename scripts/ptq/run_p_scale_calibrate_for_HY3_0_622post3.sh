export VLLM_ALLOW_INSECURE_SERIALIZATION=1
# Force Ray to reload code (disable code caching)
export RAY_DEDUP_LOGS=0
# Force Python to not use bytecode cache
export PYTHONDONTWRITEBYTECODE=1

export VLLM_ENABLE_CHUNKED_PREFILL=1
export VLLM_DISTRIBUTED_EXECUTOR_BACKEND=mp
export MOE_MODE=fused
export VLLM_ATTENTION_BACKEND=FLASHINFER
export ASYNC_SCHEDULING=1
export VLLM_ENABLE_PREFIX_CACHING=1
export PRECISIONMODE=HF

# Ray Compiled DAG per-step timeout (seconds).
# Default is 300s, which is too short because each prefill step runs a heavy
# hook: recompute P = softmax(Q @ K^T * scale) for every layer, evaluate 6
# candidate FP8 scales, and accumulate MSE. Bump to 2 hours to be safe.
export RAY_CGRAPH_get_timeout=7200

run_name=hy3.0_vllm_p_scale_search-codeagent-32k-e92-s60-160-512
model_path=/dockerdata/e92-s60-bf16
ptq_data_path=/apdcephfs_sh9/share_303664697/krizaltang/datasets/512applied_codeagent_1k.json
output_dir=/root/e92-s60-bf16-codeagent-p-scale-search-per-head-32k-160-512

tp_size=16
batch_size=4
num_samples=32
max_length=32768

# Distributed backend: use "ray" for multi-node TP (e.g. TP=16 across 2 nodes),
# or "mp" for single-node multi-GPU.
distributed_executor_backend="ray"

# Boolean flags (non-empty to enable, empty to disable)
skip_weight_loading=""  # set to "--skip-weight-loading" to enable debug mode

# P-matrix scale search settings
# Comma-separated list of candidate FP8 scales to evaluate.
# Values should cover the expected range of softmax probabilities (0~1).
scale_list="0.00625,0.005681818181818182,0.005208333333333333,0.004807692307692308,0.004464285714285714,0.004166666666666667,0.00390625,0.003472222222222222,0.003125,0.002840909090909091,0.0026041666666666665,0.002403846153846154,0.002232142857142857,0.001953125"

# Number of Q tokens processed per block during P re-computation.
# Larger values are faster but use more GPU memory.
# H20 has ~50G free per GPU at calibration time, so 2048 is safe and gives
# ~8x fewer Python/kernel-launch overheads vs 256.
q_block_size=8192

# Set to "--no-collect-p-dist" to skip per-head P-value distribution
# statistics (faster, no p_matrix_distribution_*.json outputs).
# Leave empty to keep the distribution collection enabled.
no_collect_p_dist=""

# Number of leading sample-local key columns to *exclude* from the FP8
# NMSE objective.  Set to 2 (default) to ignore the first two attention-
# sink columns identified by the per-column analysis.  Set to 0 to
# disable masking entirely (objective falls back to standard NMSE over
# all columns).
mse_skip_first_cols=2

mkdir -p logs

python3 tools/run_p_scale_calibrate.py \
    --model-path $model_path \
    --ptq-data-path $ptq_data_path \
    --output-dir $output_dir \
    --tp-size $tp_size \
    --batch-size $batch_size \
    --num-samples $num_samples \
    --max-length $max_length \
    --distributed-executor-backend $distributed_executor_backend \
    --scale-list "$scale_list" \
    --q-block-size $q_block_size \
    --mse-skip-first-cols $mse_skip_first_cols \
    $no_collect_p_dist \
    $skip_weight_loading \
    2>&1 | tee logs/${run_name}.log
