# export PYTHONPATH=/apdcephfs_gy7/share_303171455/linchuanxie/new_hy3_vllm/AngelSlim:$PYTHONPATH
# mirrors.tencent.com/hunyuan_infer/text2text_infer:H-6.2.2.post3
# Allow function serialization for apply_model in vLLM v1 engine
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
# Force Ray to reload code (disable code caching)
export RAY_DEDUP_LOGS=0
# Force Python to not use bytecode cache
export PYTHONDONTWRITEBYTECODE=1
# NOTE: VLLM_MOE_COLLECT_STATS is intentionally NOT set here –
#       this script only calibrates kv-cache, not weight/activation/MoE.

# export MAX_NUM_BATCHED_TOKENS=32768
export VLLM_ENABLE_CHUNKED_PREFILL=1
export VLLM_DISTRIBUTED_EXECUTOR_BACKEND=mp
export MOE_MODE=fused
export VLLM_ATTENTION_BACKEND=FLASHINFER
export ASYNC_SCHEDULING=1
export VLLM_ENABLE_PREFIX_CACHING=1
export PRECISIONMODE=HF

run_name=hy3.0_vllm_kvcache_only-datav3-32k-e92-s60
model_path=/dockerdata/e92-s60-bf16
# /apdcephfs_gy7/share_303171455/linchuanxie/HY3.0-A20B-TOC-RL0331
ptq_data_path=/apdcephfs_sh9/share_303664697/krizaltang/datasets/512applied_codeagent_1k.json
#/cfs_cloud/woodchenwu/HY3.0-A20B-TOC-RL0331/wood_sample.jsonl
output_dir=/root/e92-s60-bf16-datav3-kvcache-search-per-head-32k
# /apdcephfs_gy7/share_303171455/linchuanxie/HY3.0-A20B-TOC-RL0331_search_tensor
tp_size=16
batch_size=4
num_samples=512
max_length=32768

# Boolean flags (non-empty to enable, empty to disable)
skip_weight_loading=""  # set to "--skip-weight-loading" to enable debug mode

# Granularity: set to "--per-head" for per-KV-head scales, "" for per-layer (per-tensor)
per_head="--per-head"             # set to "--per-head" to enable per-head mode

# KV cache scale search settings
search_kv_scale="--search-kv-scale"   # set to "" to skip scale search
search_kv_num_samples=32               # number of samples used for the search
search_kv_min_multiplier=0.4           # lower bound of multiplier search range
search_kv_max_multiplier=8.0          # upper bound of multiplier search range
search_kv_num_steps=50                 # number of log-uniform grid points

mkdir -p logs

python3 tools/run_kvcache_calibrate.py \
    --model-path $model_path \
    --ptq-data-path $ptq_data_path \
    --output-dir $output_dir \
    --tp-size $tp_size \
    --batch-size $batch_size \
    --num-samples $num_samples \
    --max-length $max_length \
    $per_head \
    $search_kv_scale \
    --search-kv-num-samples $search_kv_num_samples \
    --search-kv-min-multiplier $search_kv_min_multiplier \
    --search-kv-max-multiplier $search_kv_max_multiplier \
    --search-kv-num-steps $search_kv_num_steps \
    2>&1 | tee logs/${run_name}.log
