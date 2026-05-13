# PTQ 校准 / 量化脚本说明

本目录包含基于 vLLM 的 PTQ（Post-Training Quantization）校准和量化脚本。
所有脚本必须从 AngelSlim 仓库根目录执行（脚本内部使用 `tools/...` 形式的相对路径）。

================================================================================
一、HY3.0 系列脚本（Hunyuan-A20B 等 HY3 模型）
================================================================================

下面 3 个脚本共享同一套 vLLM 运行时环境（chunked prefill / FlashInfer attention /
mp distributed executor / fused MoE 等），区别在于产出物不同。

--------------------------------------------------------------------------------
1. run_vllm_quant_for_HY3.sh   ★推荐的"一键流水线"
--------------------------------------------------------------------------------
功能：bf16 模型  →  vLLM 激活校准  →  FP8 HF safetensors  全流程一次完成。

阶段 1：调用 tools/run_vllm_calibrate.py
   * 用 vLLM 加载 bf16 模型，在 PTQ 数据集上跑前向，注册 weight/activation/MoE/
     KV-cache 钩子。
   * 输出到 ${act_dir}：
       - activation_stats.json       （per-tensor min/max；含合并后的 per-head 项）
       - moe_expert_stats.json       （每个 MoE expert 的输入激活统计）
       - kv_scale_multipliers*.json  （若开启 --search-kv-scale）
       - kv_cache_tuned_scales*.json （搜索后的最终 KV scale）

阶段 2：调用 tools/fp8_quant_with_vllm_activation_622p3.py
   * 读取 ${act_dir} 下的 activation_stats.json / moe_expert_stats.json，
     结合原 bf16 权重，做 per-tensor FP8 量化（含 weight + input scale），
     写出到 ${fp8_path}。
   * 当存在 per-head KV 统计时，会同时输出 kv_cache_scales.safetensors。

CLI 开关：
   bash run_vllm_quant_for_HY3.sh                    # 两阶段都跑
   bash run_vllm_quant_for_HY3.sh --skip-calibrate   # 仅量化（复用已有 act_dir）
   bash run_vllm_quant_for_HY3.sh --skip-quantize    # 仅校准
   bash run_vllm_quant_for_HY3.sh --help             # 打印用法

关键关闭项：set -euo pipefail，任一阶段失败立即中断。

--------------------------------------------------------------------------------
2. run_vllm_calibrate_for_HY3.sh   ＝ 一键脚本里的"阶段 1"独立版
--------------------------------------------------------------------------------
功能：只跑 W8A8C8 联合校准，不做量化。
入口：tools/run_vllm_calibrate.py
开启的环境变量：
   VLLM_MOE_COLLECT_STATS=1
   VLLM_MOE_COLLECT_PER_EXPERT_STATS=1
   VLLM_MOE_COLLECT_STATS_VERBOSE=0
默认采用 --kv-granularity per-head 并开启 --search-kv-scale。
产物：activation_stats.json / moe_expert_stats.json / kv_scale_multipliers.json /
      kv_cache_tuned_scales*.json，写入 ${output_dir}。

适用场景：
   * 想自己接后续量化工具，不走 fp8_quant_with_vllm_activation_622p3.py。
   * 想单独调校 PTQ 数据集 / num_samples / max_length，再用 run_vllm_quant_for_HY3.sh
     的 --skip-calibrate 复用结果。
   * Debug 用 --skip-weight-loading 跑 dummy 权重，快速验证 hook 注册流程。

--------------------------------------------------------------------------------
3. run_kvcache_calibrate_for_HY3.sh   ＝ 仅校准 KV-cache（轻量）
--------------------------------------------------------------------------------
功能：只校准 KV-cache（K/V min/max），不做 weight / activation / MoE 统计。
入口：tools/run_kvcache_calibrate.py
关键差异（与 run_vllm_calibrate_for_HY3.sh 对比）：
   * 故意 NOT 设置 VLLM_MOE_COLLECT_STATS — 完全跳过 MoE/Linear 钩子，
     启动更快、CPU 内存占用更低。
   * 默认 --per-head + --search-kv-scale，搜索范围 [0.4, 8.0]、num_steps=50，
     比 run_vllm_calibrate_for_HY3.sh 默认范围 [0.8, 16.0] 更窄、更聚焦。
   * 产物文件名不同：单独写 activation_stats_per_head.json（不再合并到
     activation_stats.json），便于做 KV-only 实验对比。

适用场景：
   * 已有 W8A8 量化模型，想单独研究/调优 KV scale。
   * 多组 KV 搜索参数对比实验，节省"无关"前向计算。

================================================================================
二、其他模型脚本
================================================================================

--------------------------------------------------------------------------------
4. run_vllm_quant_for_deepseek_v3.sh
--------------------------------------------------------------------------------
功能：DeepSeek-R1 / V3 系列的 W4A8-FP8 校准（基于配置文件驱动）。
入口：tools/run.py，使用配置文件
      configs/deepseek_r1/w4a8_fp8/deepseek_r1_w4a8_fp8_vllm_calibrate.yaml
与 HY3 系列脚本的区别：走 angelslim 标准 Engine + yaml-config 路径，
而非自定义的 run_vllm_calibrate.py。

================================================================================
三、其他脚本（暂不在本 README 描述）
================================================================================
   - run_p_first_cols_analyze_for_HY3_0_622post3.sh
   - run_p_scale_calibrate_for_HY3_0_622post3.sh
（这两个是 P-matrix / Attention-sink 列分析与 P-scale 搜索的实验性脚本，
 详细用途请参见对应 tools/run_p_first_cols_analyze.py 与
 tools/run_p_scale_calibrate.py 的脚本头注释。）

================================================================================
四、KV scale 搜索的后处理（重要）
================================================================================
对 1 / 2 / 3 三个脚本：如果开启了 --search-kv-scale，校准产出的搜索结果
（kv_cache_tuned_scales*.json）默认 *不会* 自动写回到最终量化模型中。

正常产出量化模型后，需要手工运行下面其中一个脚本，把搜索得到的 KV scale
替换进去，否则模型默认仍然使用"不搜索"时的 scale：

   * tools/replace_kv_scales.py             （per-tensor 模式）
   * tools/replace_kv_scales_perhead.py     （per-head 模式）

run_vllm_quant_for_HY3.sh 的阶段 2 (fp8_quant_with_vllm_activation_622p3.py)
当 per-head 统计存在时会自动写出 kv_cache_scales.safetensors（base scale，
*未* 应用 search multiplier），所以同样需要 replace_kv_scales_perhead.py
做最终替换。
