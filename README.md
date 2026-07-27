# Coder Review Benchmark

用于比较 Qwen2.5-Coder-7B、Qwen3-Coder-30B 与 Qwen3-Coder-Next-80B 的 agentic coding 和 code review 能力。

## Model-only Code Review Benchmark v2（推荐）

```bash
conda activate coder-bench
python -m coder_review_benchmark prepare-review-v2
python -m coder_review_benchmark validate-selection --profile swe-review-balanced-500-v1
python -m coder_review_benchmark validate-selection --profile swe-review-official-1384-v1
python -m coder_review_benchmark doctor --profile swe-review-balanced-500-v1
python -m coder_review_benchmark probe --model qwen3-coder-30b

# 500 个去重、分层的补丁接受/拒绝判断
python -m coder_review_benchmark run \
  --suite swe_review --model qwen3-coder-30b \
  --profile swe-review-balanced-500-v1 --concurrency 1 \
  --context-policy common-32k

# 50 个真实 PR、136 条人工金标问题的缺陷定位
python -m coder_review_benchmark run \
  --suite martian --model qwen3-coder-30b \
  --profile review-hour --concurrency 1 \
  --context-policy common-32k
```

v2 的 SWE-Review 有两个互斥协议：`swe-review-balanced-500-v1` 是 500 个唯一 instance、250 resolved/250 unresolved 的主比较集；`swe-review-official-1384-v1` 保留三个生成器产生的全部非空候选 patch，允许重复 instance，是论文/完整协议。用 `validate-selection` 在运行前检查 JSONL 与 manifest 一致性。单模型服务固定使用 `--concurrency 1`；模型重试次数可用 `CBM_MODEL_MAX_RETRIES` 调整，GitHub PR diff 默认重试三次并缓存在 `data/cache/pr_diffs/`。

v2 是固定上下文、单轮、无工具的 model-only 评测。`common-32k` 使用确定性的字符预算并在结果中记录原始/最终字符数、截断原因和 prompt SHA；`native-context` 仅用于诊断，不应与主比较混用。Agentic 分支仍保持原有工具循环和 Docker 评测逻辑。

SWE-Review 只允许 `approve`、`request_changes`（兼容别名 `reject`）；未知决策、错误类型、无效 JSON 和 schema 错误均为 invalid，不会被计入混淆矩阵。报告同时给出 format/schema completion、全体/有效样本准确率、balanced accuracy、defect recall、误放行/误拒绝率和 MCC。Martian 使用独立 judge 将候选 finding 与人工金标做一对一语义匹配，报告 raw TP/FP/FN、micro 与 per-PR macro 指标、零 finding PR、judge 错误率及仓库拆分。

模型服务地址和密钥从 `.env` 或当前环境变量读取，例如 `QWEN3_CODER_30B_BASE_URL`、`QWEN3_CODER_30B_API_KEY`。模型名称可通过对应的 `*_MODEL_NAME` 环境变量覆盖，不依赖服务端临时实例别名。

运行结果写入 `outputs/<时间>_<模型>_<套件>/`，其中 `metrics.json` 是汇总指标，`results.jsonl` 保留逐题结果，`run_manifest.json` 记录实际模型名称、上下文策略、prompt/dataset manifest SHA、采样设置和可选模型制品元数据，不记录 API key。v2 运行标记为 `evaluation_version=model-review-v2`；汇总器发现 v2 运行时默认排除 Legacy 运行。

所有历史运行可自动汇总为 Excel、Markdown 和 CSV：

```bash
python -m coder_review_benchmark summarize --profile review-hour
```

汇总文件默认写入 `reports/`。主比较表只纳入指定 profile，防止旧 smoke 与正式评测混合；同一模型和套件存在失败重试和多次正式运行时，优先采用样本数最多的运行，再按完成率和运行时间选择。Excel 的“全部运行”工作表仍保留所有历史记录。

## 其他数据集

CodeReviewQA 当前在 Hugging Face 上要求访问授权；准备该数据集前设置 `HF_TOKEN`。本机内网代理若没有企业 CA，可临时使用 `CBM_ALLOW_INSECURE_DOWNLOAD=1`，正式环境应安装 CA 后取消该变量。

可通过 `HF_ENDPOINT` 指定 Hugging Face 兼容镜像，例如 `https://hf-mirror.com`。镜像需要支持数据集 API、文件列表和 resolve 下载接口。

完整规格见运行后生成的 `outputs/*/run_manifest.json`；数据源和抽样版本记录在 `data/manifest.json`。
