# Status

## 已完成什么

- 已建立 Python 项目骨架、CLI、YAML 模型配置、OpenAI 兼容客户端、安全工作区、工具调用解析器、运行器、评分和报告模块。
- 当前保留 SWE-Review-Bench 和 Martian Code Review Bench；review-hour 选集为 SWE-Review 500 项、Martian 50 个 PR。
- 已安装项目依赖，Qwen2.5 文本工具调用 fallback、native tool call、路径逃逸保护和 mock agent 链路均已验证。
- Multi-SWE-bench Mini、SWE-bench Verified、SWE-bench Multilingual 和 CodeReviewQA 的本地数据、选集已清理；所有 `mswebench/*` Docker 镜像也已删除。当前仅保留 review 所需数据。
- 下载器已支持通过 `HF_ENDPOINT` 使用 Hugging Face 兼容国内镜像。

## 下次从哪里继续

- 设置三个模型的 `*_BASE_URL`/`*_API_KEY`，运行 `probe`，再使用 `review-hour` profile 执行 review 测试。
- 如需恢复 Agentic 或其他已清理数据，需要重新下载数据集并重新拉取对应 Docker 镜像。

## 哪些文件不能动

- `data/raw/`、`data/cache/` 和 `outputs/` 是运行时产物，不应提交或手工编辑。
- 模型服务地址和密钥只通过环境变量提供，不写入仓库。

## 验证结果

- `python -m pytest -q`：5 passed。
- `python -m compileall -q src`：通过。
- mock OpenAI 服务：客户端、native tool call、SafeWorkspace 和 agent loop：通过。
- 已清理本地 Agentic/code-fixing 数据及其 Docker 镜像；当前 Docker 中保留的镜像仅为模型服务或其他非 Multi-SWE 基础镜像。
