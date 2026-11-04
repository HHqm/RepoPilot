# RepoPilot

[![CI](https://github.com/HHqm/RepoPilot/actions/workflows/ci.yml/badge.svg)](https://github.com/HHqm/RepoPilot/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)
![LangGraph](https://img.shields.io/badge/orchestration-LangGraph-6f42c1)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

RepoPilot 是一个客户协作式多智能体软件公司：

```text
客户与产品经理多轮聊天 / 上传文件
→ 项目级 RAG
→ 交付总监需求避坑会
→ 客户确认 PRD
→ 项目规划与架构
→ 多语言开发
→ QA 测试与返工
→ 客户最终验收
→ ZIP 源码与详细指南交付
```

| 文档 | 内容 |
| --- | --- |
| [docs/AGENT_DESIGN.md](docs/AGENT_DESIGN.md) | 两条工作流的状态图、有界自我修正回路、检索规划、人工决策契约、失败边界 |
| [docs/RAG.md](docs/RAG.md) | 增量索引、向量缓存、混合检索、代码感知稀疏向量、评测方法与消融结果 |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | 架构总览 |
| [docs/DESIGN.md](docs/DESIGN.md) | 产品与系统完整设计 |
| [docs/ENGINEERING_AUDIT.md](docs/ENGINEERING_AUDIT.md) | 工程审计与整改记录 |
| [docs/E2E_REPORT_DEEPSEEK.md](docs/E2E_REPORT_DEEPSEEK.md) | 真实 DeepSeek 全链路验收 |

## 界面预览

工作台以“软件公司”方式呈现客户、产品经理、总监和研发团队的协作过程。左侧是客户与产品经理的真实对话，中央是阶段进度和事件流，右侧可以查看需求、RAG 证据、人工审批和交付物。

![RepoPilot 客户协作工作台](screenshots/home.png)

> 截图来自本地 Mock 模式，仅展示界面，不包含任何 API Key。模型中心、上传文件、需求评审和发布审批的操作说明见 [快速开始](docs/QUICKSTART.md)。

## 功能

- 聊天优先首页：无需先填表，首次发送消息或上传文件时自动创建项目会话；
- 客户—产品经理持续项目会话；
- PDF、Word、Excel、PPT、文本和代码上传；
- Docling/pypdf 解析、秘密脱敏和项目隔离 RAG；stat + SHA-256 两级增量索引、内容寻址向量缓存、
  索引与向量库不一致时自动重建；
- 需求澄清会议、风险规避和任务拆解；
- Python、Java、Go、C++、TypeScript/JavaScript；
- LangGraph 双 Human-in-the-loop，审批绑定产物 SHA-256 摘要，拒绝过期审批；
- 客户可以“确认 / 退回修改 / 终止项目”；QA 缺陷、发布审查意见和客户修改要求共享两轮开发返工预算；
- 仓库维护 Agent 带有界自我修正：审查拒绝后带着 findings 修订，测试失败后回滚、复盘，
  再用失败信息重新检索和规划，每个新补丁都要重新人工审批；
- 多查询检索规划（Issue / 代码符号 / 失败 traceback）+ RRF 融合；
- 每次运行汇总 LLM 调用次数、token 用量与各 Agent 耗时；
- SQLite 持久化 SSE 事件与跨重启原序号回放；
- 结构化输出校验和一次有界修复；
- 项目内部文档回写后供后续 Agent 再检索；
- 超详细环境、构建、测试、部署、使用、排障和回滚指南；
- 源码、PRD、会议、计划、架构、测试报告和清单 ZIP；
- DeepSeek‑V4‑pro、确定性 Mock 与 Agent 级多供应商模型路由；
- 前端 BYOK 模型中心、动态 `/models` 发现和自定义 OpenAI-compatible Base URL；
- DeepSeek、OpenAI、通义千问、豆包、智谱、Kimi、Gemini、ZZSHU、Ollama 和自定义兼容端点预设。

## 工程质量

| 维度 | 做法 |
| --- | --- |
| 持续集成 | GitHub Actions：Ruff 静态检查、Python 3.11/3.12 矩阵测试、Mock 端到端工作流评测、34 条检索基准质量门禁（Recall@5 ≥ 0.65、MRR ≥ 0.6，离线模型）、Docker 镜像构建 |
| 测试 | 97 项 `unittest`，全部离线运行（Mock LLM + 确定性向量），不需要 API Key |
| 可观测性 | 每个 Agent 产出结构化 trace（含检索各阶段耗时）；运行报告汇总成本与耗时；SSE 事件写入 SQLite，支持 `Last-Event-ID` 断线续传 |
| 评测 | 34 条带标注检索基准、6 种检索配置消融（`scripts/evaluate_retrieval.py --ablation`）：真实模型下生产配置 Recall@5 0.897、MRR 0.882，重排在混合召回基础上 MRR +0.078（[报告](docs/RAG_EVALUATION.md)）；Mock 端到端工作流评测 |
| 安全 | Bearer Token / 回环限制、SSRF 防护、BYOK 密钥只驻留内存、宿主机测试默认关闭、上传与索引脱敏 |
| 部署 | 非 root 容器、`HEALTHCHECK`、依赖层与源码层分离缓存、`.dockerignore` 排除密钥与运行数据 |

### 性能基准

运行时热点经过逐项剖析和优化。下表由 `python scripts/benchmark_runtime.py` 在同一台
Windows 11 开发机（Python 3.11）上分别对 v0.6.0 初版和当前代码测得：

| 热点路径 | 优化前 | 优化后 | 提升 | 手段 |
| --- | ---: | ---: | ---: | --- |
| 冷启动查询运行状态（`GET /runs/{id}`） | 17.4 ms | 0.73 ms | ~24× | LangGraph 图只编译一次，按会话 `copy()` 绑定 checkpointer |
| 发布一条 SSE 事件 | 12.3 ms | 0.11 ms | ~110× | 事件库复用长连接 + WAL `synchronous=NORMAL`，序号 O(1) 递增 |
| 扫描含 3000 个依赖文件的仓库 | 800 ms | 7.8 ms | ~100× | `os.walk` 遍历时剪枝 `node_modules/.git/.venv`，不跟随符号链接 |
| 切分 2 万行源码 | 155 ms | 99 ms | ~1.6× | 每行只分词一次，窗口与重叠复用 token 计数 |

基准完全离线，在临时目录中运行，不会连接正在运行的服务或模型 API。

## 环境

需要 Python 3.11 或 3.12：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1        # Linux/macOS: source .venv/bin/activate
pip install -e ".[dev]"
```

如需复杂 PDF、Office、扫描件和表格解析：

```powershell
pip install -e ".[documents]"
```

也可以用 `environment.yml` 创建 Conda 环境。

## 配置

`.env` 示例：

```dotenv
REPOPILOT_LLM_MODE=deepseek
DEEPSEEK_API_KEY=你的密钥
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-v4-pro

REPOPILOT_RAG_EMBEDDING_PROVIDER=bge-m3
REPOPILOT_RAG_EMBEDDING_MODEL=BAAI/bge-m3
REPOPILOT_RAG_RERANKER_PROVIDER=bge
REPOPILOT_RAG_RERANKER_MODEL=BAAI/bge-reranker-v2-m3

# 限制维护工作流只能读取这些工作区；Windows 使用分号分隔，留空为当前目录。
REPOPILOT_WORKSPACE_ROOTS=D:\workspace
# 远程访问必须配置 Bearer Token；仅 127.0.0.1 使用时可留空。
REPOPILOT_API_TOKEN=
# 默认禁止在宿主机执行仓库测试。
REPOPILOT_ALLOW_HOST_TEST_EXECUTION=0
# 默认阻止模型供应商 Base URL 访问本机或私网，避免 SSRF。
REPOPILOT_ALLOW_PRIVATE_MODEL_ENDPOINTS=0
```

不要把 `.env`、私钥或生产凭据上传到代码仓库。

### 前端 BYOK 与第三方模型

打开工作台右上角“模型中心”：

1. 选择供应商预设，或选择“自定义 OpenAI-compatible”；
2. 填写 Base URL 和 API Key；
3. 点击“测试连接并保存”，后端会优先调用 `/models`；
4. 如果供应商没有模型列表接口，填写模型 ID，系统会执行一次极小模型调用验证；
5. 选择单一、均衡、交叉审查或高级自定义路由；
6. 为产品、总监、项目、架构、开发、DevOps、QA 和发布审查 Agent 分配模型。

用户 API Key 仅保存在当前后端服务进程的内存中。浏览器不使用
`localStorage/sessionStorage` 保存密钥，项目会话、LangGraph Checkpoint、RAG、
Trace 和 ZIP 只保存 `credential_id`、供应商和模型。服务重启后需要重新连接
用户凭证；服务端环境变量托管的 DeepSeek 不受此限制。

Ollama、vLLM 和企业内网模型需要管理员显式设置
`REPOPILOT_ALLOW_PRIVATE_MODEL_ENDPOINTS=1`。公网端点必须使用 HTTPS。

## 启动

```powershell
python -m uvicorn repopilot.api:app --host 127.0.0.1 --port 8000
```

或使用 Docker Compose（同时启动 Qdrant Server；需先在 `.env` 中设置
`REPOPILOT_API_TOKEN`，因为宿主机请求经 Docker 网桥到达，不属于回环地址）：

```powershell
docker compose up --build
```

打开：

- 工作台：<http://127.0.0.1:8000/>
- API：<http://127.0.0.1:8000/docs>

## 项目会话 API

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/project-sessions` | 创建客户项目会话 |
| GET | `/project-sessions/{id}` | 读取项目会话 |
| POST | `/project-sessions/{id}/messages` | 与产品经理聊天 |
| POST | `/project-sessions/{id}/files` | 上传资料并进入 RAG |
| POST | `/project-sessions/{id}/launch` | 召开需求评审会 |
| GET | `/runs/{id}/events` | SSE 实时事件 |
| POST | `/runs/{id}/decision` | 客户确认 / 退回修改 / 终止（`approved`、`comment`、`terminate`、`artifact_digest`） |
| GET | `/runs/{id}/download` | 下载最终 ZIP |
| GET | `/model-providers/catalog` | 读取供应商预设 |
| GET/POST | `/model-providers/credentials` | 查询或连接本次进程的 BYOK 凭证 |
| POST | `/model-providers/credentials/{id}/refresh` | 重新发现模型 |
| DELETE | `/model-providers/credentials/{id}` | 从服务进程清除用户凭证 |
| POST | `/model-routing/validate` | 验证 Agent 级路由快照 |

## 测试

```powershell
ruff check src tests scripts
python -m unittest discover -s tests -v
python -m pip check
```

测试和检索评测使用独立临时运行目录，不会连接正在运行的服务所占用的
Qdrant Local，也不会写入生产 Qdrant 集合。因此可以在本地服务保持启动时执行：

```powershell
python scripts\evaluate_retrieval.py --min-recall 0.65 --min-mrr 0.6
python scripts\evaluate.py
python scripts\benchmark_runtime.py
```

真实 DeepSeek 极小端到端验收（会产生 API 费用，并自动批准本地测试项目的两个人工门禁）：

```powershell
python scripts\smoke_deepseek_tiny_e2e.py
```

该脚本会上传一份极小规格，完整经过产品对话、项目 RAG、需求会议、双审批、代码生成、
QA、发布审查和 ZIP 交付；随后只对零第三方依赖且通过 AST 白名单的 Python 项目在临时
目录运行 `unittest`。它是受限冒烟验证，不等同于 Docker/gVisor/Firecracker 沙箱。

Qdrant Local 适合单进程本地演示，不支持多个进程或多个 Uvicorn Worker 同时打开
同一个存储目录。多进程、容器或团队环境请启动 `docker-compose.yml` 中的 Qdrant
Server，并设置 `REPOPILOT_QDRANT_URL`；不要通过复制本地存储目录绕过文件锁。

测试覆盖：

- 聊天与产品理解；
- 文件解析、项目隔离和秘密脱敏；
- 需求会议风险与拆解；
- 双人工审批；
- 多语言生成；
- QA 报告；
- 退回不写入；
- 原子交付与回滚；
- ZIP 内容和下载契约；
- 实时 SSE；
- SSE 事件跨服务重启回放；
- 过期审批摘要拒绝；
- 公开 API 不能注入内部知识目录；
- Studio Reviewer 不会截断源码后继续盲审；
- 宿主机测试默认禁用和危险 pytest 参数拒绝；
- RAG 增量索引和跨项目隔离。
- ZZSHU/自定义兼容地址、私网 SSRF 阻断和 BYOK 密钥脱离工作流状态；
- Agent 级模型解析以及开发—审查模型独立路由。

## 安全边界

RepoPilot 不会在未取得客户环境和授权时声称完成生产上线。生成的源代码默认不直接
在 RepoPilot 宿主机执行；真实构建、容器、数据库、网络和第三方集成测试应在隔离
CI 或客户目标环境执行，具体命令和预期结果写入 `DEPLOYMENT.md`。

维护仓库模式中的 pytest/unittest 同样默认禁止宿主机执行。只有设置
`REPOPILOT_ALLOW_HOST_TEST_EXECUTION=1` 才会对受信任本地仓库启用；此时仍会
清除密钥环境变量、隔离临时 HOME、禁用 pytest 自动插件并拒绝高风险参数，但这
不等同容器沙箱。
