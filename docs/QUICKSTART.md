# RepoPilot 使用与部署教程

本文面向第一次运行 RepoPilot 的用户，覆盖本地 Mock 演示、DeepSeek 调用、文件上传、人工审批、RAG 和交付包下载。默认平台为 Windows 10/11，Linux/macOS 只需把路径分隔符换成 `/`。

## 1. 获取代码与准备环境

```powershell
git clone https://github.com/HHqm/RepoPilot.git
cd RepoPilot

# 推荐使用独立虚拟环境，避免污染系统 Python
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .

# 需要 PDF/Word/Excel/PPT/OCR 等解析能力时再安装可选依赖
python -m pip install -e ".[documents]"
```

如果还没有环境，可用仓库中的 `environment.yml` 创建基础环境；随后确认 `python --version` 与 `python -m pip --version` 指向同一环境。

## 2. 配置文件与密钥边界

```powershell
Copy-Item .env.example .env
notepad .env
```

首次建议保持 `REPOPILOT_LLM_MODE=mock`，不需要任何密钥即可跑通界面和自动化测试。需要调用 DeepSeek 时，只在本机 `.env` 中填写：

```dotenv
REPOPILOT_LLM_MODE=deepseek
DEEPSEEK_API_KEY=在本机填写，不要复制到 GitHub、截图或日志
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-v4-pro
```

`.env` 已被 `.gitignore` 忽略。不要把真实 key 写入 `.env.example`、README、前端代码、截图或 issue；如果 key 曾经出现在公开日志中，请立即在供应商控制台轮换。

## 3. 启动服务

```powershell
cd D:\gptbot\RepoPilot
python -m uvicorn repopilot.api:app --host 127.0.0.1 --port 8000
```

打开 <http://127.0.0.1:8000/>。API 文档在 <http://127.0.0.1:8000/docs>。

## 4. 从客户需求到交付包

1. 在左侧聊天框输入自然语言需求；首次发送会创建项目会话。
2. 点击“上传资料”，加入 PDF、Word、Excel、PPT、Markdown、文本或代码文件。文件会经过解析、脱敏、分块、向量化和项目隔离后进入 RAG。
3. 在“模型中心”选择 Mock、DeepSeek 或 OpenAI-compatible 供应商，填写 Base URL、模型 ID 和本次会话的 API Key，点击“测试连接并保存”。浏览器不会把 key 写入 `localStorage`，服务重启后需要重新连接。
4. 点击启动需求评审。产品经理会整理需求，交付总监会召集产品、架构、开发、QA 和 DevOps 进行需求避坑会，并生成 PRD、会议纪要、风险清单和项目规划。
5. 在需求确认门禁处选择“批准”或“退回”。退回时填写原因，工作流会回到对应的澄清节点，而不是丢失整个 run。
6. 通过确认后，系统按规划生成多语言项目代码、测试、架构说明和部署指南。Coder 只能写入计划内文件，Reviewer 和确定性安全规则会共同审核。
7. QA 失败会触发有限次数的开发返工；测试通过后进入最终发布门禁。用户再次批准后，下载页面会生成包含源代码、README、部署指南、测试报告和清单的 ZIP。

## 5. RAG 与 Qdrant

单进程本地演示默认使用持久化 Qdrant Local，数据写在 `.repopilot/rag/qdrant`，该目录不会提交到 Git。多人或多 Worker 部署请启动 Qdrant Server：

```powershell
docker compose up -d qdrant
# .env 中设置：
# REPOPILOT_QDRANT_URL=http://127.0.0.1:6333
```

生产 RAG 默认采用 BGE-M3 dense embedding、稀疏检索、RRF 融合和 BGE Reranker。模型第一次使用会下载到 `.repopilot/rag/models`，也可以通过 `REPOPILOT_RAG_MODEL_CACHE` 指定缓存目录。

## 6. 验证与排障

```powershell
# 不调用外部模型，完整跑单元测试
python -m unittest discover -s tests -v
python -m pip check

# 检索评测（使用临时目录）
python scripts\evaluate_retrieval.py

# 极小 DeepSeek 端到端冒烟测试，会产生 API 费用
python scripts\smoke_deepseek_tiny_e2e.py
```

如果页面无法打开，先检查 8000 端口和 uvicorn 日志；如果模型连接失败，先在模型中心点击验证，再确认 Base URL 以 `/v1` 结尾的第三方 OpenAI-compatible 服务是否允许当前网络访问。不要通过 `REPOPILOT_ALLOW_PRIVATE_MODEL_ENDPOINTS=1` 绕过 SSRF 防护，除非是在可信内网部署。

## 7. 运行边界

RepoPilot 负责需求理解、规划、代码草拟、测试建议和交付材料生成，不代表已经完成目标环境的生产上线。生成代码的构建、依赖、数据库迁移、密钥注入、容器扫描和真实发布必须在隔离的 CI/CD 或客户环境中执行；默认不会在 RepoPilot 宿主机执行客户仓库测试。
