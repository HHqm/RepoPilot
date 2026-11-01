# Changelog

格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，版本号遵循
[语义化版本](https://semver.org/lang/zh-CN/)。

## [Unreleased]

### Agent 设计

设计说明见 `docs/AGENT_DESIGN.md`。

- 仓库维护 Agent 新增有界自我修正：Reviewer 拒绝后，Coder 带着 findings 和被拒 diff 修订
  （`REPOPILOT_MAX_REVIEW_REVISIONS`，默认 1 轮）。测试失败后先回滚，Reflection 记录根因，
  再从检索开始重新规划（`REPOPILOT_MAX_REPAIR_ATTEMPTS`，默认 2 次）。每个新补丁都要重新
  人工审批。Reflection 原有的 `retry_recommended` 字段第一次真正生效。
- Retrieval Agent 改为多查询规划：Issue、从 Issue 中抽取的代码符号、上一次失败的
  traceback，三路结果用 RRF 融合。各查询的命中路径和阶段耗时写入 trace。
- 软件工作室：发布审查拒绝、客户在验收时提出修改，都会在返工预算内回到开发团队，
  不再直接结束项目。返工节点按触发来源（QA / 发布审查 / 客户）组织反馈。
- 人工决策契约新增 `terminate`，区分“退回修改”和“终止项目”。工作台新增“终止项目”按钮。
- 两个 Reporter 汇总 `run_metrics`：LLM 调用次数（含 schema 修复和工具修复）、
  token 用量、各 Agent 耗时。

### RAG

设计与评测见 `docs/RAG.md`。

- 检索评测集从 5 条扩充到 34 条，分 5 类查询；新增 6 种配置的消融实验
  （dense / sparse / RRF / DBSF / +rerank / +diversify）、分类型 MRR、各阶段耗时和 Markdown 报告；
  `--corpus-root` 支持在冻结语料上做 A/B。
- `SearchOptions` 可以单独开关每个检索阶段；`RetrievalStats` 记录查询编码、召回、重排耗时和候选漏斗。
- 真实模型（BGE-M3 + bge-reranker-v2-m3）消融：生产配置 Hit@1 0.853、Recall@5 0.897、MRR 0.882；
  报告见 `docs/RAG_EVALUATION.md`。
- 代码感知稀疏分析器 v3：保留完整标识符，同时拆出 snake/camelCase 组成部分，并做英文复数还原。
  在冻结语料上 A/B：稀疏检索单独使用时 MRR 0.613 → 0.657，生产配置不变。
- 两级增量索引：大小和 mtime 都没变的文件直接复用，不读也不哈希；最近 2 秒内修改过的
  文件不信任 mtime（与 git 的 racy-clean 处理相同）。
- 内容寻址的向量缓存（SQLite、float32、按最近使用淘汰）：编辑一个函数后重新索引，
  只有变化的块会送进模型。
- 索引自愈：每个进程首次索引某仓库时，核对清单块数和 Qdrant 实际点数，不一致就重建。
  取代了只对第一个仓库生效的全局 `collection_was_created` 标志。

### 性能

数据由 `scripts/benchmark_runtime.py` 在同一台 Windows 11 开发机上对比 0.6.0 测得。

- LangGraph 工作流只编译一次，每个会话 `copy()` 并绑定独立 SQLite checkpointer；
  冷启动运行状态查询 17.4 ms → 0.73 ms。
- SSE 事件库改用加锁复用的长连接与 WAL `synchronous=NORMAL`，内存序号 O(1)
  递增，断线续传用二分定位未读事件；单条事件发布 12.3 ms → 0.11 ms。
- RAG 仓库扫描改用 `os.walk` 在遍历时剪枝 `node_modules`、`.git`、`.venv` 等目录，
  且不跟随符号链接目录（避免别名重复索引）；含 3000 个依赖文件的仓库 800 ms → 7.8 ms。
- 分块器每行只分词一次；2 万行源码切分 155 ms → 99 ms。
- BGE-M3 稠密向量用 `ndarray.tolist()` 转换，去掉逐元素 Python 循环。

### 重构

- `DeepSeekLLM` 与 `OpenAICompatibleLLM` 抽取公共基类，统一有界重试、线程本地
  用量统计与 JSON 解析；DeepSeek 也能容忍 Markdown 代码块包裹的 JSON。
- `Settings.from_env` 改用 `_env_int/_env_float/_env_bool` 辅助函数；空值回落默认值，
  非法数字会在错误信息中指出变量名。
- `RepoPilotService.start/start_project` 合并重复的运行注册与提交逻辑。

### 工程化

- 新增 GitHub Actions：Ruff、Python 3.11/3.12 测试矩阵、Mock 端到端评测、检索质量
  门禁和 Docker 构建。
- 引入 Ruff 配置并清理全部告警（未使用导入、导入排序、过时写法等）。
- `scripts/evaluate_retrieval.py` 新增 `--min-recall/--min-mrr` 回归门禁。
- 新增 `scripts/benchmark_runtime.py` 可复现性能基准。
- Dockerfile 改为依赖层与源码层分离、非 root 用户运行、增加 `HEALTHCHECK`；新增
  `.dockerignore`，确保 `.env` 与运行数据不进入构建上下文。
- Docker Compose 将 API 端口绑定到回环地址，并使用命名卷保存运行数据。
- 版本号单一来源：`pyproject.toml` 与 API 均读取 `repopilot.__version__`，测试校验
  前端缓存版本不漂移。
- 设计、审计和端到端报告移入 `docs/`。

### 修复

- PRD 被连续退回到预算用尽时，运行以非终态 `revising_requirements` 结束，SSE 和等待方会永久挂起。
  现在映射为 `cancelled`；服务端新增不变量检查：图执行结束时状态必须是终态。
- `repopilot demo` 在默认安全配置下无法运行：临时演示目录不在 `REPOPILOT_WORKSPACE_ROOTS` 内，
  宿主机测试也默认关闭。现在只信任演示自身的临时副本。
- 上下文压缩器的 Coder 字段白名单会丢掉 `tool_error`，导致定向编辑修复时模型看不到失败原因。
- Schema 修复调用的 token 用量会覆盖首次调用的用量，现在分开记录。
- nDCG@k 的理想值原先按返回条数计算，结果偏高；现在按 k 个位置计算。
- 自动生成的项目名会在句子中间截断（如“…目标用户”），现在优先取出产品名所在的分句。
- `/` 和 `/health` 支持 HEAD 请求，负载均衡和探活不再返回 405。
- `requirements.txt` 缺少 `python-multipart`，按该文件安装时上传接口无法注册。
- `test` 可选依赖误写为 `httpx2`，FastAPI `TestClient` 实际依赖 `httpx`。

### 测试

- 新增依赖目录剪枝、分块窗口覆盖与预算、SSE 断线续传、图模板复用、LLM 重试/
  快速失败/用量统计、配置解析等回归测试。

## [0.6.0]

- 客户协作式多智能体软件公司：产品对话、项目 RAG、需求评审、双人工审批、多语言
  开发、QA 返工与 ZIP 交付。
