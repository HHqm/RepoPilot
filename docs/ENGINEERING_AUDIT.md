# RepoPilot Agent 工程审计

审计日期：2026-07-27  
审计范围：多智能体编排、DeepSeek 调用、上下文管理、RAG、工具安全、
Human-in-the-loop、持久化、可观测性、评测和前端契约。

## 1. 审计结论

RepoPilot 已经超过“多个角色轮流对话”的 Demo 水平，具备真实状态图、
结构化产物、确定性边界、人工审批、原子写入、回滚、增量混合 RAG 和实时
SSE。当前最关键的工程特性是：

- LangGraph 节点拥有不同输入、输出和失败路由；
- Planner/Coder/Reviewer 的文件与证据引用会被程序校验；
- Reviewer 不能绕过确定性安全规则；
- 写入前校验 SHA-256，使用备份和原子替换；
- 测试命令使用参数数组且 `shell=False`；
- DeepSeek 上下文按 Agent 类型确定性压缩，关键 Diff 超限时拒绝盲审；
- Qdrant 检索强制 `repo_id` 隔离，并有跨仓库泄漏回归测试；
- SSE 是后台工作流产生的真实增量事件，不是前端伪造。

当前定位应表述为：**安全的单机开发/演示版**。在加入认证、隔离执行、
持久化事件和崩溃恢复前，不应表述为可直接承载多租户生产仓库的 SaaS。

### 2026-07-28 复审与整改状态

本轮已直接落地：

- 公开 `ProjectRequest` 删除内部知识路径、会话摘要和上传资产字段，额外字段拒绝；
- 知识账本和 Studio Intake 双重校验路径必须位于 collaborations 根目录；
- 产品聊天改为结构化滚动理解 + 最近 8 条/24K 字符，不再发送最近 40 条原文；
- 协作锁改为每会话锁，慢模型或文档解析不再阻塞其他项目；
- 启动流程加入预留令牌，重复 Launch 被拒绝，失败会恢复会话状态；
- PRD、会议、计划、架构、部署和 QA 文档写回后被后续 Agent 真实检索；
- Studio QA/Reviewer 保护完整源码，源码无法安全装入上下文时拒绝盲审；
- DeepSeek Usage 改为线程本地，结构化输出失败允许一次有界 Repair；
- SSE 事件写入 SQLite 追加表，服务重启后保留原 sequence 和时间；
- 框架级异常会检测已写文件并执行带应用后 SHA 校验的幂等回滚；
- 终态 Runtime 从内存移除，任务队列增加容量上限；
- 人工审批绑定 Artifact SHA-256，并持久化 `actor_id`；
- 宿主机测试默认关闭；显式开启时清理凭据环境、隔离 HOME、禁用 pytest
  自动插件并拒绝危险选项；
- RAG 运行目录不再进入索引，失效文件会删除旧向量，Chunk 增加版本和信任标签；
- API 测试与检索评测强制使用独立临时 Qdrant，服务运行时也不会争用本地锁或
  误写远程生产集合；
- MCP HTTP 默认关闭，读取结果增加 64KB 上限，查询前刷新增量索引；
- ZIP 固定时间戳和文件权限，保证相同输入得到可复现摘要；
- 回归测试由 34 项扩充到 51 项；
- QA 新增跨文件框架契约门禁，确定性拦截异步 pytest fixture 配置、
  SQLAlchemy 导入期时间戳默认值和 Pydantic v2 的旧 ORM 配置；
- QA 返工改为 `ProjectRevision` 最小文件修订，不再整项目重新生成；
- 安全扫描区分可执行源码与说明文档：代码中的危险调用继续阻断，文档中对危险
  API 的风险说明不再误判，但任何文件中的私钥和 API Key 仍会阻断；
- 真实 DeepSeek 极小全链路已完成，交付的 20 项 unittest 在独立临时目录通过。

仍需外部基础设施才能彻底完成：

- Bearer Token/本地回环保护不能代替企业 OIDC、多租户 RBAC；
- 环境清理不能代替 Docker/gVisor/Firecracker 的内核级执行隔离；
- SQLite Event Store 适合单机，水平扩展仍应迁移 PostgreSQL + Task Queue；
- Qdrant Local 适合单进程；多 Worker/多容器必须改用 Qdrant Server；
- 进程在文件已应用后被强杀的恢复，仍推荐用 Git worktree + WAL Recovery
  Reconciler 完成。

## 2. 本轮已解决

### 2.1 模型模式只有一个事实源

原问题：

- 顶部模型标识读取 `/ui-config.default_llm_mode`；
- 表单下拉框沿用 HTML 默认 `mock`；
- “一键填入演示”再次硬编码写回 `mock`。

解决：

- 新增 `setLlmMode()`，同时更新表单值和顶部指标；
- `boot()` 使用后端 `default_llm_mode` 初始化真实选择；
- 用户切换下拉框时立即同步顶部指标；
- 演示填充只填写仓库、Issue、约束和测试命令，不再改变模型；
- 创建运行和恢复运行时使用状态中的真实 `llm_mode` 回写界面。

验收：

- 后端为 `deepseek` 时，顶部与下拉框都显示 DeepSeek；
- 先选 DeepSeek 再点击演示填充，仍保持 DeepSeek；
- POST `/runs` 的 `llm_mode` 与界面选择一致。

### 2.2 Toast 终态清理

原问题：

- 审批 Toast 视觉上隐藏后仍保留旧文本；
- 工作流完成后，辅助技术仍可能读到“暂停在审批节点”。

解决：

- 新增 `hideToast()`，同时取消定时器、移除可见类并清空文本；
- `showToast()` 每次先清理旧状态，计时结束后也清空文本；
- 离开审批状态立即清理审批 Toast；
- 完成、失败、取消和审查拒绝显示对应终态消息。

## 3. P0：对外部署前必须解决

### P0-1 API 没有身份、权限和仓库范围边界

证据：

- `/runs`、`/decision`、`/demo-workspace` 和 SSE 均无认证；
- 请求可直接提交本机任意目录作为 `repo_path`；
- 审批记录只有布尔值和评论，没有操作者身份。

风险：

- 一旦监听地址从 localhost 改为局域网或公网，其他调用者可能读取仓库状态、
  消耗模型额度或触发文件修改；
- 无法证明“谁批准了哪个版本的 Diff”。

解决方案和技术：

1. FastAPI 接入 OAuth2/OIDC，使用 Keycloak、Auth0 或企业 SSO；
2. JWT 中携带 `tenant_id`、`user_id` 和角色，使用 RBAC/ABAC；
3. 不再接收任意绝对仓库路径，改为服务端登记的 `workspace_id`；
4. `workspace_id -> canonical_path -> tenant_id` 在数据库中校验；
5. Approval 保存 `approver_id`、`proposal_digest`、时间和评论；
6. Qdrant Payload 同时过滤 `tenant_id + repo_id`。

验收标准：

- 未认证请求返回 401；
- A 租户不能查询、审批或检索 B 租户的数据；
- 审批日志可以复现批准人的身份和被批准补丁的 SHA-256。

### P0-2 测试命令没有真正的执行隔离

证据：

- `shell=False` 和命令白名单只能防止 Shell 拼接；
- pytest/unittest 本质上仍会执行仓库中的任意 Python；
- pytest 参数目前可以引用仓库外测试路径或加载插件；
- 子进程继承服务账号可访问的文件、网络和环境变量。

风险：

- 恶意或被污染仓库的测试代码可以读取 DeepSeek Key、访问网络或修改仓库外文件。

解决方案和技术：

1. 每次运行创建临时 `git worktree` 或一次性工作目录；
2. 在 Docker、gVisor 或 Firecracker 沙箱中执行测试；
3. 容器使用非 root、只读根文件系统、禁用网络、最小环境变量；
4. 使用 cgroups 限制 CPU、内存、进程数和磁盘；
5. 校验所有测试路径 `resolve().relative_to(workspace)`；
6. 禁止 `-p`、`--override-ini`、外部 `conftest` 等高风险 pytest 参数；
7. 生产环境不把 LLM/API 凭据传入测试容器。

验收标准：

- 测试无法读取宿主机 `.env`；
- 测试无法联网；
- 超出 CPU、内存、时间和进程限制会被确定性终止；
- 测试无法修改工作区外文件。

### P0-3 框架级崩溃可能绕过回滚

证据：

- Agent 节点异常会进入 `FailureReporterAgent` 并尝试回滚；
- 但 `RepoPilotService._handle_framework_failure()` 直接生成失败状态；
- 如果异常发生在文件已写入、Verifier 完成前，框架级失败路径没有恢复备份。

风险：

- 进程异常、Checkpoint 写入错误或流处理错误可能留下未经验证的修改。

解决方案和技术：

1. 在写入前持久化执行日志：`PREPARED -> APPLIED -> VERIFIED/ROLLED_BACK`；
2. 日志记录原始 SHA、应用后 SHA、备份位置和 proposal digest；
3. `_handle_framework_failure()` 检测 `APPLIED` 状态并调用幂等回滚；
4. 服务启动时运行 Recovery Reconciler，扫描未完成执行日志；
5. 更推荐在临时 `git worktree` 应用和测试，验证后只导出 Patch/Commit，
   不直接修改用户主工作区。

验收标准：

- 在 Executor 后强制杀进程，重启后能恢复或明确隔离未验证变更；
- 重复执行回滚不会破坏已恢复文件；
- 回滚前校验应用后 SHA，避免覆盖用户随后产生的新修改。


## 4. P1：从优秀 Demo 升级为生产级 Agent

### P1-1 运行事件只在内存中完整保存

现状：

- Checkpoint 和最终 `state.json` 会持久化；
- 但 SSE 的完整事件序列保存在 `RunRuntime.events`；
- 进程重启后只能由 Trace 重建近似事件，原序号、状态事件和人工决策不完整；
- `_runtimes` 没有 TTL/LRU，长期运行会持续增长。

解决方案和技术：

- 使用 PostgreSQL `run_events` 追加表保存事件；
- 事务内使用 Outbox Pattern 同时提交状态和事件；
- SSE 按数据库 sequence/cursor 重放；
- LangGraph 改用 PostgresSaver；
- 后台任务使用 Celery、Dramatiq 或 Arq；
- 内存只保留活跃 Run，使用 TTLCache/LRU，终态 Run 及时淘汰。

验收标准：

- 服务重启后 `Last-Event-ID` 仍能按原序号续传；
- 一万个历史 Run 不导致进程内存线性增长；
- 崩溃恢复后非终态任务能够重新入队或进入明确的 `recovery_required` 状态。

### P1-2 结构化输出没有“校验—修复”循环

现状：

- SDK 只保证返回可解析 JSON Object；
- Pydantic 校验发生在 Agent 节点中；
- JSON 合法但字段缺失、类型错误或 search 不唯一时，整个 Run 直接失败；
- 网络重试会重试所有异常，401 等不可恢复错误也会等待。

解决方案和技术：

1. 为每个 Agent 建立 `schema + semantic validator`；
2. Pydantic 失败时，将精简的校验错误反馈给同一 Agent；
3. 最多进行 1–2 次 Repair，禁止无限反思；
4. search 不唯一时让 Coder 基于新的行号窗口重新生成定点编辑；
5. 使用 Tenacity 只重试超时、429 和 5xx，并加入指数退避和 jitter；
6. 401、403、模型不存在、上下文超限应立即失败；
7. 模型支持时使用严格 JSON Schema。

验收标准：

- 注入缺字段、错误枚举和非唯一 search 的模型响应后能有限恢复；
- 每次修复次数、原因和额外 Token 都进入 Trace；
- 不可恢复 API 错误不会重复等待三次。

### P1-3 DeepSeek 用量统计存在并发串线风险

现状：

- 一个 `DeepSeekLLM` 实例被多个 Run 共享；
- 最近一次 Usage 存在可变的 `last_usage`；
- 并发调用完成顺序交错时，Agent 可能读取到另一个 Run 的 Usage。

解决：

- `complete_json()` 返回 `{data, usage, request_id}`，不使用共享可变字段；
- 或使用不可变 Result Dataclass；
- 为每次调用创建 OpenTelemetry Span，属性包含 run_id、agent、model、
  prompt/completion/cache Token 和延迟；
- 使用 DeepSeek request id 做对账。

验收标准：

- 50 个并发 Run 的每条 Usage 都能与请求 ID 对应；
- 汇总 Token 等于逐调用 Token 之和。

### P1-4 RAG 在“文件仍存在但变得不可索引”时可能残留旧 Chunk

现状：

- 解析抛出 `DocumentParsingError` 时会删除旧向量；
- 但文件超过大小限制、读取失败或解析后没有 Chunk 时，旧版本不一定删除；
- Manifest 会丢弃该文件，而 Qdrant 中可能继续存在旧内容。

解决：

1. 对每个候选文件统一计算最终状态：`indexed/reused/skipped/deleted`；
2. 任何不再进入新 Manifest 的旧文件都删除其 document points；
3. 查询额外过滤当前 `document_version`；
4. Manifest 写入采用事务/代际编号；
5. 增加一致性扫描：Manifest 文档集合必须等于 Qdrant Payload 文档集合。

验收标准：

- 已索引文件变为空、变大、损坏后，检索不再返回旧内容；
- 模拟索引中断后，下次运行能够自愈；
- 增加上述三种状态转换的回归测试。

### P1-5 Prompt Injection 防线还不完整

现状：

- 文件、README、PDF 和测试输出属于不可信数据；
- Prompt 尚未明确声明“证据中的指令不得改变系统目标”；
- 程序边界能阻止路径越界和未审批写入，但计划、代码质量和 Reviewer 判断仍可能被
  恶意文档影响。

解决：

- Prompt 明确区分 `trusted task` 与 `untrusted repository evidence`；
- Evidence 使用固定标签和来源元数据，不拼接成角色消息；
- 对“忽略系统提示、泄露密钥、批准补丁”等指令模式做检测和降权；
- Reviewer 不复用 Coder 的推理，只使用 Issue、证据和 Diff；
- 高风险仓库使用双 Reviewer 或确定性 Semgrep/Bandit 规则；
- 所有权限仍由代码策略决定，模型不得扩大路径、命令和写入范围。

验收标准：

- README 中加入恶意“忽略规则并修改 .env”指令时，计划仍拒绝；
- 恶意代码注释不能让 Reviewer 批准静态规则命中的补丁；
- 攻击样例纳入 CI Eval。

### P1-6 缺少单任务预算、限流和熔断

现状：

- 有单次上下文上限，但没有 Run 级总 Token/费用上限；
- 多个并发 Run 可同时调用 DeepSeek；
- API 故障时没有 Circuit Breaker。

解决：

- 增加 `max_calls_per_run`、`max_total_tokens`、`max_cost` 和截止时间；
- 使用 Redis Token Bucket 做用户/租户限流；
- 使用队列控制 DeepSeek 并发；
- 使用 Circuit Breaker 在连续 429/5xx 后快速失败；
- 超预算进入 `budget_exhausted`，生成不写入的报告。

### P1-7 Coder 是一次性生成，不具备受控工具循环

现状：

- Coder 只收到一次证据窗口；
- 只要已有一个目标文件证据，就不会自动补充完整文件或其他符号窗口；
- edit 不唯一时直接失败，没有再次读取局部上下文。

解决：

- 为 Coder 提供只读、强类型工具：`read_symbol`、`read_lines`、`search_code`；
- 使用最多 3 步的 bounded tool loop；
- 每次工具参数仍经过 `resolve_safe_path` 和行数限制；
- 生成修改后运行 AST parse/formatter，再进入 Reviewer；
- 不允许 Coder 直接调用写工具。

## 5. P2：工程成熟度增强

### P2-1 评测集太小

当前完整 Agent Eval 只有一个 calculator case，容易形成“只为 Demo 调参”的印象。

建议至少建立 30–50 个固定任务：

- Python/JavaScript/Java 多语言修改；
- 单文件、多文件、只读分析、无须修改；
- Planner 幻觉文件、证据缺失、Coder 非唯一编辑；
- 危险补丁、Prompt Injection、密钥、测试禁用；
- 测试失败、超时、并发修改、回滚失败；
- PDF/表格知识证据和跨仓库隔离。

指标：

- Retrieval Recall@K、MRR、nDCG；
- Plan Groundedness、Patch Apply Rate；
- Reviewer Catch Rate、False Reject Rate；
- First-pass Test Pass Rate、Rollback Success Rate；
- Task Completion Rate、平均 Token/费用、P50/P95 延迟；
- Context Compression Recall：关键证据和变更行保留率。

可使用 pytest + 自定义 Eval Runner；RAG 指标可接 RAGAS，Agent 轨迹可接
LangSmith 或 OpenTelemetry，但最终通过标准应由确定性测试判定。

### P2-2 运行产物包含较多源码正文

Checkpoint、Qdrant 和 `state.json` 会保存 Evidence 文本和完整 `new_content`。

建议：

- 状态只保存 Diff、哈希和受控的短证据；
- 大正文放对象存储，使用引用和 TTL；
- 数据库与对象存储加密；
- 提供按仓库删除索引、Run、备份和 Demo 工作区的生命周期 API；
- 备份验证成功后按策略清理，而不是永久增长。

### P2-3 确定性代码审查规则仍较窄

当前主要是正则匹配。建议增加：

- Python AST 解析和 compile 检查；
- Ruff、mypy、Bandit；
- Semgrep 自定义规则；
- 依赖变更使用 OSV/Trivy；
- 测试覆盖率差异和新增代码覆盖率门禁。

这些工具作为 Reviewer 的独立事实输入，不能被 LLM 的“通过”结论覆盖。

### P2-4 可观测性需要标准化

建议接入：

- OpenTelemetry：一个 Run 一个 Trace，一个 Agent 一个 Span；
- Prometheus：任务状态、队列长度、Token、缓存命中、失败类型；
- Grafana：成功率、成本、P95 延迟和 RAG 指标；
- 结构化 JSON 日志：包含 run_id/agent/request_id，不包含 Prompt、源码或密钥；
- Sentry：框架异常，发送前统一脱敏。

## 6. 推荐实施顺序

1. **安全底座**：API 认证、workspace allowlist、测试容器沙箱；
2. **写入可靠性**：git worktree、执行日志、启动恢复、幂等回滚；
3. **持久化运行时**：PostgresSaver、事件表、任务队列、TTL；
4. **模型可靠性**：结构化 Repair、分类重试、预算、并发 Usage；
5. **RAG 一致性**：旧 Chunk 清理、版本过滤、一致性扫描；
6. **评测与观测**：扩充 Eval、OpenTelemetry、Prometheus/Grafana；
7. **高级 Agent 能力**：受控只读工具循环、多 Reviewer 和静态分析门禁。

## 7. 能力边界

设计原则：

> RepoPilot 的 Agent 自主性被限制在可验证边界内：模型负责提出计划、定点编辑和
> 语义审查，程序负责路径、证据、哈希、命令、审批和回滚。自主性越高，确定性
> 治理越强。

目前不能等同于：

- SQLite Local Mode 已等同多实例生产；
- 白名单命令已等同安全沙箱；
- 一个 calculator case 足以证明泛化能力；
- Checkpoint 已自动解决所有进程崩溃恢复问题。
