# RepoPilot DeepSeek 端到端测试报告

测试日期：2026-07-28  
Python 环境：`D:\envs\lstd`（Python 3.11）  
服务地址：`http://127.0.0.1:8000`  
模型配置：`deepseek-v4-pro`

## 1. 最终结论

RepoPilot 的真实 DeepSeek 小型全链路已经跑通。最终成功 Run：

- Session ID：`c4130c3fbb1f49668d8682d1f3744ef9`
- Run ID：`b4478a2debdd4ba893cd34002369efbb`
- 最终状态：`completed`
- Agent Trace：16 条
- Human-in-the-loop：2 次真实服务端决策
- 持久事件：23 条 `run_status`、16 条 `agent_trace`、2 条
  `human_decision`
- 事件序号：严格递增且无重复
- 工作流耗时：约 495 秒

本次不是 Mock。健康检查、每个 Agent 的上下文元数据和 Trace 中均记录了真实
DeepSeek 用量：

| 指标 | 数值 |
| --- | ---: |
| Prompt tokens | 80,435 |
| Completion tokens | 13,124 |
| Total tokens | 93,559 |
| Prompt cache hit tokens | 1,408 |
| Prompt cache miss tokens | 79,027 |

## 2. 测试项目

测试项目是一个零第三方依赖的纯 Python 折扣计算器，只有一个业务函数：

```text
calculate_total(unit_price_cents, quantity, discount_percent) -> int
```

规格明确了整数与布尔值校验、参数边界、整数分四舍五入公式、禁止网络/文件写入/
子进程，以及 `unittest` 验收要求。规格先作为 Markdown 上传并进入项目 RAG。

DeepSeek 最终生成 4 个项目文件：

```text
calculator.py
tests/__init__.py
tests/test_calculator.py
README.md
```

交付 ZIP 共 11 个文件，还包含 PRD、需求会议纪要、项目计划、架构、QA 报告、
部署指南和 `MANIFEST.json`。

## 3. 实际经过的流程

1. DeepSeek 健康检查；
2. 创建客户项目会话；
3. 上传规格并完成解析、切块和向量入库；
4. 客户与产品经理真实对话，第一轮达到 `ready_for_workshop=true`；
5. Knowledge Agent 从上传资料和内部知识检索；
6. 产品经理形成结构化 PRD；
7. 交付总监主持需求评审、风险规避和工作拆解；
8. 第一次 Human-in-the-loop 按产物 SHA-256 摘要批准需求；
9. 项目经理规划预算、阶段和退出条件；
10. 架构师选择 Python 3.11 标准库方案；
11. 开发团队生成源码、测试和 README；
12. DevOps 形成运行、验收、排障和回滚指南；
13. QA 完成静态、契约、安全、文档和追溯检查；
14. 独立 Reviewer 审查交付；
15. 第二次 Human-in-the-loop 按新的产物摘要批准发布；
16. 原子写入项目、完整性检查、生成 ZIP；
17. 在临时目录解压 ZIP，经路径、依赖和 AST 白名单后实际运行生成的
    `unittest`。

最终生成代码运行结果：20/20 项测试通过，退出码为 0。

## 4. 真实测试发现并解决的问题

### 4.1 SSE 验收脚本事件名不一致

测试脚本最初期待 `status`，服务端真实契约是 `run_status`，导致已完成工作流被
测试脚本误报失败。脚本已改为按后端真实契约验收。

### 4.2 产品经理可能在已无问题时继续追问

真实模型曾出现 `open_questions=[]` 但 `ready_for_workshop=false`，下一轮又产生
新问题。现在由确定性 Discovery Gate 统一修复矛盾，并在核心信息完整后最多三轮
结束客户澄清；可选实现细节转入评审会假设，不再无限追问。

### 4.3 QA 返工整项目重生成导致缺陷漂移

早期真实 Run 在 QA 失败后让开发 Agent 重写全部文件，两轮返工会修复旧问题又
引入新问题。现在改为结构化 `ProjectRevision`：

- 只返回需要修改的完整文件；
- 明确 `delete_paths`；
- 记录每个已处理缺陷；
- 保留全部无关文件；
- 源码被上下文截断时拒绝编辑。

### 4.4 LLM QA 会漏掉框架级运行缺陷

较大的真实生成项目曾被 LLM QA 判为通过，但二次检查发现：

- `pytest-asyncio` 严格模式下异步 fixture 误用 `@pytest.fixture`；
- SQLAlchemy 时间戳默认值在模块导入时被求值；
- Pydantic v2 仍使用旧 `orm_mode`。

现在这些规则由确定性跨文件框架契约门禁检查，结果优先于 LLM，模型无法批准掉。

### 4.5 文档中的危险 API 名称被误当成代码调用

第一次极小 Run 的源码和 QA 均通过，但需求会议文档用 `os.system()` 举例说明风险，
安全扫描把文字说明当成真实源码调用，发布被错误拒绝。

现在扫描分为两类：

- 私钥、API Key 等密钥模式：所有文件都扫描并阻断；
- `eval`、`os.system`、`shell=True` 等可执行模式：只在代码/配置中阻断。

对应安全回归测试已补充。

### 4.6 极小运行验收白名单过严

成功 Run 的测试仅用 `sys.path` 和 `os.path` 定位项目根目录，首版验收器把 `sys`
整体拒绝。现在只放行 `sys.path.insert` 和 `os.path.abspath/dirname/join`，其他
OS、网络、子进程和动态执行调用仍拒绝。修正后 20 项测试全部运行通过。

## 5. RepoPilot 本身的回归结果

- `python -m unittest discover -s tests -v`：51/51 通过；
- `python -m pip check`：无依赖冲突；
- `python -m compileall -q src tests scripts`：通过；
- RAG 离线评测：
  - Recall@5：1.0
  - MRR：0.8667
  - NDCG@5：0.90
- Mock 端到端评测：
  - 任务完成率：1.0
  - 检索命中率：1.0

## 6. 仍然存在的边界

- 极小测试使用 AST 白名单、清理后的环境变量、临时目录、`shell=False` 和超时，
  但不等同于内核级网络/文件系统隔离；
- 通用生成项目默认仍只做静态预检，容器、数据库、真实网络和第三方系统必须在
  Docker/gVisor/Firecracker 或客户 CI/目标环境验证；
- Qdrant Local 适合单进程本地模式，多 Worker/多容器应使用 Qdrant Server；
- SQLite Event Store 适合单机，水平扩展应迁移 PostgreSQL 与任务队列；
- 当前 API Key 曾在聊天中公开，必须在 DeepSeek 控制台撤销并更换，不能继续视为
  安全密钥。

## 7. 可复现命令

先启动服务：

```powershell
cd D:\gptbot\RepoPilot
D:\envs\lstd\python.exe -m uvicorn repopilot.api:app --host 127.0.0.1 --port 8000
```

再运行真实极小全链路（会消耗 DeepSeek 配额，并自动批准本地测试项目的两个门禁）：

```powershell
cd D:\gptbot\RepoPilot
D:\envs\lstd\python.exe scripts\smoke_deepseek_tiny_e2e.py
```

本地回归：

```powershell
D:\envs\lstd\python.exe -m unittest discover -s tests -v
D:\envs\lstd\python.exe -m pip check
D:\envs\lstd\python.exe scripts\evaluate_retrieval.py
D:\envs\lstd\python.exe scripts\evaluate.py
```
