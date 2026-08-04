# RepoPilot 架构说明

## 总体流程

```text
客户聊天/文件
    ↓
产品经理 Intake → 文件解析与项目 RAG
    ↓
交付总监主持需求避坑会（产品/架构/开发/QA/DevOps/安全）
    ↓
PRD + 会议纪要 + 风险清单 → Human 门禁 1
    ↓
项目规划 → 架构设计 → 任务拆解 → 多语言代码生成
    ↓
静态安全规则 + Reviewer → QA 测试 → 有限返工
    ↓
部署指导 → Human 门禁 2 → ZIP 交付
```

## Agent 与状态契约

工作流由 LangGraph 编排，每个节点只读取自己的输入字段并产出 Pydantic 结构化结果。产品经理负责澄清，不直接生成代码；总监负责会议和风险收敛；架构师选择技术路线；Coder 只能修改 Planner 允许的文件；QA 只负责验证；发布审查负责把结果整理成交付包。条件边根据需求是否完整、QA 是否通过、审批决定和错误状态进行路由。

状态中保留 `project_id`、`run_id`、阶段产物、证据 ID、审批状态、错误信息和 trace。SQLite checkpoint 让人工审批后可按 `run_id` 恢复；长流程的上下文由摘要、证据引用和受限文件片段组成，而不是把整个历史无限拼接给模型。

## 项目记忆与 RAG

上传文件以及 PRD、会议纪要、架构设计、测试报告和部署指南会写回项目知识库。解析器支持 PDF、Office、Markdown、文本和代码；敏感信息先脱敏，再按 token 窗口分块。检索阶段使用 dense + sparse 混合召回，RRF 融合后由 reranker 重排，并给每条证据分配稳定 Evidence ID。所有查询必须携带 `project_id`，避免跨项目泄漏。

## 模型路由与工具调用

模型中心维护供应商凭据、Base URL、模型发现结果和 Agent 路由快照。后端只在当前进程内保存 BYOK 凭据，工作流状态只保存 `credential_id`、供应商和模型，不把 key 写入 checkpoint、RAG 或 ZIP。DeepSeek、OpenAI、通义、豆包、智谱、Kimi、Gemini、Ollama 和自定义 OpenAI-compatible 服务统一为兼容适配层。

工具调用集中在文件沙箱、RAG 检索、补丁写入、测试执行和交付打包工具中。路径使用 `resolve_safe_path` 校验，写入使用 SHA-256 乐观锁、临时文件和原子替换；测试命令采用白名单、`shell=False`、超时和输出截断。LLM Reviewer 不能覆盖确定性危险规则。

## 可观测性与恢复

每个节点产生带阶段、角色、输入摘要、输出摘要和时间戳的 trace，前端通过 SSE 订阅事件。服务端会把事件持久化后再发送，客户端断线可按游标补齐。节点异常进入统一 error 状态并保留 checkpoint，避免返回裸异常；内存缓存有数量上限，生产多实例部署应将 checkpoint 和事件存储迁移到 PostgreSQL/Redis，并使用独立 Qdrant Server。
