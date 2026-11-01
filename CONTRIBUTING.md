# Contributing to RepoPilot

## 本地检查

```powershell
pip install -e ".[dev]"
ruff check src tests scripts
python -m unittest discover -s tests -v
python -m pip check
node --check src\repopilot\web\app.js
```

以上检查与 CI（`.github/workflows/ci.yml`）一致；涉及检索或运行时热点的改动，请同时运行
`python scripts\evaluate_retrieval.py --min-recall 0.65 --min-mrr 0.6` 与
`python scripts\benchmark_runtime.py`，并在 PR 中附上前后对比。

默认测试使用 MockLLM，不需要外部 API Key，也不应把真实密钥加入测试。新增 Agent 时，请同时补充输入/输出模型、路由条件、失败路径、trace 字段和最小单元测试；新增工具时，请补充路径、参数、超时和权限边界测试。

## 提交规范

- 小步提交，提交信息使用 `feat:`, `fix:`, `docs:`, `test:` 等前缀。
- 不提交 `.env`、`.repopilot/`、模型缓存、运行日志、生成 ZIP 或 `__pycache__`。
- Pull Request 需要说明影响的工作流阶段、人工审批行为、RAG 隔离和安全边界变化。
