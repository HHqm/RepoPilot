# 安全与隐私边界

## 不提交秘密

- 真实 `.env`、DeepSeek/API keys、Bearer token、数据库密码和生产配置只放在本机或密钥管理服务。
- `.env` 已被 `.gitignore` 忽略，`.env.example` 只保留空值和说明。
- 不要把密钥放进前端静态文件、截图、测试 fixture、checkpoint、RAG 文档、ZIP 或 GitHub issue。

## 文件与代码执行

上传文件和生成代码默认只能在配置的 workspace 根目录内操作；绝对路径、路径穿越、敏感目录和敏感文件会被拒绝。默认禁止在宿主机运行客户仓库测试，生产构建应放在隔离 CI、容器或沙箱中。即使打开 `REPOPILOT_ALLOW_HOST_TEST_EXECUTION=1`，也只适用于明确可信的本地仓库，不等于容器级隔离。

## 模型端点与 BYOK

自定义 OpenAI-compatible Base URL 默认拒绝本机和私网地址，防止 SSRF。仅在可信内网且明确知道风险时设置 `REPOPILOT_ALLOW_PRIVATE_MODEL_ENDPOINTS=1`。用户在模型中心输入的 key 只用于当前服务进程的连接测试和调用，服务端状态保存的是凭据 ID，不是明文 key；服务重启后需要重新连接。

## 上线检查清单

1. 使用 HTTPS、反向代理和 API Token/OIDC 保护 FastAPI。
2. 将 checkpoint、事件、RAG 和临时文件放到独立有权限控制的存储。
3. 在隔离 CI 中执行依赖扫描、SAST、容器扫描和真实构建。
4. 对生成的迁移、网络访问、文件权限和第三方 SDK 做人工代码审查。
5. 轮换曾经出现在日志、聊天记录或错误报告中的密钥。
