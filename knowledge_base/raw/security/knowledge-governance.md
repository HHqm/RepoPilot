# 知识库安全与治理

## 数据边界

索引器拒绝 `.env`、SSH 私钥、credentials、secrets、版本控制目录和依赖目录。
符号链接解析后必须仍然位于目标仓库内部。
允许文件中的 API Key、Access Token、密码赋值和私钥块会在 Embedding 与
Qdrant Upsert 之前替换为 `[REDACTED]`，索引报告记录脱敏次数。

## 版本与删除

文件内容使用 SHA-256 做幂等判断。修改文件时先删除该文档的旧 Chunk，再写入
新版本；删除源文件时同步删除 Qdrant 中对应的全部向量，避免“幽灵知识”。

## 权限

当前本地版按 `repo_id` 强制过滤。多租户部署还必须加入 `tenant_id`、ACL 和
用户身份条件，并为所有权限字段创建 Payload Index。生成答案前后都要保留
来源路径、页码或行号。
