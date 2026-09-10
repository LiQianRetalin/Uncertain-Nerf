# P02 服务器认证恢复后的续跑说明

## 当前不要执行训练

当前唯一阻塞发生在 SSH 认证阶段，服务器上没有启动 P02 命令，也没有消耗 smoke 或正式训练预算。不要手工逐条启动四项训练，也不要创建 Corner、OAC、Room factor2、Garden 或 PART 任务。

## 用户只需恢复合法 SSH 身份

1. 使用你已有的服务器私钥，或请服务器管理员把你新生成的公钥加入 `chenglong` 用户的 `~/.ssh/authorized_keys`。不要把私钥或密码粘贴到报告、Git、聊天或项目目录。
2. 如果使用私钥文件，可以在 `C:\Users\liqian\.ssh\config` 的 `Host 172.16.55.2` 段添加对应 `IdentityFile`；也可以由管理员按你们现有规范启用 Windows `ssh-agent`。本工作包不擅自生成、替换或上传身份凭据。
3. 在 Windows PowerShell 中运行：

```powershell
ssh -o BatchMode=yes -o ConnectTimeout=10 chenglong@172.16.55.2 "echo P02_SSH_OK"
```

正确标志是只看到 `P02_SSH_OK` 且没有密码提示、`Permission denied` 或超时。若仍失败，把完整错误文本交回当前 Codex 任务即可，不要反复开训。

## Git 时点

当前 P02 本地准备是一组明确改动，建议只做一次图形化提交并推送 `ru-part`。提交备注可写：

`P02: 准备外部方法流水线并记录服务器认证阻塞`

应包含 `.gitignore` 以及 `uncertain-nerf` 下新增的 P02 patches、tools、scripts、tests、reports；不应包含 `P02-external`、E 盘数据集、虚拟环境、模型缓存或训练输出。图形界面显示 `ru-part` 已与远端同步、没有待推送提交，即表示完成。

## 认证通过后由 Codex 连续执行

认证恢复后，把 `P02_SSH_OK` 的结果告诉 Codex。Codex 将先在服务器核对刚推送的精确提交、实际仓库路径、数据落点和 L20 GPU 清单；随后自动完成数据传输/复验、两套隔离环境、四项有限 smoke。只有四项 smoke 全部通过才会连续运行固定四项 seed42/30k、独立评测和严格计时。

后台执行后，进度查看命令格式为：

```bash
bash uncertain-nerf/scripts/p02_status.sh /服务器上的绝对路径/P02-work
```

其中绝对路径会在 Codex 完成服务器盘符和容量核验后明确给出，不要求用户猜测。成功结束标志是 `pipeline.json` 中 `stage` 为 `p02`、`status` 为 `COMPLETE`，且报告目录同时存在四行有效 summary、128 行逐图指标和 12 行计时记录。
