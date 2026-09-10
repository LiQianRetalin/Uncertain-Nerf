# REPORT_P02

## 结论

P02 当前状态为 `BLOCKED_SERVER_SSH_AUTHENTICATION`，不是训练失败，也不是实验结论。已完成 P01 实际提交核对、本地共同数据逐文件校验、两个固定外部版本审计、隔离执行流程和最小适配补丁；服务器在认证阶段拒绝连接，因此服务器同步、远端共同数据校验、隔离环境、有限 smoke、四项 seed42/30k 正式训练、独立评测及计时均未开始。

没有伪造、补写或引用作者协议数字作为 Reproduced 结果。P02 到此停止，未启动 Corner、OAC、Room factor2、Garden 或 PART 修补。

## 已核验事实

- 当前分支仍为 `ru-part`。本地 `HEAD` 与 `origin/ru-part` 均为 P01 实际提交 `df2b28093d9f946fcc6f66f2abb6ac058824449c`，提交说明为“P01: 冻结GS-RU资产与外部比较共同协议”。
- P01 的 `protocol_manifest.json` 内嵌 `4b548999...` 是生成清单时的父提交快照；实际承载并已推送 P01 资产的提交是 `df2b2809...`，两者角色不同，不构成远端未同步。
- Android：263 张图像逐一通过 SHA-256，122 train / 19 test / 122 excluded，1007×755，初始点 112790。
- Patio-High：267 张图像逐一通过 SHA-256，221 train / 45 test / 1 excluded，1007×755，初始点 36922。
- 两个场景的相机、位姿、投影、点云和 `cameras.bin/images.bin/points3D.bin` 哈希均通过 P01/P02 联合复验。
- RobustSplat 固定为 `a130281d...`，SpotLessSplats 固定为 `0caae3cc...`；最小补丁反向应用检查、Python 语法检查均通过。

## 唯一实际阻塞

对 `chenglong@172.16.55.2` 使用 BatchMode 和 10 秒连接超时进行只读登录检查，Windows OpenSSH 返回码 1：`Permission denied (publickey,password).`。本机 `.ssh/config` 只有主机和用户名配置，没有可用私钥；Windows `ssh-agent` 停止且禁用。失败发生在远端命令启动之前，因此不能诚实声称服务器仓库、数据、GPU、环境或训练已经检查。

## 已就绪的续跑边界

- GPU 选择严格按 6→7→0→1→2→3→4→5，并要求无计算进程、显存占用不高于 2048 MiB、利用率不高于 10%。
- smoke 固定四项各 100 更新；当前已用 0/800。任一 smoke 不通过则不启动正式训练。
- 正式身份固定四项、seed42、30,000 更新；技术失败最多允许一次受审计重试，脚本不会自动重训，也不会调参或加 seed。
- 四项通过后才以统一独立评测器评 PSNR/SSIM/LPIPS，并执行 10 warmup + 完整测试集 3 次原生渲染计时。

详细证据见 [p02/preflight.json](p02/preflight.json)、[p02/local_common_preflight.json](p02/local_common_preflight.json)、[p02/run_ledger.csv](p02/run_ledger.csv)、[p02/protocol_diff.md](p02/protocol_diff.md)、[p02/status.json](p02/status.json) 和 [p02/RESUME_AFTER_SSH_CN.md](p02/RESUME_AFTER_SSH_CN.md)。
