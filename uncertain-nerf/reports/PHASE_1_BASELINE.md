# PURI-GS Phase 1：B0/B1 配置与启动检查

## 配置结论

三份配置使用 JSON-compatible YAML，可用 Python 标准库读取，无新增 PyYAML 依赖。

| 字段 | B0 | B1 | A1 |
|---|---:|---:|---:|
| strategy | DefaultStrategy | DefaultStrategy | DefaultStrategy |
| absgrad | false | true | true |
| grow_grad2d | 0.0002 | 0.0006 | 0.0006 |
| responsibility | false | false | true |
| data_factor / test_every / SH / DSSIM | 相同 | 相同 | 相同 |
| seed | 42 | 42 | 42 |

A1 暂时以 B1 为直接基底，使 `B1 vs A1` 只隔离责任损失。最终强基线仍需 Mip-NeRF 360 clean 比较后才能确认；当前不得声称 B1 已胜出。

## 启动器

`run_puri_gs.py` 完成以下检查：

- gsplat commit 必须精确为 v1.5.3 固定 commit；
- 跟踪补丁必须已经应用且可反向检查；
- 数据必须存在 `images_<factor>` 与 `sparse/0` 三个 COLMAP 文件；
- 动态协议必须同时提供非空 train/test keyword；
- 输出目录非空时停止，避免覆盖实验；
- 写入 `config.yaml`、`run_command.txt`、`git_commit.txt`、`environment.json`；
- trainer 写入包含实际文件名的 `dataset_split.json`。

## 已完成检查

- 配置读取与字段约束：PASS。
- B0/B1/A1 Linux dry-run：PASS。
- Fern 与 Android 路径/文件检查：PASS。
- Windows/WSL Python 语法：PASS。
- gsplat 补丁反向应用检查：PASS。
- 既有 gsplat 协议和 reproduction gate 回归测试：PASS。
- 仓库全量测试：`72 passed in 24.51s`。
- 已创建隔离的 `.venv-gsplat153`，固定版本导入和 CPU checkpoint 往返：PASS。
- 实际 gsplat CUDA 最小前向：FAIL，错误为 `no kernel image is available for execution on the device`。

## 本地 CUDA smoke 阻断

本机 RTX 5070 Ti Laptop GPU 的 compute capability 为 `sm_120`。固定的
PyTorch 2.4.0+cu121 只包含至 `sm_90` 的内核；现有 gsplat wheel 经
`cuobjdump` 检查只包含 `sm_70/75/80/86/90` cubin，且不包含 PTX。

真实调用已经进入 gsplat 的 `projection_ewa_3dgs_packed_fwd`，随后因没有
可执行 kernel image 停止。因此没有重复启动 B0/B1/A1 三次必然失败的训练，
也没有把纯 PyTorch loss smoke 记为真实 rasterization smoke。

要完成本门禁需二选一并由用户决定：

1. 在支持该固定 wheel 的 GPU（例如已有固定服务器环境对应的 `sm_90` GPU）上运行；
2. 用户另行授权更换本地 PyTorch/CUDA 并针对 `sm_120` 重编译核心 rasterizer。

第二条会改变核心环境和二进制，已有本地 CUDA smoke 必须全部重跑；本轮未执行。

## Phase 1 状态

配置与启动检查完成。虽然本机 GPU 与固定二进制架构不兼容，但随后已在保持
PyTorch 2.4.0+cu121 和 gsplat 1.5.3+pt24cu121 不变的 L20 服务器上完成
B0/B1 的真实 rasterization、反向、checkpoint 保存和独立加载评测。因此
“B0/B1 baseline smoke 完成”的 Git 里程碑已经达到。

用户随后明确授权使用已有 L20 服务器，并允许为代码同步执行必要 Git 操作。
因此 `scripts/run_puri_gs_cuda_smoke.sh` 随 `dev` 提交：它要求
服务器工作区干净、固定环境可用、物理 GPU 为空闲 NVIDIA L20，并依次执行 B0/B1/A1
各 10 步、三份 checkpoint 独立评测、有限值检查和 A1 responsibility map 检查。
服务器最终已生成 `decision=PASS`，完成本里程碑。

## L20 服务器首次 smoke 诊断

首次服务器运行已经确认：

- 物理 GPU 6 是空闲 NVIDIA L20；
- 固定环境版本正确；
- 已安装 wheel 的真实 rasterization 前向/反向：PASS；
- wheel 路径的 checkpoint 保存/加载：PASS；
- Fern 解析为 20 张图，split 为 17 train / 3 test；
- B0 正确初始化 10091 个 Gaussian。

B0 在 step 0 的 rasterization 前失败。原因不是 CUDA、数据或 wheel，而是启动器把
`external/gsplat-v1.5.3` 加入 `PYTHONPATH`，使 `simple_trainer.py` 的
`import gsplat` 遮蔽已安装 wheel，错误进入源码 checkout 的 JIT fallback；该 checkout
没有 GLM 子模块，因而报 `glm/gtc/type_ptr.hpp` 缺失。

修复保持软件栈不变：外部 checkout 只提供打补丁后的 example trainer 和 dataset loader，
`gsplat` runtime 必须从固定 wheel 导入。启动器不再继承任意外部 `PYTHONPATH`，服务器
脚本还会记录实际 `gsplat.__file__`，一旦指向 external checkout 就在训练前停止。
没有下载 GLM、初始化子模块或重编译核心 rasterizer。

## L20 服务器最终 smoke 结果

修复随 `dev` commit `4f6fb3386caf70ebca66bf5cb3a0d8e4b7003a53`
推送后，使用相同 Fern 数据、GPU 6 和新结果目录重新运行。训练前检查确认：

- GPU：NVIDIA L20，compute capability `sm_89`；
- PyTorch：`2.4.0+cu121`，torch CUDA：`12.1`；
- gsplat：`1.5.3+pt24cu121`；
- `gsplat.__file__` 指向 `.venv-gsplat153/lib/python3.10/site-packages`，未被源码树遮蔽；
- 真实 rasterization 前向/反向：PASS；
- checkpoint roundtrip：PASS；
- Fern split：17 train / 3 test；
- 三种配置均训练 10 step，均保存并独立加载 `ckpt_9_rank0.pt`；
- 所有检查最终判定：`decision=PASS`。

| 配置 | PSNR | SSIM | LPIPS | GS 数 | checkpoint bytes | 训练显存 GiB | 10-step trainer time s |
|---|---:|---:|---:|---:|---:|---:|---:|
| B0 | 11.232206 | 0.426360 | 0.936244 | 10091 | 2384058 | 0.131077 | 0.744124 |
| B1 | 11.232204 | 0.426360 | 0.936245 | 10091 | 2384058 | 0.131077 | 0.715672 |
| A1 | 11.218639 | 0.426121 | 0.936116 | 10091 | 2384058 | 0.134024 | 0.815875 |

这些数值仅用于证明三条路径在同一真实 CUDA 栈中可执行、可反向、可保存并可评测。
10 step 尚未到正常增密和收敛阶段，不用于选择 B0/B1，也不构成 A1 效果结论。
