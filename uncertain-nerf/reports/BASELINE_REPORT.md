# A0 Baseline 恢复报告

状态：**正式运行完成，但未通过基础模型验收；停止进入 V7.5 与 V8-Core。**

## 实验身份

- 实现：`puri-nerf-a0-original-1`
- 场景：LLFF fern
- 图像分辨率：factor=2（1512 x 2016）
- split：14 train / 3 val / 3 test
- 随机种子：0
- 训练步数：200000
- 网络：经典 coarse/fine NeRF，两个 8 x 256 MLP
- 采样：1024 rays，64 coarse + 64 fine samples
- 渲染：LLFF forward-facing NDC
- 优化：Adam，初始学习率 5e-4，原始指数衰减，FP32
- 训练设备：物理 GPU 6，NVIDIA L20
- uncertainty、COLMAP 几何损失、HashGrid、Mip、edge sampling：全部关闭

本实验是独立的经典 NeRF A0；不复用 V5--V8 的重建网络或损失。与历史版本的
指标比较仅用于诊断，不代表架构依赖。

## 训练结果

| 指标 | 结果 |
|---|---:|
| best val PSNR | 22.4673 dB @ 195000 |
| final val PSNR | 22.4558 dB @ 200000 |
| target PSNR | 25.0 dB（未达到） |
| time to target PSNR | null |
| 总训练时间 | 16161.01 s（4 h 29 min 21 s） |
| 纯优化时间 | 9885.55 s（2 h 44 min 46 s） |
| 验证时间 | 6275.46 s（1 h 44 min 35 s） |
| 峰值显存 | 3069.25 MiB |
| 参数量 | 1,192,202 |

下载的评测目录没有包含完整 `train_metrics.jsonl`，因此最终训练 batch 的
`PSNR_train` 和梯度范数不在本报告中臆测填写。正式训练前的全尺寸100步测试中，
density/color 梯度均为有限非零值，且全过程无 NaN、Inf 或 OOM。

## 测试集结果

| checkpoint | PSNR | SSIM | LPIPS | render compute | rays/s |
|---:|---:|---:|---:|---:|---:|
| 100000 | 22.4997 | 0.6694 | 0.4304 | 294.08 s | 31,095 |
| 190000 | 22.9505 | 0.6931 | 0.3858 | 294.51 s | 31,050 |
| 200000 | **22.9529** | **0.6942** | **0.3815** | 294.08 s | 31,095 |

200000步三张测试图的逐视角 PSNR 为 21.1757、23.7150 和 23.9681 dB。
190000到200000步仅增加0.0025 dB，说明模型已经进入稳定平台，而不是训练不足。

200000步测试渲染的平均 opacity 为0.9999966，平均 terminal transmittance 为
3.42e-6，平均 NDC depth 为0.5970。三张图均未出现破洞、NaN 或离散 floater；
主要误差是细叶边界、高光和遮挡交界的平滑、半透明重影，以及背景高频纹理被平均。

A0 没有不确定性输出。评测器对全零 uncertainty 强制生成的 NLL、AUSE、AURG、
AUROC 和 UCE 不具有 A0 模型含义，不纳入验收或版本比较。

## 与历史诊断结果的关系

- A0 200000步比旧 V7-trunk diagnostic baseline 100000步（21.9691 dB）高
  0.9838 dB，证明独立 A0 确实修复了一部分退化。
- A0 比该 diagnostic baseline 的早停20000步（22.6042 dB）高0.3487 dB。
- A0 比历史 V6（22.8417 dB）高0.1112 dB，但 LPIPS 0.3815 仍明显差于
  V6 的0.3415。
- A0 仍只有约22.95 dB，且未达到规格中的25 dB目标，因此不能被视为“历史正常
  基线已经恢复”。

## 验收决定

规格明确规定：基础模型同样只有约22 dB时立即停止 V8，实现前优先排查相机位姿、
场景尺度、采样范围、图像预处理、曝光和评测代码。当前结果触发该条件。

- 不进入 V7.5。
- 不进入 V8 SDF、结构化反射、介质或后验 UQ。
- 不通过修改目标值把22.95 dB事后定义为通过。
- 下一阶段只允许进行 baseline 协议审计和单变量复现实验。

完整停止依据见 `reports/STOP_BASELINE_FAILURE.md`。

## 2026-08-27 路线终止

CamP + Zip-NeRF 的最终 aligned PSNR/SSIM 为 17.4028/0.4254，确认相机优化路线
不能恢复基线。A0、V7.5、CamP 和 Zip-NeRF 均不再继续；下一实验入口切换到
`ROBOT_GAUSSIAN_GUIDE.md` 的显式 Gaussian 机器人短筛。
