# PURI-NeRF V8 仓库审计

审计日期：2026-08-20  
审计提交：`e02df19`（本地与 `origin/dev` 一致）  
审计范围：V5、V6、V7、现有数据接口、训练日志和本地归档结果。

## 1. 框架与入口

项目是纯 PyTorch、自定义 `configargparse` 配置的 NeRF 实现，不依赖
Nerfstudio。三个历史入口分别为：

- V5：`run_nerf.py`、`v5/`、`configs/llff_colmap_v5.txt`；
- V6：`run_nerf_v6.py`、`v6/`、`configs/llff_colmap_v6.txt`；
- V7：`run_nerf_v7.py`、`v7/`、`configs/llff_colmap_v7.txt`。

V8 第一批入口为 `run_nerf_v8.py`。它目前只允许 `baseline` 和
`v7_5`，还没有导入 SDF、反射或介质模块，确保验收失败时不会越过停止条件。

## 2. V5、V6、V7 的实际差异

### V5

- 纯 PyTorch HashGrid、小型共享 trunk、RGB/density/uncertainty 三个头；
- uncertainty 通过 `m=1/(1+4u^2)` 直接缩放 density，随后改变 alpha、
  transmittance、渲染权重、RGB、深度和重要性采样；
- COLMAP rendered-depth 约束又乘以 `exp(-2.3u)`；
- uncertainty 教师、幅度正则和空间正则均可更新共享 trunk；
- 每张训练图像有独立背景参数。

因此 V5 存在明确的 reconstruction-avoidance 通路。

### V6

- density 使用 Softplus，不再由 uncertainty 缩放，体渲染权重与 UQ 数值独立；
- 新增 appearance embedding、distortion loss、评测和点云导出；
- geometry 中的 uncertainty 已 `detach()`，但其前向数值仍乘在几何残差上，
  因而仍会缩放 reconstruction gradient；
- heteroscedastic RGB NLL 中 predicted RGB 没有 detach，会更新 RGB 和共享 trunk；
- uncertainty head 直接读取共享 hidden，没有 detach。

V6 已消除“UQ 改密度”，但没有完全隔离 UQ loss 与 reconstruction。

### V7

- uncertainty head 输入 `hidden.detach()`；
- UQ 聚合使用 detached termination weights；
- RGB NLL 使用 `predicted_rgb.detach()`；
- geometry 完全不读取 uncertainty；
- 关闭 appearance embedding 和 spatial UQ loss；
- 加入可见性教师、termination-distribution 几何监督、mip-aware HashGrid、
  edge/UQ 自适应采样和可选 occupancy grid。

V7 的 RGB、density、alpha 和 geometry 已不存在 uncertainty 数值门控。当前仍存在的
间接通路是：25% rays 根据历史 uncertainty map 采样，这会改变 reconstruction 看到的
训练分布。因此，不能把 V7 的下降直接归因于“前向 density 门控”。

## 3. uncertainty 参与位置

| 位置 | V5 | V6 | V7 |
|---|---|---|---|
| density / alpha / transmittance | 直接参与 | 不参与 | 不参与 |
| RGB MSE | 通过渲染权重间接参与 | 不参与 | 不参与 |
| RGB NLL 对 reconstruction 梯度 | 共享 trunk | 存在 | 已隔离 |
| geometry | 可学习降权且可回传 | detached 数值降权 | 不参与 |
| sampling | 改变 coarse importance weights | 不参与 | 参与 ray batch 分布 |
| background / exposure | V5 每图背景 | 无 UQ 门控 | 无 UQ 门控 |
| UQ 正则 | 幅度、空间正则 | NLL、教师、空间正则 | 旁路 NLL、教师 |

## 4. 共享主干与 detach 审计

- V5：RGB、density 和 uncertainty 共享 trunk，UQ 梯度可进入 reconstruction；
- V6：仍共享 trunk，geometry 虽 detach UQ，但前向数值仍改变几何梯度大小；
- V7：模型结构仍读取共享 hidden，但 uncertainty head 的输入已 detach；UQ loss
  不能更新 HashGrid、trunk、density、feature 或 color；
- V7 uncertainty sampling 使用 detached map，但会改变后续 batch 分布。这是数值/数据
  路径，不是 autograd 路径。

V7.5 因此必须同时做到：独立 UQ optimizer、固定采样、`m=1`、UQ evidence detach，
而不能只再增加一个 `detach()`。

## 5. 教师不确定性与输出偏置

V5 教师为：

`u* = sigmoid(kappa * fused)`，其中 `kappa=2` 且 `fused>=0`。

所以 V5 教师严格满足 `u* >= 0.5`，最大约为 `sigmoid(2)=0.881`。这会把学生
长期推向至少中等 uncertainty，是一个明确的标度设计风险。

- V5 uncertainty head：未显式初始化输出偏置，使用 PyTorch Linear 默认初始化；
- V6 uncertainty head：输出偏置固定为 `-2`；
- V7 uncertainty head：输出偏置固定为 `-2`。

## 6. 已有指标

本地没有 V5 的测试指标 JSON，只保存了 200k path render 视频和逐帧结果，不能从视频
可靠反推出 PSNR/SSIM/LPIPS。

| 版本 | 训练设置 | 最佳 val PSNR | 最终 val PSNR | test PSNR | SSIM | LPIPS |
|---|---:|---:|---:|---:|---:|---:|
| V5 | fern, factor=4, 200k | 未归档 | 未归档 | 未归档 | 未归档 | 未归档 |
| V6 | fern, factor=2, 100k | 22.0916 | 22.0901 | 22.8417 | 0.7216 | 0.3415 |
| V7 | fern, factor=2, 100k | 22.1611 | 21.5383 | 21.9873 | 0.6487 | 0.3839 |

V7 在 25k 后持续回落，必须用 baseline/V7.5 判断是 reconstruction 设置、采样分布、
相机/场景尺度还是 UQ 机制导致。历史 V5 使用 factor=4 和 200k，不能与 V6/V7 直接做
严格数值比较。

## 7. 相机、坐标、边界与颜色

- 输入是 LLFF `poses_bounds.npy` + 同一次 COLMAP `sparse/0`；
- 相机支持 PINHOLE、SIMPLE_PINHOLE、SIMPLE_RADIAL；
- LLFF 的 recenter 和 scene scale 通过同一个 `world_transform` 作用于 COLMAP 点；
- V5 AABB 由合格 COLMAP 点 1%/99% 分位扩张 10%；V6/V7 再以中心扩张 1.25 倍；
- near=`0.9*min(bounds)`，far=`max(bounds)`；V6/V7 默认 disparity sampling；
- V6/V7 fern 默认 factor=2、test 每 8 张一张，原 train 再每 8 张划一张 val；
- 图像直接归一化到 `[0,1]`，未发现 sRGB 到线性 RGB 转换；
- 背景是训练图像均值初始化的单一可学习 RGB，不再是 V5 的 per-image background。

## 8. baseline / V7.5 的可比性约束

新增两种模式只允许以下差异：

- baseline：没有可学习 uncertainty head；
- V7.5：额外 side UQ head 和独立 UQ optimizer。

两者使用相同重建参数初始化顺序、固定 prior/edge/uniform sampler、相同 RGB/geometry/
distortion loss、相同 reconstruction optimizer、相同相机与数据 split。V7.5 的 UQ
输出不进入 density、RGB、geometry、sampling、background 或曝光。可选 UQ head 的初始化
和 Teacher ray 选择都在保存/恢复 RNG 的隔离区中执行，不能改变后续 reconstruction
抽到的射线序列。

## 9. 进入 V8-Core 前的阻塞项

1. 服务器上的 V5/V6/V7 最优 checkpoint 还没有完成独立归档；
2. V5 缺少机器可读的 train/val/test 指标；
3. baseline 尚未在 L6 训练，历史正常 PSNR 尚未确认；
4. V7.5 尚未完成与 baseline 的 `<=0.3 dB` 验收；
5. 参与介质阶段需要真实雾数据或明确标注为 synthetic 的替代数据。

以上任一 gate 未通过时，不进入 SDF/反射/介质实现。
