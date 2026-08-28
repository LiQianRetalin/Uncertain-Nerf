# PURI-GS 本地数据清单

- 扫描根目录：`E:\7-DataSet\nerf数据集`
- 生成时间（UTC）：`2026-08-28T10:02:12.105912+00:00`
- 扫描方式：只读；未移动、修改、下载、解压或计算大文件哈希。

## 数据集状态

| 数据集 | 状态 | 说明 |
|---|---|---|
| `mipnerf360` | **PARTIAL** | download/extraction is incomplete; no clean conclusion is allowed |
| `nerf_example_data` | **READY** | one or more COLMAP scenes were detected |
| `nerf_llff_data` | **READY** | one or more COLMAP scenes were detected |
| `nerf_on-the-go` | **UNSUPPORTED_FORMAT** | a minimal transforms-to-COLMAP-compatible adapter is required |
| `nerf_raw` | **PARTIAL** | one or more COLMAP scenes were detected |
| `nerf_ref` | **UNSUPPORTED_FORMAT** | no current-loader-compatible COLMAP scene was detected |
| `nerf_ref_real` | **READY** | one or more COLMAP scenes were detected |
| `nerf_robustnerf` | **PARTIAL** | COLMAP scenes are present; use filename keywords, not every-eighth splits |
| `nerf_synthetic` | **PARTIAL** | a minimal transforms-to-COLMAP-compatible adapter is required |
| `SeathruNeRF_dataset` | **PARTIAL** | one or more COLMAP scenes were detected |
| `_SeathruNeRF_download_temporary_files` | **UNSUPPORTED_FORMAT** | no current-loader-compatible COLMAP scene was detected |

## Mip-NeRF 360

- status: **PARTIAL**
- missing_items: `["bicycle", "bonsai", "counter", "garden", "kitchen", "room", "stump"]`
- reason: download/extraction is incomplete; no clean conclusion is allowed
- 在状态变为 READY 前，不进行 B0/B1 clean 正式比较、稀疏视角实验或最终 clean 结论。

## RobustNeRF 场景

| 场景 | 状态 | images | images_4 | 注册相机 | clutter(train) | extra(test) |
|---|---:|---:|---:|---:|---:|---:|
| `android` | READY | 263 | 263 | 263 | 122 | 19 |
| `crab1` | READY | 150 | 143 | 143 | 72 | 71 |
| `crab2` | READY | 412 | 412 | 412 | 109 | 194 |
| `statue` | PARTIAL | 416 | 416 | 415 | 255 | 19 |
| `yoda` | READY | 420 | 420 | 420 | 109 | 202 |

## 建议

- 代码 smoke 场景：`E:\7-DataSet\nerf数据集\nerf_example_data\nerf_llff_data\fern`
- 第一个动态场景：`android`（格式完整且协议最简单）。
- RobustNeRF 固定划分：文件名含 `clutter` 的图像训练，含 `extra` 的干净图像测试；不采用每 8 张抽一张。
- clean 候选：Mip-NeRF 360 下载完整后使用 `garden` 与 `room`；当前不生效。
- NeRF On-the-go 当前为 transforms JSON，需后续单独批准的最小格式适配；本阶段不转换。
