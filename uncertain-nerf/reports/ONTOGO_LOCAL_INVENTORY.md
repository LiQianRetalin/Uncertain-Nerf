# NeRF On-the-go 本地数据盘点

盘点状态：`ONTOGO_LOCAL_INVENTORY_COMPLETE`

当前停止状态：`PATIO_HIGH_LOADER_BLOCKED`

## 本地根目录

```text
E:\7-DataSet\nerf数据集\nerf_on-the-go\on-the-go
```

本地共有 12 个场景：`arcdetriomphe`、`corner`、`drone`、`fountain`、`mountain`、`patio`、`patio_high`、`spot`、`statue`、`train`、`train_station`、`tree`。

## 场景清单

| 场景 | 图像 | transforms frames | clutter | extra | COLMAP | 当前 loader 直读 |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| arcdetriomphe | 226 | 226 | 196 | 29 | 无 | 否 |
| corner | 122 | 122 | 101 | 20 | 无 | 否 |
| drone | 237 | 237 | 205 | 31 | 无 | 否 |
| fountain | 185 | 185 | 168 | 16 | 无 | 否 |
| mountain | 131 | 131 | 119 | 11 | 无 | 否 |
| patio | 125 | 125 | 98 | 26 | 无 | 否 |
| patio_high | 267 | 267 | 221 | 45 | 无 | 否 |
| spot | 179 | 179 | 168 | 10 | 无 | 否 |
| statue | 170 | 170 | 152 | 17 | 无 | 否 |
| train | 180 | 179 | 159 | 19 | 无 | 否 |
| train_station | 207 | 207 | 180 | 26 | 无 | 否 |
| tree | 212 | 212 | 175 | 36 | 无 | 否 |

`train` 目录额外存在一个未被 `transforms.json` 引用的 `IMG_7332.JPG`；本阶段固定 Patio-High，因此不为该极少数情况增加 fallback。

## Patio-High 结论

`patio_high` 确实存在：267 张图、267 条 pose，所有 pose 引用的图像均存在；官方 `split.json` 包含 221 个 clutter index 与 45 个 extra index，无重叠，另有 frame index 266 未分配给这两组。

但是当前冻结入口 `run_puri_gs.py:_verify_dataset` 和 gsplat COLMAP parser 要求：

```text
images_<data_factor>
sparse/0/cameras.bin
sparse/0/images.bin
sparse/0/points3D.bin
```

Patio-High 目前只有：

```text
images
split.json
transforms.json
```

项目入口不读取 `transforms.json` 或 `split.json`，因此不能直接训练。按照本阶段“条件不满足即停止、不自动换场景、不自动换方案”的要求，本次没有选择其他 On-the-go 场景，没有运行 COLMAP，也没有实现新 loader。后续必须先由用户确认采用哪一种**单一**数据适配协议，并单独验证 pose/split 等价性，然后才可打包与上传 Patio-High。

