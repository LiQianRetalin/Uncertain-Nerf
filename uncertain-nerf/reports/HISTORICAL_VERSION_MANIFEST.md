# V5–V7 冻结清单

本文件冻结“代码身份与已知实验资产”，不把缺失资产写成已完成。

## V5

- Git tag：`v5-fern-200k`
- commit：`c234eb2e178eda3ac6c6ec1d817e5296af361acb`
- 入口：`run_nerf.py`
- 模型：`v5/`
- 配置：`configs/llff_colmap_v5.txt`
- 配置 SHA-256：`D36A0F530A4A8AFA3CE46AB975C71E7EC256048C1928DC0C1272B21B7FC8DC69`
- 本地结果：`训练及渲染结果/mp4-V5渲染/`
- 最优 checkpoint：仅服务器待核对/归档
- 指标 JSON：缺失，待从服务器日志补齐

## V6

- Git tag：`v6.1-dev-fern-seed0-100k`
- commit：`ec0d0afd077515a977f8341f189bdfe2061d31c2`
- 入口：`run_nerf_v6.py`
- 模型：`v6/`
- 配置：`configs/llff_colmap_v6.txt`
- 配置 SHA-256：`8F18D68F8E8BDB8EF16987078511DC3E2D82704FC39104D715F107EBCE060B44`
- 本地日志/结果：`训练及渲染结果/v6-jsonl/`
- test：PSNR 22.8417、SSIM 0.7216、LPIPS 0.3415
- 最优 checkpoint：仅服务器待核对/归档

## V7

- 当前冻结 commit：`e02df19db1b3149c95e7b2507e49911313aeecbd`
- 建议 tag：`v7-fern-seed0-100k`
- 入口：`run_nerf_v7.py`
- 模型：`v7/`
- 配置：`configs/llff_colmap_v7.txt`
- 配置 SHA-256：`3F3E08E9A980ECDE28A756DB3F5AEB467C00D17E6CC40303788C7C57D2238548`
- 本地日志/结果：`训练及渲染结果/v7-jsonl/`
- test：PSNR 21.9873、SSIM 0.6487、LPIPS 0.3839
- 最优 checkpoint：仅服务器待核对/归档

## 冻结纪律

- 不修改 `v5/`、`v6/`、`v7/`、三个历史入口和三个历史配置；
- 不用 V8 checkpoint 覆盖历史 `logs*`；
- 不把数据集、checkpoint、视频或本地“训练及渲染结果”加入 Git；
- 服务器归档完成后记录 checkpoint 路径、文件大小和 SHA-256；
- V7 tag 由用户在图形化 Git 客户端创建并推送，不由代码脚本自动创建。
