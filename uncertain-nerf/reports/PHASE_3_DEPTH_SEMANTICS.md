# PURI-GS Phase 3：CVTR 深度语义审计

## 结论

CVTR 固定使用 gsplat 1.5.3 的：

```text
render_mode="RGB+ED"
```

其最后一个通道是按累积 alpha 归一化的 expected camera-z depth：

```text
ED = sum_i(w_i * z_i) / sum_i(w_i)
```

其中 `z_i` 是 Gaussian 中心变换到当前相机坐标后的第三个分量，不是欧氏 ray
distance，也不是未归一化的 accumulated depth。CVTR 因而用
`K^{-1}[u,v,1]^T * z` 反投影，并在邻居相机中比较投影 camera-z 与邻居 ED。

## 代码证据

固定源码为 gsplat commit：

```text
937e29912570c372bed6747a5c9bf85fed877bae
```

源码审计结果：

1. `gsplat/rendering.py` 的 Depth Rendering 文档明确区分：
   - `D = sum_i(w_i * z_i)`；
   - `ED = sum_i(w_i * z_i) / sum_i(w_i)`；
2. `RGB+ED` 将投影阶段生成的 `depths` 追加为最后一个待光栅化通道；
3. 光栅化结束后，最后一通道除以 `render_alphas.clamp(min=1e-10)`；
4. `ProjectionEWA3DGSFused.cu` 和 `ProjectionEWA3DGSPacked.cu` 均写入
   `depths = mean_c.z`；
5. COLMAP loader 保存 `camtoworld`，trainer 调用 rasterizer 时传入
   `viewmats=torch.linalg.inv(camtoworlds)`，内参为同一视图的 `K`。

因此，当前相机内外参和深度的组合在代数上闭合：

```text
camera = inverse(camtoworld) * world
pixel  = K * camera / camera.z
world  = camtoworld * (inverse(K) * [u,v,1] * expected_camera_z)
```

## 单 Gaussian 数值测试

`tests/test_cvtr_geometry.py::test_single_gaussian_rgb_ed_is_expected_camera_z`
构造：

- 单个 Gaussian 中心 `(0, 0, 2.5)`；
- 单位 world-to-camera；
- pinhole `K`；
- `render_mode="RGB+ED"`；
- 对所有 alpha 大于 `1e-3` 的像素断言 ED 为 `2.5 ± 1e-4`。

本地 RTX 5070 Ti Laptop 为 `sm_120`，固定 PyTorch 2.4.0 + CUDA 12.1 +
gsplat 1.5.3 wheel 只包含到 `sm_90`，所以该 CUDA 数值测试在本机按设计跳过，
没有升级 PyTorch、CUDA、gsplat，也没有重编译 rasterizer。CPU 投影—反投影闭环、
深度相对一致性和保守无邻居测试均已通过。

在服务器开始任何合成 mask 验证前，必须先在原 L20 固定环境运行该测试。只有显示
`1 passed` 才能继续；失败或跳过均停止，且不得启动 continuation。

## 禁止的替代解释

CVTR 不接受以下替代：

- 将 `D`（accumulated depth）误当作 `ED`；
- 将 camera-z 误当作沿单位 ray 的距离；
- 使用归一化显示深度；
- alpha 不足或有效邻居不足时推测深度；
- 数值测试失败后切换深度模式或加入 fallback。
