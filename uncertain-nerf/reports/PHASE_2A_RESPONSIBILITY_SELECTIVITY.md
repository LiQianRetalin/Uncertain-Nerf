# PURI-GS Phase 2A：Responsibility 选择性分析

## 结论

```text
MODERATE_EDGE_SENSITIVITY
```

本结论不声称识别了真实动态区域。Android 没有提供官方动态 mask，本分析只判断
低责任区域与普通静态边缘、高频纹理之间的关联。

## 执行约束

- checkpoint：`android_short10k_seed42_32a7d78/a1/ckpts/ckpt_9999_rank0.pt`；
- checkpoint step：9999；
- Gaussian 参数全部冻结；
- backward 调用数：0；
- 每帧 rasterization 次数：1；
- 训练集按文件名排序后，从 122 张图确定性抽取索引
  `[0, 17, 35, 52, 69, 86, 104, 121]`；
- 使用固定 A1 参数：start step 3000、threshold 1.5、min weight 0.2、
  pool size 3；
- 运行时为 `.venv-gsplat153` 中的固定 gsplat 1.5.3 wheel。

八张图为：`2clutter019.JPG`、`2clutter036.JPG`、`2clutter054.JPG`、
`2clutter071.JPG`、`2clutter088.JPG`、`2clutter105.JPG`、
`2clutter123.JPG`、`2clutter140.JPG`。

## 数值统计

| 统计量 | 均值 | 中位数 | 最小值 | 最大值 |
|---|---:|---:|---:|---:|
| Spearman(1-q, gradient) | 0.352883 | 0.395710 | 0.120060 | 0.579101 |
| Spearman(1-q, residual) | 0.732958 | 0.723660 | 0.697341 | 0.792035 |
| q<0.8 像素比例 | 19.42% | 17.60% | 15.98% | 23.70% |
| q<0.5 像素比例 | 10.90% | 9.88% | 6.06% | 19.54% |
| q<0.8 与最高 10% 梯度重合率 | 22.32% | 22.56% | 14.23% | 27.34% |
| q<0.5 与最高 10% 梯度重合率 | 18.09% | 17.87% | 10.56% | 24.38% |
| responsibility q 均值 | 0.880708 | 0.887273 | 0.830819 | 0.910110 |

## 人工复核

八张固定范围七联图显示，低责任响应主要覆盖高残差物体、遮挡和重建不一致区域，
但百叶窗、桌布格纹、书盒文字、家具与物体轮廓等普通静态高频结构也反复出现响应。
其中 `2clutter105.JPG` 的梯度相关达到 0.579101，多个视图中可见静态轮廓与
低 q 区域重叠。

另一方面，低 q 区域与最高 10% 梯度区域的平均重合率仅为 18%--22%，没有在多数
像素或多数视图中被普通静态边缘完全主导，因此证据不足以判为
`HIGH_EDGE_SENSITIVITY`。由于静态高频响应并非偶发，也不宜判为
`LOW_EDGE_SENSITIVITY`，最终分类为 `MODERATE_EDGE_SENSITIVITY`。

## 证据

完整证据包 SHA-256：

```text
575557E482A102B9C1FABB1687A8963F3C94E02C3F84BF5EC37AB4CBA7C94419
```

