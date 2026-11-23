# pairwise-meshlet-cluster-cull

## 领域

游戏、图形与多媒体

## 主题

网格着色器 cluster 剔除

## 说明

深度金字塔栅栏、保守包围体与间接参数独占。

初始状态下 `graphics/meshlet_cluster_cull.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。
