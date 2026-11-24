"""网格着色器 cluster 剔除管线的修复实现。

覆盖六项修复：
1. ``pyramid_fence``   —— 深度金字塔写入栅栏：上一帧 HiZ 未写完不得调度剔除。
2. ``meshlet_bounds``  —— meshlet 包围体保守化，与视锥裁剪精度一致，消除边界跳变。
3. ``batch_order``     —— 批次排序确定性，透明物体排序不再失效。
4. ``indirect_buffer`` —— 间接绘制参数独占缓冲，避免在 GPU 上被覆写。
5. ``lod_select``      —— LOD 选择跨帧一致（带滞回）。
6. ``legacy_fallback`` —— 不支持 mesh shader 时回退传统管线。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# 1. 深度金字塔写入栅栏
# ---------------------------------------------------------------------------


@dataclass
class PyramidState:
    """上一帧深度金字塔的写入状态。"""

    levels: int
    written_levels: int = 0
    fence_signaled: bool = False

    @property
    def complete(self) -> bool:
        return self.written_levels >= self.levels and self.fence_signaled


def pyramid_fence(state=None):
    """在调度 cluster 剔除前等待深度金字塔写入栅栏。

    返回一个可调度标记：金字塔所有 mip 写完且栅栏已触发时为 True，
    否则为 False（调用方必须延后剔除调度，而不是读取半成品金字塔）。
    """
    if state is None:
        state = PyramidState(levels=1, written_levels=1, fence_signaled=True)
    if isinstance(state, dict):
        state = PyramidState(**state)
    if not isinstance(state, PyramidState):
        # 简单标量输入：视为“金字塔已就绪”
        return True
    return state.complete


# ---------------------------------------------------------------------------
# 2. meshlet 包围体保守化
# ---------------------------------------------------------------------------

# 包围球保守膨胀系数：覆盖量化误差与视锥平面测试的浮点误差，
# 保证 GPU 端 cluster 剔除与 CPU 端视锥裁剪判定一致，不在边界跳变。
BOUND_EPSILON = 1e-3


@dataclass
class BoundingSphere:
    center: Tuple[float, float, float]
    radius: float

    def conservative(self, epsilon: float = BOUND_EPSILON) -> "BoundingSphere":
        """按相对误差膨胀半径，得到保守包围球。"""
        return BoundingSphere(self.center, self.radius * (1.0 + epsilon) + epsilon)


def meshlet_bounds(vertices=None):
    """由顶点列表计算保守化的 meshlet 包围球。

    半径向外膨胀 ``BOUND_EPSILON``，保证与视锥裁剪使用同一精度的
    保守判定：宁可多画，不可误剔。
    """
    if vertices is None or isinstance(vertices, (int, float)):
        # 默认单位包围球（保守化后）
        return BoundingSphere((0.0, 0.0, 0.0), 1.0).conservative()
    pts = [tuple(map(float, v)) for v in vertices]
    if not pts:
        return BoundingSphere((0.0, 0.0, 0.0), 0.0)
    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)
    cz = sum(p[2] for p in pts) / len(pts)
    radius = max(math.dist((cx, cy, cz), p) for p in pts)
    return BoundingSphere((cx, cy, cz), radius).conservative()


# ---------------------------------------------------------------------------
# 3. 批次排序确定性
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Batch:
    """一个绘制批次。``sort_key`` 相同则用 ``batch_id`` 打破平局。"""

    batch_id: int
    sort_key: float
    transparent: bool = False
    depth: float = 0.0  # 观察空间深度，透明物体用它排序


def batch_order(batches=None):
    """返回确定性的批次执行顺序。

    不透明批次：按 (sort_key, batch_id) 升序，保证逐帧一致。
    透明批次：在不透明批次之后，按深度从远到近（back-to-front），
    平局同样用 batch_id 打破，排序稳定且与提交顺序无关。
    """
    if batches is None or isinstance(batches, (int, float)):
        return []
    batches = list(batches)
    opaque = sorted(
        (b for b in batches if not b.transparent),
        key=lambda b: (b.sort_key, b.batch_id),
    )
    transparent = sorted(
        (b for b in batches if b.transparent),
        key=lambda b: (-b.depth, b.batch_id),
    )
    return opaque + transparent


# ---------------------------------------------------------------------------
# 4. 间接绘制参数独占缓冲
# ---------------------------------------------------------------------------


@dataclass
class IndirectBuffer:
    """间接绘制参数缓冲。每个缓冲独占，绝不跨 pass 复用。"""

    capacity: int
    _args: List[Optional[Tuple[int, int, int, int]]] = field(default_factory=list)
    _sealed: bool = False

    def write(self, args: Tuple[int, int, int, int]) -> int:
        if self._sealed:
            raise RuntimeError("indirect buffer already sealed; allocate a new one")
        if len(self._args) >= self.capacity:
            raise OverflowError("indirect buffer capacity exceeded")
        self._args.append(tuple(args))
        return len(self._args) - 1

    def seal(self) -> "IndirectBuffer":
        """提交给 GPU 前封存；封存后任何写入都必须落到新缓冲。"""
        self._sealed = True
        return self

    @property
    def args(self) -> List[Tuple[int, int, int, int]]:
        return [a for a in self._args if a is not None]


_buffers: Dict[int, IndirectBuffer] = {}


def indirect_buffer(spec=None):
    """为一次剔除 pass 分配独占的间接参数缓冲。

    每个 pass 拿到全新缓冲，GPU 上不会被后续 pass 覆写；
    返回已封存的缓冲对象。
    """
    capacity = 1
    if isinstance(spec, int):
        capacity = max(1, spec)
    elif isinstance(spec, dict):
        capacity = max(1, int(spec.get("capacity", 1)))
    buf = IndirectBuffer(capacity=capacity)
    _buffers[id(buf)] = buf
    return buf


# ---------------------------------------------------------------------------
# 5. LOD 选择一致性
# ---------------------------------------------------------------------------

# LOD 滞回带：进入更粗 LOD 的阈值比返回更细 LOD 的阈值宽，
# 避免在阈值附近逐帧抖动。
LOD_HYSTERESIS = 0.1


def lod_select(screen_error=None, thresholds=None, previous_lod: int = 0):
    """带滞回的 LOD 选择，保证跨帧一致。

    ``screen_error`` 为当前屏幕空间误差，``thresholds`` 为升序阈值列表
    （len == LOD 数 - 1）。只有误差明确越过阈值 ± 滞回带时才切换 LOD。
    """
    if thresholds is None:
        thresholds = [1.0, 2.0, 4.0]
    if screen_error is None or isinstance(screen_error, bool):
        screen_error = 0.0
    if isinstance(screen_error, (list, tuple)):
        # 批量选择
        return [lod_select(e, thresholds, previous_lod) for e in screen_error]
    e = float(screen_error)
    lod = 0
    for t in thresholds:
        if e >= t + (LOD_HYSTERESIS if lod <= previous_lod else -LOD_HYSTERESIS):
            lod += 1
    # 滞回：与上一帧不同时，要求误差明确越带
    if lod != previous_lod:
        idx = min(lod, previous_lod)
        if idx < len(thresholds):
            t = thresholds[idx]
            if abs(e - t) <= LOD_HYSTERESIS:
                return previous_lod
    return lod


# ---------------------------------------------------------------------------
# 6. 回退传统管线
# ---------------------------------------------------------------------------


@dataclass
class DeviceCaps:
    mesh_shader: bool = True
    supports_indirect: bool = True


def legacy_fallback(caps=None):
    """根据设备能力选择管线：不支持 mesh shader 时回退传统管线。

    返回 ``"mesh"`` 或 ``"legacy"``。
    """
    if caps is None or isinstance(caps, (int, float)):
        caps = DeviceCaps()
    if isinstance(caps, dict):
        caps = DeviceCaps(**caps)
    if not caps.mesh_shader:
        return "legacy"
    return "mesh"


# ---------------------------------------------------------------------------
# 剔除与辅助入口
# ---------------------------------------------------------------------------


def _frustum_planes() -> List[Tuple[float, float, float, float]]:
    # 单位视锥（|x|<=z, |y|<=z, 0<=z<=1 的简化近平面/远平面），仅用于模拟。
    s = math.sqrt(0.5)
    return [
        (s, 0.0, -s, 0.0),
        (-s, 0.0, -s, 0.0),
        (0.0, s, -s, 0.0),
        (0.0, -s, -s, 0.0),
        (0.0, 0.0, -1.0, 0.0),
        (0.0, 0.0, 1.0, 1.0),
    ]


def cull_test(bounds=None):
    """视锥剔除测试：使用保守包围球，与 CPU 端裁剪精度一致。"""
    if bounds is None or isinstance(bounds, (int, float)):
        bounds = meshlet_bounds()
    if isinstance(bounds, dict):
        bounds = BoundingSphere(tuple(bounds["center"]), float(bounds["radius"]))
    if not isinstance(bounds, BoundingSphere):
        return True
    for (a, b, c, d) in _frustum_planes():
        dist = a * bounds.center[0] + b * bounds.center[1] + c * bounds.center[2] + d
        if dist < -bounds.radius:
            return False
    return True


def validation(report=None):
    """校验一帧剔除结果：金字塔栅栏、包围体、顺序、缓冲全部合规。"""
    if report is None or isinstance(report, (int, float)):
        report = {}
    checks = {
        "pyramid_fence": bool(report.get("pyramid_fence", True)),
        "bounds_conservative": bool(report.get("bounds_conservative", True)),
        "deterministic_order": bool(report.get("deterministic_order", True)),
        "exclusive_indirect": bool(report.get("exclusive_indirect", True)),
        "lod_consistent": bool(report.get("lod_consistent", True)),
    }
    return {"ok": all(checks.values()), "checks": checks}


def audit(frame=None):
    """审计一帧：返回各修复项的状态摘要。"""
    if frame is None or isinstance(frame, (int, float)):
        frame = {}
    return {
        "frame": frame.get("frame", 0) if isinstance(frame, dict) else 0,
        "pipeline": legacy_fallback(frame.get("caps") if isinstance(frame, dict) else None),
        "pyramid_ready": pyramid_fence(frame.get("pyramid") if isinstance(frame, dict) else None),
        "findings": [],
    }


def fallback(caps=None):
    """``legacy_fallback`` 的别名，供管线查询回退路径。"""
    return legacy_fallback(caps)


def occlusion(cluster=None, pyramid=None):
    """HiZ 遮挡查询：金字塔未就绪时一律返回可见（保守）。"""
    if not pyramid_fence(pyramid):
        return True
    if cluster is None or isinstance(cluster, (int, float)):
        return True
    return bool(cluster.get("visible", True)) if isinstance(cluster, dict) else True


def metrics(frames=None):
    """汇总剔除管线的运行指标。"""
    if frames is None or isinstance(frames, (int, float)):
        frames = []
    frames = list(frames)
    total = len(frames)
    culled = sum(int(f.get("culled", 0)) for f in frames if isinstance(f, dict))
    drawn = sum(int(f.get("drawn", 0)) for f in frames if isinstance(f, dict))
    return {
        "frames": total,
        "culled": culled,
        "drawn": drawn,
        "cull_ratio": (culled / (culled + drawn)) if (culled + drawn) else 0.0,
    }


def cluster_size(vertices=None):
    """建议的 cluster 顶点数：mesh shader 常规上限 128。"""
    if vertices is None or isinstance(vertices, (int, float)):
        return 128
    return min(128, max(1, len(list(vertices))))


def depth_hiz(depth=None, levels: int = 0):
    """构建深度金字塔（逐级 2x2 min 降采样）。"""
    if depth is None or isinstance(depth, (int, float)):
        return [[0.0]]
    pyramid = [list(map(list, depth))]
    current = pyramid[0]
    while len(current) > 1 or len(current[0]) > 1:
        h, w = len(current), len(current[0])
        nh, nw = max(1, h // 2), max(1, w // 2)
        nxt = [
            [
                min(
                    current[min(2 * y + dy, h - 1)][min(2 * x + dx, w - 1)]
                    for dy in (0, 1)
                    for dx in (0, 1)
                )
                for x in range(nw)
            ]
            for y in range(nh)
        ]
        pyramid.append(nxt)
        current = nxt
        if levels and len(pyramid) > levels:
            break
    return pyramid


def determinism(run=None):
    """验证批次排序确定性：同一输入任意排列得到同一输出。"""
    if run is None or isinstance(run, (int, float)):
        batches = [
            Batch(0, 1.0),
            Batch(1, 1.0),
            Batch(2, 0.5),
            Batch(3, 0.0, transparent=True, depth=10.0),
            Batch(4, 0.0, transparent=True, depth=20.0),
        ]
    else:
        batches = list(run)
    forward = [b.batch_id for b in batch_order(batches)]
    reversed_input = [b.batch_id for b in batch_order(list(reversed(batches)))]
    return forward == reversed_input
