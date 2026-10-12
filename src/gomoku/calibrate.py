"""标定：从一张含棋盘的图里自动找出 15×15 个交叉点，不需要人工四点标定。

做法：木纹色掩膜 → 最大连通域 = 棋盘面 → 对面内亮度做列/行中位数投影 → 找低于基线的
谷（网格线）。实测这套两点标定在三种窗口位置/尺寸下都稳定，且格距 std 仅 0.59px，
所以采集是轴对齐的，不需要单应矩阵。

找不到棋盘时必须干净失败（抛 CalibrationError），绝不能吐出一副错的矩阵。

同一条纪律也管住「复用上次格点」这条路：`verify_grid` 是门卫，它对整格平移（图案自相似、
只有盘面边界会越出缓存框）必须拦得下来，拦不住就宁可每拍重标。判据的来历见 `_separation`。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .config import Config, GridConfig, WoodConfig


class CalibrationError(Exception):
    """画面里没有可标定的棋盘，或检出的网格不像棋盘。"""


@dataclass(frozen=True)
class Geometry:
    size: int
    xs: tuple[int, ...]          # 交叉点列坐标（像素）
    ys: tuple[int, ...]          # 交叉点行坐标（像素）
    bbox: tuple[int, int, int, int]   # 棋盘面 x0,y0,x1,y1（含端点，像素）

    @property
    def spacing(self) -> tuple[float, float]:
        return (
            float(np.mean(np.diff(self.xs))),
            float(np.mean(np.diff(self.ys))),
        )

    def point(self, x: int, y: int) -> tuple[int, int]:
        return self.xs[x], self.ys[y]


def luminance(rgb: np.ndarray) -> np.ndarray:
    r, g, b = (rgb[:, :, i].astype(np.float32) for i in range(3))
    return 0.299 * r + 0.587 * g + 0.114 * b


def wood_mask(rgb: np.ndarray, cfg: WoodConfig) -> np.ndarray:
    r, g, b = (rgb[:, :, i].astype(np.int16) for i in range(3))
    return (
        (r > cfg.min_r)
        & (g > cfg.min_g)
        & (b > cfg.min_b)
        & (b < cfg.max_b)
        & (r > b + cfg.min_rb_gap)
    )


def _projection_peaks(profile: np.ndarray, cfg: GridConfig) -> list[int]:
    """一维亮度投影上的网格线位置：明显低于中位基线的连续段，取段内均值。"""
    base = float(np.median(profile))
    idx = np.where(profile < base - cfg.dip_offset)[0]
    groups: list[list[int]] = []
    for i in idx:
        if groups and i - groups[-1][-1] <= cfg.merge_gap:
            groups[-1].append(int(i))
        else:
            groups.append([int(i)])
    return [int(np.mean(g)) for g in groups]


def _best_run(coords: list[int], size: int) -> tuple[list[int], float]:
    """候选峰里挑等距最均匀的一段长度 size。

    投影窗外扩过盘面边缘，就会把外框的暗带也读成一条「线」。暗带到最近真网格线的
    距离由外扩量兜住（远小于一格），所以它只可能污染段的两端 —— 全长等距这一段就是答案。
    """
    best: list[int] = coords[:size]
    best_cv = float("inf")
    for i in range(len(coords) - size + 1):
        run = coords[i : i + size]
        steps = np.diff(run)
        cv = float(steps.std() / steps.mean())
        if cv < best_cv:
            best, best_cv = run, cv
    return best, best_cv


def _check_axis(
    coords: list[int], axis: str, size: int, origin: int, cfg: GridConfig
) -> tuple[int, ...]:
    if len(coords) < size:
        raise CalibrationError(
            f"{axis}轴只检出 {len(coords)} 条网格线，期望 {size} 条（棋盘可能被遮挡或不在画面内）"
        )
    run, cv = _best_run(coords, size)
    if cv > cfg.max_spacing_cv:
        raise CalibrationError(f"{axis}轴网格线间距不均匀 (cv={cv:.3f})，检出的不是棋盘网格")
    return tuple(origin + c for c in run)


def _line_darkness(rgb: np.ndarray, centers: np.ndarray, span: tuple[int, int],
                   half: int) -> np.ndarray:
    """一批**列**坐标各自的暗度： centres 必须沿 x 分布，span 是行的取值范围。

    每条线上先在该中心点 ±half 的垂直窗口里取最暗（网格线只有 2px 宽，取中位会被 1px
    抖动打掉），再沿线取中位（沿线会穿过 15 个交叉点，黑白子在那儿压出局部极值，
    中位吃得住）。返回长度 = len(centers)。
    """
    lo, hi = span
    w = rgb.shape[1]
    idx = np.clip(np.asarray(centers), 0, w - 1)
    offs = np.arange(-half, half + 1)
    win = np.clip(idx[:, None] + offs[None, :], 0, w - 1)          # (n, 2*half+1)
    patch = rgb[lo:hi + 1][:, win, :].astype(np.float32)           # (band, n, 2*half+1, 3)
    lum = 0.299 * patch[..., 0] + 0.587 * patch[..., 1] + 0.114 * patch[..., 2]
    return np.median(lum.min(axis=2), axis=0)


def _separation(on: np.ndarray, off: np.ndarray) -> float:
    """一条轴的线谷分离度：谷（线间）最暗的那一处 − 线（交叉点）最亮的那一处。

    必须用极值，不能用中位或百分位：**整格平移**时网格图案自相似、逐线逐谷全部照旧分明，
    唯一的破绽在边界那一两处（越出棋盘面、落到窗框或桌面上）。实测拖窗一格那拍
    谷间暗度是 `[54, 207, 207, 210, ...]`：中位判据给 98、第 25 百分位给 98，都放行，
    而盘面已经错了 18 格。极值判据在这一拍给 −60，直接拦下。
    反过来说误拦的代价只是一次全量标定（实测 16ms），所以宁可从紧。
    """
    return float(np.min(off) - np.max(on))


def grid_contrast(rgb: np.ndarray, geom: Geometry, half: int = 4) -> tuple[float, float]:
    """两条轴各自的线谷分离度；格点还压在网格线上时应当明显为正。"""
    xs, ys = np.asarray(geom.xs), np.asarray(geom.ys)
    x_on = _line_darkness(rgb, xs, (ys[0], ys[-1]), half)
    x_off = _line_darkness(rgb, (xs[:-1] + xs[1:]) // 2, (ys[0], ys[-1]), half)
    swap = rgb.transpose(1, 0, 2)                                  # 行列互换，同一套取样代码
    y_on = _line_darkness(swap, ys, (xs[0], xs[-1]), half)
    y_off = _line_darkness(swap, (ys[:-1] + ys[1:]) // 2, (xs[0], xs[-1]), half)
    return _separation(x_on, x_off), _separation(y_on, y_off)


def verify_grid(rgb: np.ndarray, geom: Geometry, cfg: Config) -> bool:
    """缓存的格点还能不能用（能就可以跳过本拍的全量标定）。

    按主循环的真实形状量的：棋盘在假桌面上整体挪一笔、ROI 取样框不动，再用旧格点重读
    整盘。放行区间取决于这一面盘自己的对比度，三面 fixture 实测：空盘与中局盘 0–4px 放行
    （分离度 90→52，重读一格没错）、5px 掉到 8–9 拦下（这一档其实还没读错，多跑一次全量
    标定不亏）、6px 起 −5；压着黑色「＋」标记子的那面 2px 就掉到 30、3px 到 6.5 拦下。
    整格 ±54/108/162px 一律 −57 到 −145，全部拦下。
    阈值沿用标定自己的 dip_offset=12：静止盘面最紧的一面也给到 87，余量 7 倍。
    """
    h, w = rgb.shape[:2]
    if geom.xs[0] < 0 or geom.ys[0] < 0 or geom.xs[-1] >= w or geom.ys[-1] >= h:
        return False                       # 格点已跑到帧外，取样带会是空切片
    cx, cy = grid_contrast(rgb, geom)
    need = cfg.grid.dip_offset
    return cx >= need and cy >= need


def detect_board(rgb: np.ndarray, cfg: Config) -> Geometry:
    board, wood, grid = cfg.board, cfg.wood, cfg.grid
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise CalibrationError(f"输入必须是 HxWx3 的 RGB 图，收到 {rgb.shape}")

    lum = luminance(rgb)
    if float(lum.max()) <= 1.0:
        raise CalibrationError("整帧全黑：显示器睡了或正在锁屏，采集链没拿到画面（不是棋盘被挡）")
    mask = wood_mask(rgb, wood)
    if not mask.any():
        raise CalibrationError("画面里没有木纹色区域，棋盘可能未显示或被完全遮挡")

    labels, n = ndimage.label(mask)
    # 只数像素挑最大域：ndimage.sum 在 882×882 的掩膜上要 7.4ms，bincount 3.6ms，
    # 四面 fixture 上挑出的 bbox 逐帧一致。
    sizes = np.bincount(labels.ravel(), minlength=n + 1)
    biggest = int(np.argmax(sizes[1:])) + 1
    ys, xs = np.where(labels == biggest)
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())

    # 木纹掩膜在主连通域的最外圈网格线处就断了：那条线把盘面边缘的窄木纹带切成独立
    # 连通域，于是 bbox 比盘面短约一条线宽，第 15 条线落在窗外（实测短 4px / 12px）。
    # 投影窗按盘面尺寸外扩一点把它纳进来。外扩量远小于「最外圈线 → 外框暗带」的距离，
    # 就算外框暗带真被读成一条线，它与最近的真网格线也不差一格，_best_run 挑均匀段时
    # 只会把它留在段外，不会吐出错位的矩阵。
    pad = max(6, int(round(0.02 * max(x1 - x0, y1 - y0))))
    px0, px1 = max(0, x0 - pad), min(rgb.shape[1], x1 + pad + 1)
    py0, py1 = max(0, y0 - pad), min(rgb.shape[0], y1 + pad + 1)
    proj = lum[py0:py1, px0:px1]
    cols = _projection_peaks(np.median(proj, axis=0), grid)
    rows = _projection_peaks(np.median(proj, axis=1), grid)

    gxs = _check_axis(cols, "x", board.size, px0, grid)
    gys = _check_axis(rows, "y", board.size, py0, grid)

    w, h = x1 - x0 + 1, y1 - y0 + 1
    if abs(w - h) > 0.1 * max(w, h):
        raise CalibrationError(f"棋盘面不是正方形 ({w}x{h})，误检风险高")

    return Geometry(size=board.size, xs=gxs, ys=gys, bbox=(x0, y0, x1, y1))
