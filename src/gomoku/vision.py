"""解析：按标定结果在 225 个交叉点上采样，判成黑/白/空。

两条判据，全部来自实测：

- **亮度**定黑白：黑子中位 ~47，空格 ~210。
- **暖度 (R−B)** 定有无：星位 ≈85、棋盘底色 ~93，棋子 ~0–1。白子亮度 207–212 落在空格亮度
  202–214 里面，只看亮度在数学上就分不开白子和空点，所以有无必须交给色度。

每点取 (2r+1)² 的中位数而不是均值，网格线只会占窗口一小条，中位数吃得住。

2026-09-11 撤掉了曾经的第三条判据「实心度」（整窗亮度低分位）。它是 09-10 为了挡一枚号称
半透明的光标落子预览加的，而那次核对错了：被挡掉的是**实心**白子——App 会在人机刚落、我方
尚未应对的白子上压一个黑色「＋」，纯黑的臂把整窗低分位打到 0，于是对手每一手都要等我方应了
才看得见，盯屏助手因此永远慢一手。教训写在这儿：**判据宁可少一条，也不许拿它去挡肉眼确认
在场的子**；真出现需要区分实心/半透明的那天，先拿截图证据说话。
"""

from __future__ import annotations

import numpy as np

from .board import BLACK, EMPTY, WHITE, Board, coord_name
from .calibrate import Geometry
from .config import Config


class SampleError(Exception):
    """采样窗口越界，通常是 ROI 裁得太紧。"""


def patch_medians(rgb: np.ndarray, geom: Geometry, radius: int) -> np.ndarray:
    """(N,3) 的邻域 RGB 中位数，行优先。"""
    h, w = rgb.shape[:2]
    r = radius
    for c in (min(geom.xs) - r, max(geom.xs) + r):
        if c < 0 or c >= w:
            raise SampleError(f"采样窗口超出图像宽度 (x={c}, w={w})")
    for c in (min(geom.ys) - r, max(geom.ys) + r):
        if c < 0 or c >= h:
            raise SampleError(f"采样窗口超出图像高度 (y={c}, h={h})")
    patches = np.stack([
        rgb[cy - r : cy + r + 1, cx - r : cx + r + 1].reshape(-1, 3).astype(np.float32)
        for cy in geom.ys
        for cx in geom.xs
    ])
    return np.median(patches, axis=1).astype(np.float32)


def _lum_warmth(medians: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lum = 0.299 * medians[:, 0] + 0.587 * medians[:, 1] + 0.114 * medians[:, 2]
    return lum, medians[:, 0] - medians[:, 2]


def classify(medians: np.ndarray, cfg: Config) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回 (亮度, 暖度, 每点代码)。先用暖度把木纹和星位整个排除，剩下的中性点按亮度分黑白。"""
    b = cfg.board
    lum, warmth = _lum_warmth(medians)
    codes = np.full(lum.shape, EMPTY, dtype="<U1")
    cold = warmth < b.max_warmth
    codes[cold & (lum <= b.black_max_lum)] = BLACK
    codes[cold & (lum >= b.white_min_lum)] = WHITE
    return lum, warmth, codes


def parse_board(rgb: np.ndarray, geom: Geometry, cfg: Config) -> Board:
    lum, warmth, codes = classify(patch_medians(rgb, geom, cfg.board.sample_radius), cfg)
    return Board(tuple(tuple(codes[y * geom.size : (y + 1) * geom.size]) for y in range(geom.size)))


def diagnostics(rgb: np.ndarray, geom: Geometry, cfg: Config) -> list[dict]:
    """逐点读数，用于调阈值和排查误读。"""
    lum, warmth, codes = classify(patch_medians(rgb, geom, cfg.board.sample_radius), cfg)
    out = []
    k = 0
    for y in range(geom.size):
        for x in range(geom.size):
            out.append(
                {
                    "x": x,
                    "y": y,
                    "name": coord_name(x, y),
                    "px": geom.point(x, y),
                    "lum": float(lum[k]),
                    "warmth": float(warmth[k]),
                    "code": str(codes[k]),
                }
            )
            k += 1
    return out
