"""离线探针：锁定之后能不能跳过每拍的全量标定，省多少、在多大位移内可信。

模型按 `loop.Watcher` 的真实形状来，不用 np.roll：把 fixture 贴到一块非木纹底上，
棋盘整体挪 N px、**ROI 取样框不动**，然后拿锁定时缓存的那份格点去重读整盘。
np.roll 会把画面另一头回绕进来，整格平移时边界露不出馅，量出来的灵敏度是假的。

每条位移都给两个数：门卫的分离度（判据本身）和**实际误读格数**（拿旧格点重读的差）。
复用只在「放行且误读 0」的档位上成立，所以阈值要卡在误读还是 0 的那一档以内。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, "src")

from gomoku import config
from gomoku.calibrate import Geometry, detect_board, grid_contrast, verify_grid
from gomoku.vision import parse_board

CFG = config.load()
FIX = Path(__file__).resolve().parents[1] / "tests" / "fixtures"
DESKTOP = 90          # 假桌面：非木纹、非棋盘色


def rgb_of(name: str) -> np.ndarray:
    with Image.open(FIX / name) as im:
        return np.asarray(im.convert("RGB"))


def roi_of(rgb: np.ndarray, geom: Geometry) -> np.ndarray:
    """按 capture.grab_rect 的外扩比例裁出锁定后每拍真正会拿到的那块。"""
    h, w = rgb.shape[:2]
    x0, y0, x1, y1 = geom.bbox
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    hw, hh = (x1 - x0) / 2.0 * CFG.capture.roi_expand, (y1 - y0) / 2.0 * CFG.capture.roi_expand
    px, py = max(int(cx - hw), 0), max(int(cy - hh), 0)
    return rgb[py : min(int(py + 2 * hh), h), px : min(int(px + 2 * hw), w)]


def best(fn, n: int = 30) -> float:
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000)
    ts.sort()
    return ts[len(ts) // 2]


def misread(frame: np.ndarray, geom: Geometry, ref: list[str]) -> int:
    grid = ["".join(row) for row in parse_board(frame, geom, CFG).grid]
    return sum(a != b for a, b in zip(grid, ref))


def main() -> int:
    for name in ("midgame_window.png", "empty_window.png", "marked_ai_stone.png"):
        fixture = rgb_of(name)
        fh, fw = fixture.shape[:2]
        ox = oy = 250
        desk = np.full((1600, 1600, 3), DESKTOP, np.uint8)
        desk[oy : oy + fh, ox : ox + fw] = fixture
        rect_geom = detect_board(desk, CFG)
        frame0 = roi_of(desk, rect_geom)
        geom = detect_board(frame0, CFG)          # 锁定时缓存的那一份
        ref = ["".join(row) for row in parse_board(frame0, geom, CFG).grid]
        cell = round(geom.spacing[0])

        def dragged(dx: int, dy: int) -> np.ndarray:
            d = np.full((1600, 1600, 3), DESKTOP, np.uint8)
            d[oy + dy : oy + dy + fh, ox + dx : ox + dx + fw] = fixture
            return roi_of(d, rect_geom)           # 取样框不动 = ROI 复用时的同一矩形

        t_det = best(lambda: detect_board(frame0, CFG))
        t_ver = best(lambda: verify_grid(frame0, geom, CFG))
        print(f"\n### {name}  ROI 帧 {frame0.shape[1]}x{frame0.shape[0]}px  格距 {geom.spacing[0]:.1f}px"
              f"  一格 = {cell}px")
        print(f"    detect_board {t_det:.1f}ms   verify_grid {t_ver:.1f}ms   省 {t_det - t_ver:.1f}ms/拍")
        print("    棋盘在帧里挪 N px（格点不动）：分离度 → 门禁 → 误读格数")
        for d in (0, 1, 2, 3, 4, 5, 6, 8, 27, cell, 2 * cell, 3 * cell, -cell):
            f = dragged(d, d)
            cx, cy = grid_contrast(f, geom)
            gate = "放行" if verify_grid(f, geom, CFG) else "拦下"
            bad = misread(f, geom, ref)
            flag = "  ← 放行却读错" if gate == "放行" and bad else ""
            print(f"      {d:>4}px  x={cx:>7.1f} y={cy:>7.1f}  {gate}  误读 {bad:>2}格{flag}")
        if any(misread(dragged(d, d), geom, ref) and verify_grid(dragged(d, d), geom, CFG)
               for d in (0, 1, 2, 3, 4, 5, 6, 8, 27, cell, 2 * cell, 3 * cell, -cell)):
            print("    结论：门禁有放行却读错的档位，不能复用。")
        else:
            print("    结论：凡是放行的档位误读都是 0，可以复用。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
