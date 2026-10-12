"""从真实截图生成 golden fixture（只裁棋盘那块，不含桌面信息，可直接提交）。

跑法：PYTHONPATH=src .venv/bin/python tests/make_fixtures.py
期望值不在这里生成 —— ground_truth.json 是人工核对后写死的，本脚本只负责裁图。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gomoku import config
from gomoku.__main__ import load_image
from gomoku.calibrate import detect_board

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tests" / "fixtures"
PAD = 40

# (源文件, fixture 名) —— 源文件在本机桌面上，换机器后重新截图再跑本脚本
SOURCES = {
    "empty_window.png": "/Users/lwl/Desktop/截屏2026-09-10 10.32.24.png",
    "midgame_window.png": "/Users/lwl/Desktop/截屏2026-09-10 10.51.42.png",
    "midgame_fullscreen.png": "/tmp/qoder_live_130048.png",
    "hover_fullscreen.png": "/Users/lwl/Desktop/截屏2026-09-10 13.24.55.png",
    # 人机未应的子（带「＋」）。源截图 截屏2026-09-11 14.10.02.png 已不在桌面，
    # 源文件换掉后需要重新截一局「刚被人机应过一手」的画面再跑本脚本。
    "marked_ai_stone.png": "/tmp/face_141002.png",
}


def crop(rgb: np.ndarray, bbox: tuple[int, int, int, int], y1: int | None = None) -> np.ndarray:
    x0, y0, x1, by1 = bbox
    x0 = max(x0 - PAD, 0)
    y0 = max(y0 - PAD, 0)
    x1 = min(x1 + PAD, rgb.shape[1] - 1)
    by1 = min(by1 + PAD, rgb.shape[0] - 1) if y1 is None else min(y0 + y1, rgb.shape[0] - 1)
    return rgb[y0 : by1 + 1, x0 : x1 + 1]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = config.load()
    for name, src in SOURCES.items():
        path = Path(src)
        if not path.exists():
            print(f"跳过 {name}: 源文件不存在 {src}")
            continue
        rgb = load_image(path)
        bbox = detect_board(rgb, cfg).bbox
        piece = crop(rgb, bbox)
        Image.fromarray(piece.astype(np.uint8)).save(OUT / name)
        print(f"{name}: 源 {rgb.shape[1]}x{rgb.shape[0]} bbox={bbox} → {piece.shape[1]}x{piece.shape[0]}px")

    full = load_image(Path(SOURCES["midgame_window.png"]))
    bbox = detect_board(full, cfg).bbox
    Image.fromarray(crop(full, bbox, y1=380).astype(np.uint8)).save(OUT / "board_partial.png")
    print(f"board_partial.png: 只保留 {bbox[3] - bbox[1]}px 高的上半盘（应标定失败）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
