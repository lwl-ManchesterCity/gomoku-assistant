"""离线回放测试：真实截图 → 期望矩阵，锁住感知层的正确性。

跑法：PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
fixture 由 tests/make_fixtures.py 从真机截图裁出，期望值手写在 ground_truth.json。
"""

from __future__ import annotations

import json
import sys
import unittest
from unittest import mock
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gomoku import config
from gomoku.board import BLACK, WHITE, coord_name
from gomoku.calibrate import CalibrationError, Geometry, detect_board, verify_grid
from gomoku.capture import Frame
from gomoku.loop import Watcher
from gomoku.vision import parse_board

FIXTURES = Path(__file__).parent / "fixtures"
TRUTH = json.loads((Path(__file__).parent / "ground_truth.json").read_text())
CFG = config.load()


def cfg_with(**state) -> object:
    """出厂 `confirm_ticks=1`（读到即提交）；测消抖机制本身的两支要显式调回 2。"""
    return replace(CFG, state=replace(CFG.state, **state))


def analyze(path: Path, radius: int | None = None):
    cfg = replace(CFG, board=replace(CFG.board, sample_radius=radius)) if radius else CFG
    rgb = np.asarray(Image.open(path).convert("RGB"))
    geom = detect_board(rgb, cfg)
    return rgb, geom, parse_board(rgb, geom, cfg)


def expected_map(gt: dict) -> dict[str, str]:
    return {name: color for color in (BLACK, WHITE) for name in gt[color]}


def shift(geom: Geometry, dx: int, dy: int) -> Geometry:
    """把一副格点在帧内平移 —— 等价于棋盘朝反方向挪同样的量。"""
    return Geometry(
        size=geom.size,
        xs=tuple(x + dx for x in geom.xs),
        ys=tuple(y + dy for y in geom.ys),
        bbox=(geom.bbox[0] + dx, geom.bbox[1] + dy, geom.bbox[2] + dx, geom.bbox[3] + dy),
    )


class GoldenReplay(unittest.TestCase):
    def test_每面盘的子位与颜色都符合期望(self):
        for name, gt in TRUTH["boards"].items():
            with self.subTest(fixture=name):
                _, _, board = analyze(FIXTURES / name)
                self.assertEqual(board.stones(), expected_map(gt))

    def test_空格必须保持全空(self):
        for name, gt in TRUTH["boards"].items():
            with self.subTest(fixture=name):
                _, _, board = analyze(FIXTURES / name)
                c = board.counts()
                self.assertEqual(c[BLACK], len(gt[BLACK]))
                self.assertEqual(c[WHITE], len(gt[WHITE]))
                self.assertEqual(c["."], 225 - len(expected_map(gt)))

    def test_悬停帧与基线帧零差分(self):
        for a, b in TRUTH["identical_pairs"]:
            with self.subTest(pair=f"{a} vs {b}"):
                _, _, ba = analyze(FIXTURES / a)
                _, _, bb = analyze(FIXTURES / b)
                self.assertEqual(ba.diff(bb), [])

    def test_采样半径在3到12之间结果不变(self):
        _, _, base = analyze(FIXTURES / "midgame_window.png")
        for r in range(3, 13):
            with self.subTest(radius=r):
                _, _, board = analyze(FIXTURES / "midgame_window.png", radius=r)
                self.assertEqual(base.diff(board), [])

    def test_星位不得读成棋子(self):
        _, _, board = analyze(FIXTURES / "empty_window.png")
        for name in ("D4", "L4", "D12", "L12", "H8", "H4", "D8"):
            with self.subTest(point=name):
                x, y = (ord(name[0]) - 65, int(name[1:]) - 1)
                self.assertEqual(board.cell(x, y), ".")

    def test_带标记的人机子必须读成白子(self):
        """人机刚落、我方还没应的那颗白子上压着一个黑色「＋」（=「该你应了」），应一手它就消失。
        按整窗低分位判实心会被这个「＋」打死，把对手最新的一手读成空点——「永远慢一手」就是这么
        来的。实心度现在只看扣掉十字带后的四象限盖，「＋」挡不住白子。"""
        for name, cell in (("midgame_window.png", "H9"), ("midgame_fullscreen.png", "H9"),
                           ("hover_fullscreen.png", "H9"), ("marked_ai_stone.png", "I9")):
            with self.subTest(fixture=name):
                _, _, board = analyze(FIXTURES / name)
                x, y = ord(cell[0]) - 65, int(cell[1:]) - 1
                self.assertEqual(board.cell(x, y), WHITE)

    def test_标定结果的几何自洽(self):
        for name in TRUTH["boards"]:
            with self.subTest(fixture=name):
                _, geom, _ = analyze(FIXTURES / name)
                self.assertEqual(len(geom.xs), 15)
                self.assertEqual(len(geom.ys), 15)
                sx, sy = geom.spacing
                self.assertAlmostEqual(sx / sy, 1.0, delta=0.02)
                self.assertGreater(sx, 20)


class AbortBehaviour(unittest.TestCase):
    """读不出来时必须失败，绝不能给出一副看着像样的错矩阵。"""

    def test_棋盘被裁掉一半时标定失败(self):
        for name in TRUTH["must_fail"]:
            with self.subTest(fixture=name):
                with self.assertRaises(CalibrationError):
                    analyze(FIXTURES / name)

    def test_画面里没有棋盘时标定失败(self):
        for rgb in (
            np.full((900, 900, 3), 128, dtype=np.uint8),
            np.random.default_rng(0).integers(0, 256, (900, 900, 3), dtype=np.uint8),
        ):
            with self.subTest(shape=rgb.shape):
                with self.assertRaises(CalibrationError):
                    detect_board(rgb, CFG)

    def test_整帧全黑要报全黑而不是报遮挡(self):
        """合盖睡眠时 WindowServer 没有活动显示器，screencapture 返回全零帧。
        这种帧和「棋盘被挡住」的处置方向完全相反，报错必须把两者分开。"""
        with self.assertRaises(CalibrationError) as cm:
            detect_board(np.zeros((900, 900, 3), dtype=np.uint8), CFG)
        self.assertIn("全黑", str(cm.exception))
        self.assertNotIn("遮挡", str(cm.exception))

    def test_网格规格不是15时标定失败(self):
        rgb, geom, _ = analyze(FIXTURES / "midgame_window.png")
        odd = replace(CFG, board=replace(CFG.board, size=19))
        with self.assertRaises(CalibrationError):
            detect_board(rgb, odd)


class GridReuse(unittest.TestCase):
    """缓存格点复用的门卫：小抖动放行，整格滑移必须判假。

    拖窗时的真实形状是「棋盘在帧里挪、缓存格点不动」。夹具本身棋盘占满整帧，
    所以把盘面贴到非木纹底上再挪格点，等价且不引入回绕假象。
    """

    MARGIN = 200

    def setUp(self):
        self.rgb, self.geom, self.ref = analyze(FIXTURES / "midgame_window.png")
        h, w = self.rgb.shape[:2]
        self.canvas = np.full((h + 2 * self.MARGIN, w + 2 * self.MARGIN, 3), 90, np.uint8)
        self.canvas[self.MARGIN : self.MARGIN + h, self.MARGIN : self.MARGIN + w] = self.rgb
        self.on_canvas = shift(self.geom, self.MARGIN, self.MARGIN)

    def test_不挪动时放行(self):
        self.assertTrue(verify_grid(self.canvas, self.on_canvas, CFG))

    def test_抖几像素放行且读数不变(self):
        for dx, dy in ((1, 1), (2, -2), (3, 3)):
            with self.subTest(shift=(dx, dy)):
                self.assertTrue(verify_grid(self.canvas, shift(self.on_canvas, dx, dy), CFG))
                board = parse_board(self.canvas, shift(self.on_canvas, dx, dy), CFG)
                self.assertEqual(self.ref.diff(board), [])

    def test_挪过六像素就判假(self):
        for d in (6, 10, 27, 40):
            with self.subTest(shift=d):
                self.assertFalse(verify_grid(self.canvas, shift(self.on_canvas, d, d), CFG))

    def test_整格滑移必须判假(self):
        """网格图案自相似，一格整数倍时线谷照样分明 —— 中位数判据在这里放行并误读十几格，
        所以判据取百分位而不是中位。这条是那次误读的直接回归测试。"""
        step = int(round(self.geom.spacing[0]))
        for k in (1, 2, 3):
            for sign in (1, -1):
                with self.subTest(cells=k, sign=sign):
                    self.assertFalse(
                        verify_grid(self.canvas, shift(self.on_canvas, sign * step * k, sign * step * k), CFG)
                    )

    def test_格点跑出帧外时判假而不是算出nan(self):
        far = shift(self.on_canvas, self.MARGIN + 600, 0)
        self.assertFalse(verify_grid(self.canvas, far, CFG))


class WatcherFollow(unittest.TestCase):
    """不碰真屏幕地验证跟随逻辑：把 fixture 贴到一块假桌面上，看 bbox 是否稳定跟随。"""

    OFFSET = (250, 300)

    def setUp(self):
        fixture = np.asarray(
            Image.open(FIXTURES / "midgame_window.png").convert("RGB")
        )
        h, w = fixture.shape[:2]
        self.fixture, self.fh, self.fw = fixture, h, w
        self.canvas = np.full((1600, 1600, 3), 90, dtype=np.uint8)
        ox, oy = self.OFFSET
        self.canvas[oy : oy + h, ox : ox + w] = fixture
        self.pristine = self.canvas.copy()
        self.blank = np.full((1600, 1600, 3), 90, dtype=np.uint8)
        self.calls: list[str] = []
        self.current = self.canvas

    def _move(self, dx: int, dy: int) -> None:
        """整面棋盘在假桌面上挪一笔 —— 用户在屏上拖窗就是这个形状。"""
        self.canvas[...] = self.blank
        ox, oy = self.OFFSET
        self.canvas[oy + dy : oy + dy + self.fh, ox + dx : ox + dx + self.fw] = self.fixture
        self.current = self.canvas

    def _stub(self):
        test = self

        class StubCapture:
            def grab_full(self):
                test.calls.append("full")
                return Frame(rgb=test.current, origin=(0, 0))

            def grab_rect(self, bbox, expand=1.0):
                test.calls.append("roi")
                x0, y0, x1, y1 = bbox
                cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
                hw, hh = (x1 - x0) / 2.0 * expand, (y1 - y0) / 2.0 * expand
                px, py = int(cx - hw), int(cy - hh)
                return Frame(rgb=test.current[py : py + int(2 * hh), px : px + int(2 * hw)], origin=(px, py))

            def close(self):
                pass

        return StubCapture()

    def _watcher(self, cfg=None):
        return Watcher(cfg=cfg or CFG, cap=self._stub(), log=lambda *a, **k: None)

    def _paint(self, geom, x: int, y: int, color=(46, 46, 46), r: int = 25):
        """在假桌面上按像素画一颗子，模拟刚落下的一手。"""
        cx, cy = geom.point(x, y)
        ys, xs = np.ogrid[-r : r + 1, -r : r + 1]
        patch = (xs * xs + ys * ys) <= r * r
        self.canvas[cy - r : cy + r + 1, cx - r : cx + r + 1][patch] = color
        return coord_name(x, y)

    def test_定位后ROI跟随保持同一屏坐标(self):
        w = self._watcher()
        first = w.step()
        self.assertEqual(self.calls, ["full"])
        self.assertIsNotNone(first)
        located = w.screen_bbox
        ox, oy = self.OFFSET
        self.assertEqual(located, (ox + 40, oy + 40, ox + 841, oy + 841))
        second = w.step()
        self.assertEqual(self.calls, ["full", "roi"])
        self.assertEqual(w.screen_bbox, located)
        self.assertEqual(second.mode, "ROI")
        self.assertEqual(first.board.diff(second.board), [])

    def test_连续丢失后才退回全屏重定位(self):
        w = self._watcher()
        w.step()
        self.current = self.blank
        limit = CFG.capture.roi_relocate_fails
        for i in range(limit):
            self.assertIsNone(w.step(), msg=f"第 {i + 1} 次丢失应当失败")
        self.assertTrue(w.locked is False, "连续丢失后必须退回全屏")
        self.assertEqual(self.calls, ["full", *["roi"] * limit])
        self.current = self.canvas
        self.assertIsNotNone(w.step())
        self.assertEqual(w.fails, 0)

    def test_出厂档下新子当拍就进局面(self):
        """时效契约：人机落下的那一子，采集到它的那一拍就提交，不再压一拍。"""
        w = self._watcher()
        w.step()
        name = self._paint(w.geometry, 5, 8)
        tick = w.step()
        self.assertEqual([str(c) for c in tick.changes], [f"{name}:.->X"])
        self.assertEqual(tick.held, [])
        self.assertEqual(tick.board.stones()[name], BLACK)

    def test_新子需第二拍复验才提交(self):
        """消抖机制本身的端到端覆盖：出厂档是 1，这里显式调回 2。"""
        w = self._watcher(cfg_with(confirm_ticks=2))
        w.step()                               # 第一拍走全屏，geometry 即画布坐标
        name = self._paint(w.geometry, 5, 8)   # F9 在中局 fixture 上是空点
        held = w.step()
        self.assertEqual([str(c) for c in held.changes], [])
        self.assertEqual([str(c) for c in held.held], [f"{name}:.->X"])
        self.assertIsNone(held.board.stones().get(name))
        second = w.step()
        self.assertEqual([str(c) for c in second.changes], [f"{name}:.->X"])
        self.assertEqual(second.board.stones()[name], BLACK)
        self.assertEqual(second.board.counts()[BLACK], 4)

    def test_动画中间帧不承担入局面(self):
        """消抖机制本身的端到端覆盖：出厂档是 1（中间帧实测抓不到），这里显式调回 2。"""
        w = self._watcher(cfg_with(confirm_ticks=2))
        w.step()
        name = self._paint(w.geometry, 5, 8)
        self.assertEqual([str(c) for c in w.step().held], [f"{name}:.->X"])
        self.canvas[...] = self.pristine       # 下一拍这颗子没了 = 过渡帧
        for i in range(3):
            sync = w.step()
            self.assertEqual(sync.changes, [], msg=f"第 {i + 1} 拍不该提交闪现的子")
            self.assertNotIn(name, sync.board.stones())
        self.assertNotIn(name, w.tracker.board.stones())

    def test_锁定后棋盘没挪就不重标定(self):
        """门卫放行时这一拍不调 detect_board —— 省下的就是这 16ms。"""
        w = self._watcher()
        w.step()                                # 全屏定位
        w.step()                                # ROI 上标定一次，格点坐标系换成 ROI 帧
        cached = w.geometry
        with mock.patch("gomoku.loop.detect_board", side_effect=detect_board) as spy:
            tick = w.step()
        self.assertIsNotNone(tick)
        self.assertEqual(spy.call_count, 0, "棋盘没挪，这一拍不该重标定")
        self.assertIs(w.geometry, cached, "复用的必须还是那一份格点")
        self.assertEqual(tick.mode, "ROI")
        self.assertTrue(w.events[-1]["reuse"], "沿用旧格点必须记进日志，误读定责要靠它")

    def test_窗口抖三像素时沿用格点且读数不错(self):
        w = self._watcher()
        ref = w.step().board
        w.step()
        self._move(3, 3)
        with mock.patch("gomoku.loop.detect_board", side_effect=detect_board) as spy:
            tick = w.step()
        self.assertEqual(spy.call_count, 0, "3px 抖动在门卫的容忍档位内（实测到 4px 都放行），不该重标定")
        self.assertEqual(tick.raw.diff(ref), [], "复用格点期间不许读出假落子")

    def test_抖过五像素就退回全量标定(self):
        w = self._watcher()
        ref = w.step().board
        w.step()
        self._move(5, 5)
        with mock.patch("gomoku.loop.detect_board", side_effect=detect_board) as spy:
            tick = w.step()
        self.assertEqual(spy.call_count, 1, "门卫拦下就必须重新标定，不能带着旧格点读")
        self.assertFalse(w.events[-1]["reuse"], "重标定的一拍日志要如实记成没沿用")
        self.assertEqual(tick.raw.diff(ref), [])

    def test_拖窗一格不得静默错位(self):
        """整格平移时网格图案看着完全对齐，误读出来的是一副「到处都变了」的盘面；
        而状态层只看变化数，连续 6 拍稳定就会把它当真局面采纳。所以必须在这一拍就拦住。"""
        w = self._watcher()
        ref = w.step().board
        w.step()
        self._move(round(w.geometry.spacing[0]), round(w.geometry.spacing[1]))
        ticks = [w.step() for _ in range(8)]
        ok = [t for t in ticks if t is not None]
        self.assertTrue(ok, "挪动后要么重新定位、要么报错，不能一直静默失败")
        for i, t in enumerate(ok):
            with self.subTest(tick=i):
                self.assertEqual(t.raw.diff(ref), [], "错位格点读出的盘面绝不能流出去")


class BoardModel(unittest.TestCase):
    def test_坐标名往返(self):
        self.assertEqual(coord_name(6, 5), "G6")
        self.assertEqual(ord("G") - ord("A"), 6)

    def test_差分给出变化点与前后状态(self):
        from gomoku.board import Board

        _, _, board = analyze(FIXTURES / "midgame_window.png")
        grid = [list(row) for row in board.grid]
        grid[8][6] = BLACK
        changed = board.diff(Board(tuple(tuple(r) for r in grid)))
        self.assertEqual(len(changed), 1)
        self.assertEqual(str(changed[0]), "G9:.->X")


if __name__ == "__main__":
    unittest.main()
