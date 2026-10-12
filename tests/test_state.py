"""状态同步层的离线测试：不碰屏幕，直接喂一串 Board 进 Tracker。

覆盖面就是 memory 里记下的三条实盘约束：单拍多手、颜色只来自像素、
异常局面要能丢弃也能重同步。
"""

from __future__ import annotations

import unittest
from dataclasses import replace

from gomoku.board import BLACK, Board, EMPTY, WHITE
from gomoku.config import load
from gomoku.state import Tracker, empty_board

CFG = load()
SIZE = CFG.board.size
LETTERS = "ABCDEFGHIJKLMNO"


def board_of(**stones: str) -> Board:
    """board_of(G8='X', H7='O') → 15x15 Board。"""
    rows = [[EMPTY] * SIZE for _ in range(SIZE)]
    for name, code in stones.items():
        x, y = LETTERS.index(name[0]), int(name[1:]) - 1
        rows[y][x] = code
    return Board(tuple(tuple(r) for r in rows))


def committed(sync) -> list[str]:
    return [str(c) for c in sync.changes]


def cfg_with(**state) -> object:
    return replace(CFG, state=replace(CFG.state, **state))


def fresh(cfg=None) -> Tracker:
    """已用空盘占住基准的 Tracker：首次读数会被直接采纳，先喂空盘才看得到后续消抖。

    默认开 `confirm_ticks=2`：`Confirm` 那一组测的就是「连续同色才提交」这套机制本身，
    而它不是出厂档位（出厂是 1，见 config/default.toml 的依据注释）。
    """
    t = Tracker(cfg if cfg is not None else cfg_with(confirm_ticks=2))
    t.update(board_of())
    return t


class Confirm(unittest.TestCase):
    def test_出厂档位读到就提交(self):
        """默认 confirm_ticks=1 的依据：563 拍 / 40 颗子里中间帧 0 处，多等的那一拍换不来过滤。"""
        self.assertEqual(CFG.state.confirm_ticks, 1)
        t = fresh(CFG)
        sync = t.update(board_of(G8=BLACK))
        self.assertEqual(committed(sync), ["G8:.->X"])
        self.assertEqual(sync.held, [])
        self.assertEqual(sync.board, board_of(G8=BLACK))

    def test_新子需连续两拍同色才提交(self):
        t = fresh()
        first = t.update(board_of(G8=BLACK))
        self.assertEqual(committed(first), [])
        self.assertEqual([str(c) for c in first.held], ["G8:.->X"])
        second = t.update(board_of(G8=BLACK))
        self.assertEqual(committed(second), ["G8:.->X"])
        self.assertEqual(second.board, board_of(G8=BLACK))
        self.assertEqual(second.held, [])

    def test_中间帧消失后永不提交(self):
        t = fresh()
        t.update(board_of(G8=BLACK))     # 疑似半过渡帧
        t.update(board_of())             # 动画结束前又空了
        third = t.update(board_of())
        self.assertEqual(committed(third), [])
        self.assertEqual(third.board, empty_board(SIZE))
        self.assertEqual(t._pending, {})

    def test_同一点颜色反复横跳时不提交(self):
        t = fresh()
        for raw in (board_of(G8=BLACK), board_of(G8=WHITE), board_of(G8=BLACK), board_of(G8=WHITE)):
            sync = t.update(raw)
            self.assertEqual(committed(sync), [])
        self.assertEqual(sync.board.cell(6, 7), EMPTY)

    def test_颜色稳定后立刻提交不受历史抖动影响(self):
        t = fresh()
        t.update(board_of(G8=BLACK))
        t.update(board_of(G8=WHITE))
        self.assertEqual(committed(t.update(board_of(G8=WHITE))), ["G8:.->O"])

    def test_确认的悔棋照样提交(self):
        t = fresh()
        t.update(board_of(G8=BLACK))
        t.update(board_of(G8=BLACK))                       # 已提交
        sync = t.update(board_of(G8=WHITE))                 # 颜色变了，先挂起
        self.assertEqual(committed(sync), [])
        sync = t.update(board_of())                         # 真的拿走了
        self.assertEqual(committed(sync), [])
        sync = t.update(board_of())
        self.assertEqual(committed(sync), ["G8:X->."])


class Ownership(unittest.TestCase):
    def test_一拍多手全部提交(self):
        """同一拍的两手必须一起提交：出厂档（1，读到即提）当拍全提，消抖档（2）第二拍全提。"""
        two = board_of(F9=BLACK, C12=WHITE)   # 复刻 14:11:07.218 那一拍
        for ct, warmup in ((1, 0), (2, 1)):
            with self.subTest(confirm_ticks=ct):
                t = Tracker(cfg_with(confirm_ticks=ct))
                t.update(board_of())
                for _ in range(warmup):
                    self.assertEqual(committed(t.update(two)), [])
                sync = t.update(two)
                self.assertEqual(committed(sync), ["F9:.->X", "C12:.->O"])
                self.assertEqual(sync.board.counts(),
                                 {BLACK: 1, WHITE: 1, EMPTY: SIZE * SIZE - 2})

    def test_连续两拍同色落子不要求黑白交替(self):
        t = Tracker(CFG)
        s0, s1, s2 = board_of(), board_of(G8=BLACK), board_of(G8=BLACK, H8=BLACK)
        acc: list[str] = []
        for raw in (s0, s0, s1, s1, s2, s2):
            acc += committed(t.update(raw))
        self.assertEqual(acc, ["G8:.->X", "H8:.->X"])
        self.assertEqual(t.board.counts()[BLACK], 2)
        self.assertEqual(t.board.counts()[WHITE], 0)


class Priming(unittest.TestCase):
    def test_首次读数直接采纳不误判为异常(self):
        mid = board_of(G6=BLACK, H7=BLACK, H8=BLACK, G7=WHITE, G8=WHITE, H9=WHITE)
        t = Tracker(CFG)
        sync = t.update(mid)
        self.assertIsNone(sync.reason)
        self.assertEqual(committed(sync), [])
        self.assertEqual(sync.board, mid)

    def test_已有十子的开局盘面一次成型(self):
        many = board_of(**{f"{LETTERS[i]}{i + 1}": BLACK for i in range(6)},
                        **{f"{LETTERS[i]}{i + 2}": WHITE for i in range(5)})
        t = Tracker(CFG)
        sync = t.update(many)
        self.assertEqual(sync.board, many)
        self.assertEqual(sync.board.counts()[BLACK], 6)


class BulkChange(unittest.TestCase):
    def test_单拍变化超阈值整拍丢弃(self):
        t = Tracker(CFG)
        t.update(board_of())
        sync = t.update(board_of(G8=BLACK, H8=BLACK, I8=BLACK, J8=BLACK))
        self.assertEqual(committed(sync), [])
        self.assertIsNotNone(sync.reason)
        self.assertIn("整拍丢弃", sync.reason)
        self.assertEqual(sync.board, empty_board(SIZE))

    def test_异常局面连续稳定后按新基准重同步(self):
        t = Tracker(cfg_with(resync_ticks=3))
        weird = board_of(G8=BLACK, H8=BLACK, I8=BLACK, J8=BLACK, K8=BLACK)
        t.update(board_of())
        for _ in range(2):
            sync = t.update(weird)
            self.assertIsNotNone(sync.reason)
            self.assertIn("整拍丢弃", sync.reason)
        sync = t.update(weird)
        self.assertIn("重同步", sync.reason)
        self.assertEqual(sync.board, weird)
        self.assertEqual(len(sync.changes), 5)

    def test_异常局面不稳定时不重同步(self):
        t = Tracker(cfg_with(resync_ticks=2))
        t.update(board_of())
        base = {"G8": BLACK, "H8": BLACK, "I8": BLACK, "J8": BLACK, "K8": BLACK}
        for i in range(2, 8):
            stones = dict(base, **{f"{LETTERS[i]}9": WHITE})
            sync = t.update(board_of(**stones))
            self.assertIn("整拍丢弃", sync.reason or "")
            self.assertEqual(committed(sync), [])
        self.assertEqual(t.board, empty_board(SIZE))


class NoChange(unittest.TestCase):
    def test_局面不变时不产生任何事件(self):
        t = Tracker(CFG)
        static = board_of(G8=BLACK, H8=WHITE)
        t.update(static)
        for _ in range(3):
            sync = t.update(static)
            self.assertEqual((committed(sync), sync.held, sync.reason), ([], [], None))


if __name__ == "__main__":
    unittest.main()
