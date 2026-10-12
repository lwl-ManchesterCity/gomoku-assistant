"""棋型扫描测试：每一级标签都用一个手摆的最小局面钉住。

这些标签是 explain 层的输入契约 —— 模型禁止自己判断局面，所以这里判错的
东西会原样进 prompt 变成给用户的错结论。用例全部离线。
"""

from __future__ import annotations

import time
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from gomoku.calibrate import detect_board
from gomoku.vision import parse_board
from gomoku.board import BLACK, Change, EMPTY, WHITE
from gomoku.config import load
from gomoku.pattern import (answer_points, describe, five_cells, five_on_board, hopeless,
                            makes_five, safety_note, scan)
from gomoku.state import empty_board

CFG = load()
SIZE = CFG.board.size
FIXTURES = Path(__file__).parent / "fixtures"


def board_of(*stones):
    return empty_board(SIZE).with_changes([Change(x, y, EMPTY, c) for x, y, c in stones])


def labels(colour: str, *stones) -> list[str]:
    ts = [t.label for t in scan(board_of(*stones)) if t.colour == colour]
    return ts


class 成五点(unittest.TestCase):
    def test_缺口四连的成五点只有那个缺口(self):
        b = board_of((5, 5, BLACK), (6, 5, BLACK), (7, 5, BLACK), (9, 5, BLACK))
        self.assertEqual(five_cells(b, BLACK), ((8, 5),))

    def test_连子两端的补盘点都算(self):
        b = board_of((5, 5, BLACK), (6, 5, BLACK), (7, 5, BLACK), (8, 5, BLACK))
        self.assertEqual(set(five_cells(b, BLACK)), {(4, 5), (9, 5)})

    def test_已成立五连时不再报成五点(self):
        b = board_of(*[(x, 5, BLACK) for x in range(5, 10)])
        self.assertTrue(five_on_board(b, BLACK))

    def test_落点必须真的共线(self):
        b = board_of((5, 5, BLACK), (6, 6, BLACK), (7, 7, BLACK), (9, 9, BLACK))
        self.assertTrue(makes_five(b, 8, 8, BLACK))          # 斜线缺口
        self.assertFalse(makes_five(b, 8, 5, BLACK))         # 看着像、其实不共线


class 四级阶梯(unittest.TestCase):
    def test_四连两端全开是活四(self):
        b = board_of(*[(x, 5, BLACK) for x in range(5, 9)])
        t = next(t for t in scan(b) if t.label == "活四")
        self.assertEqual(set(t.cells), {(4, 5), (9, 5)})

    def test_贴边的四连只是冲四(self):
        self.assertEqual(labels(BLACK, *[(x, 0, BLACK) for x in range(4)]), ["冲四"])

    def test_缺口四连是冲四不是活三(self):
        stones = [(5, 5, BLACK), (6, 5, BLACK), (7, 5, BLACK), (9, 5, BLACK)]
        self.assertEqual(labels(BLACK, *stones), ["冲四"])

    def test_跳三判成活三(self):
        b = board_of((5, 5, BLACK), (7, 5, BLACK), (8, 5, BLACK))
        t = next(t for t in scan(b) if t.label == "活三")
        self.assertEqual(t.cells, ((6, 5),))          # 补上缺口即成活四

    def test_两端被堵死眠三被堵一边降级为眠三(self):
        live = labels(BLACK, (5, 5, BLACK), (6, 5, BLACK), (7, 5, BLACK), (4, 5, WHITE))
        self.assertEqual(live, ["眠三"])
        dead = labels(BLACK,
                      (5, 5, BLACK), (6, 5, BLACK), (7, 5, BLACK),
                      (4, 5, WHITE), (8, 5, WHITE))
        self.assertNotIn("眠三", dead)                 # 两头都堵，什么都不是

    def test_已有五连时本方只报五连但对手棋型照列(self):
        b = board_of(
            *[(x, 5, BLACK) for x in range(5, 10)],
            (1, 1, WHITE), (2, 1, WHITE),                     # 白活二
        )
        self.assertEqual([t.label for t in scan(b) if t.colour == BLACK], ["五连"])
        self.assertEqual([t.label for t in scan(b) if t.colour == WHITE], ["活二"])

    def test_活二按形状判(self):
        self.assertEqual(labels(BLACK, (5, 5, BLACK), (6, 5, BLACK)), ["活二"])
        self.assertEqual(labels(BLACK, (0, 0, BLACK), (1, 0, BLACK)), ["眠二"])


class 双方与措辞(unittest.TestCase):
    def test_黑白各自的棋型都列出且按严重度排序(self):
        b = board_of(
            (5, 5, BLACK), (6, 5, BLACK), (7, 5, BLACK), (8, 5, BLACK),      # 黑活四
            (1, 9, WHITE), (2, 9, WHITE),                                     # 白活二
        )
        self.assertEqual([t.label for t in scan(b)], ["活四", "活二"])

    def test_空盘没有棋型(self):
        self.assertEqual(scan(empty_board(SIZE)), [])
        self.assertEqual(describe(empty_board(SIZE)), "双方均无成活、成四或成三的棋型。")

    def test_摘要里落点与已有连子分得开(self):
        """explain 的硬性契约：四/三级给的是空点，二级给的是已有连子，文字必须点明。

        A/B 实测 DeepSeek 把落点读成已有的子，于是推荐了已被占据的交叉点。
        """
        four_b = board_of((5, 5, BLACK), (6, 5, BLACK), (7, 5, BLACK), (8, 5, BLACK))
        two_b = board_of((5, 5, BLACK), (6, 5, BLACK))
        self.assertEqual(describe(four_b), "黑方：活四，走 E6 J6 任一点即成五，对手一手堵不完")
        self.assertEqual(describe(two_b), "黑方：活二，连子 F6 G6 两端皆空，共 1 处")
        self.assertEqual((four_b.cell(4, 5), four_b.cell(9, 5)), (EMPTY, EMPTY))   # 落点是空位
        self.assertEqual((two_b.cell(5, 5), two_b.cell(6, 5)), (BLACK, BLACK))     # 连子已存在


class 开销(unittest.TestCase):
    def test_中盘局面一次双方扫描在一秒半内(self):
        rgb = np.asarray(Image.open(FIXTURES / "midgame_window.png").convert("RGB"))
        b = parse_board(rgb, detect_board(rgb, CFG), CFG)
        t0 = time.perf_counter()
        scan(b)
        ms = (time.perf_counter() - t0) * 1000
        counts = b.counts()
        print(f"\n  {counts[BLACK]}黑{counts[WHITE]}白 扫描 {ms:.0f}ms → {describe(b)}")
        self.assertLess(ms, 1500)


class 安全网(unittest.TestCase):
    """`safety_note`：推荐手没处理对方先手时必须在出口说一声。

    回归 2026-09-11 16:07 实盘被秒 —— 引擎走增量通道回了 E7（没堵白方 F6-F7-F8 的活三），
    还顶着「必胜(余11)」，下一拍白 F9 成活四直接终局。整盘重灌关掉了那个来源，这里锁兜底。
    """

    @staticmethod
    def xy(name: str) -> tuple[int, int]:
        return ord(name[0]) - 65, int(name[1:]) - 1

    def 判(self, *stones):
        b = board_of(*[(*self.xy(name), c) for name, c in stones])
        return lambda mover, name: safety_note(b, mover, self.xy(name))

    def test_被秒那一手要报堵住的点要静默(self):
        判 = self.判(('H8', BLACK), ('I7', BLACK), ('D8', BLACK),
                     ('F6', WHITE), ('F7', WHITE), ('F8', WHITE))
        self.assertIn("活三", 判(BLACK, "E7"))          # 实盘送出去的那手：两头都没堵
        self.assertIn("活三", 判(BLACK, "H7"))          # 自家扩张也算没应
        self.assertIsNone(判(BLACK, "F5"), "堵 F5 把活三降级成眠三")
        self.assertIsNone(判(BLACK, "F9"), "堵 F9 同理")

    def test_冲四不堵就报(self):
        """白 F5-F8 四连，F9 那端已被黑堵 ⇒ 只剩 F4 一个成五点，不堵就是等死。"""
        判 = self.判(('F9', BLACK), ('H8', BLACK), ('I8', BLACK), ('D2', BLACK),
                     ('F5', WHITE), ('F6', WHITE), ('F7', WHITE), ('F8', WHITE))
        self.assertIn("冲四", 判(BLACK, "H7"))
        self.assertIsNone(判(BLACK, "F4"))

    def test_眠三不必应(self):
        """F5 已被黑占 ⇒ 白方那一排只能做冲四、做不出活四，不强制，别为它刷屏。"""
        判 = self.判(('F5', BLACK), ('H8', BLACK), ('D2', BLACK),
                     ('F6', WHITE), ('F7', WHITE), ('F8', WHITE))
        self.assertIsNone(判(BLACK, "I8"))

    def test_我方反将时不回防也算对(self):
        # 黑 D8-G8、白 H8：C8 是一手成五；白 J6-J8 那条活三此刻无关紧要
        成五 = self.判(('D8', BLACK), ('E8', BLACK), ('F8', BLACK), ('G8', BLACK), ('H8', WHITE),
                       ('J6', WHITE), ('J7', WHITE), ('J8', WHITE))
        self.assertIsNone(成五(BLACK, "C8"))
        # 自家冲四早就摆在那儿、这一手却不去用 ⇒ 两条先手一起败掉，照样要报
        self.assertIn("活三", 成五(BLACK, "C7"))
        # 黑 D8-F8、白 G8：走 C8 做出黑方冲四（唯一五点 B8），白方的活三就得先放一放
        做四 = self.判(('D8', BLACK), ('E8', BLACK), ('F8', BLACK), ('H14', BLACK),
                       ('G8', WHITE), ('J6', WHITE), ('J7', WHITE), ('J8', WHITE))
        self.assertIsNone(做四(BLACK, "C8"))
        self.assertIn("活三", 做四(BLACK, "D9"))

    def test_开局无先手时静默(self):
        self.assertIsNone(safety_note(empty_board(SIZE), BLACK, (7, 7)))

    def test_活四被堵成冲四仍要报堵不住(self):
        """2026-09-28 21:21 那一帧：白 G7 H6 I5 J4 斜活四，两端 F8 K3 全空。

        旧判据是「对方 forcing 等级降了就算在处理」，堵 K3 把活四降成冲四 ⇒ 出口全静默，
        用户看到的是一行普通的 `→ 你的下一手 K3`。等级降了但那一手仍在对方手里，得说。
        """
        判 = self.判(('A1', BLACK), ('O1', BLACK), ('A15', BLACK), ('O15', BLACK),
                     ('G7', WHITE), ('H6', WHITE), ('I5', WHITE), ('J4', WHITE))
        for 堵 in ("K3", "F8"):
            报 = 判(BLACK, 堵)
            self.assertIn("堵不住", 报)
            self.assertIn("冲四", 报)
        self.assertIn("下一手是对方的成五点", 判(BLACK, "K3"))

    def test_对方已经连五时不说下一手(self):
        """终局那帧对方早成五了，「下一手是对方的成五点」不成立。"""
        判 = self.判(('K11', BLACK), ('M13', BLACK), ('O15', BLACK), ('N1', BLACK), ('O2', BLACK),
                     ('A1', WHITE), ('B1', WHITE), ('C1', WHITE), ('D1', WHITE), ('E1', WHITE))
        报 = 判(BLACK, "J10")
        self.assertIn("堵不住", 报)
        self.assertIn("盘上对方已连五", 报)
        self.assertNotIn("下一手是对方的成五点", 报)


class 拦不住(unittest.TestCase):
    """`hopeless`：落完这一手，对方手里还剩没有「下一手即成五」的点。

    给出口标性质用（零进度回手到底是「一手制胜」还是「已经堵不住」）。判据只认四一级：
    三及以下堵一边还剩另一边，那是要应的好手；四及以下那一手已经在对方手里了。
    """

    @staticmethod
    def xy(name: str) -> tuple[int, int]:
        return ord(name[0]) - 65, int(name[1:]) - 1

    def 盘(self, *names_and_colours):
        return board_of(*[(*self.xy(n), c) for n, c in names_and_colours])

    def test_活四降级成冲四是拦不住(self):
        b = self.盘(('A1', BLACK), ('O1', BLACK), ('A15', BLACK), ('O15', BLACK),
                    ('G7', WHITE), ('H6', WHITE), ('I5', WHITE), ('J4', WHITE))
        死 = hopeless(b, BLACK, self.xy("K3"))
        self.assertIsNotNone(死)
        self.assertEqual(死.label, "冲四")
        self.assertEqual(死.names, "F8")

    def test_活三堵成眠三还有得应(self):
        b = self.盘(('A1', BLACK), ('O1', BLACK), ('A15', BLACK),
                    ('F6', WHITE), ('F7', WHITE), ('F8', WHITE))
        self.assertIsNone(hopeless(b, BLACK, self.xy("F5")))

    def test_我方先成五不算拦不住(self):
        """白 F5-F8 已是冲四（F9 被黑堵，只剩 F4），黑这边 C10 一手成五 ⇒ 先赢，谈不上无解。"""
        b = self.盘(('F9', BLACK), ('D10', BLACK), ('E10', BLACK), ('F10', BLACK), ('G10', BLACK),
                    ('F5', WHITE), ('F6', WHITE), ('F7', WHITE), ('F8', WHITE), ('A1', WHITE))
        self.assertIsNone(hopeless(b, BLACK, self.xy("C10")))         # 我方直接成五
        self.assertIsNotNone(hopeless(b, BLACK, self.xy("A15")))      # 这手不去用，白 F4 就成五

    def test_开局无子时无解无从谈起(self):
        self.assertIsNone(hopeless(empty_board(SIZE), BLACK, (7, 7)))


class 与引擎交叉验证(unittest.TestCase):
    """棋型层和 RapFi 是两套独立实现，强制手判定必须吻合。

    选跳三这一例最有区分度：`黑.黑黑` 含缺口，任何「只数连续同色子」的实现都
    会把它当成两个一气而完全看不见，而它实际是一手可成活四的杀形。
    """

    ENGINE = Path(__file__).resolve().parents[1] / load().engine.binary

    @unittest.skipUnless(ENGINE.exists(), "缺引擎二进制")
    def test_跳三的落点就是引擎的杀招(self):
        from gomoku.engine import MATE, Rapfi

        b = board_of((5, 5, BLACK), (7, 5, BLACK), (8, 5, BLACK),
                     (10, 10, WHITE), (11, 11, WHITE), (12, 12, WHITE))
        three = next(t for t in scan(b) if t.label == "活三" and t.colour == BLACK)
        self.assertEqual(three.cells, ((6, 5),))
        with Rapfi(CFG).open() as e:
            plan = e.set_position(b)                    # 3黑3白 → 引擎执黑
            self.assertEqual(e.engine_colour, BLACK)
            self.assertEqual(plan.move, three.cells[0])
            self.assertIsNotNone(plan.score)
            self.assertGreaterEqual(plan.score, MATE - 100)


class 必堵点(unittest.TestCase):
    """出口那行「必堵 X」的契约：只在对方先手能被一手处理掉时才报点。

    锁的是 `answer_points` 与 `safety_note` 用同一套判据：报出来的点必须就是那几个
    成四/成五落点，活四及以上一手堵不住所以不报（报了只会把用户支到错的地方）。
    """

    def test_活三报两端(self):
        # 2026-09-11 16:07 那局：白 F6 F7 F8 竖活三，黑走，正解 F5 或 F9
        b = board_of((7, 7, BLACK), (8, 6, BLACK), (3, 7, BLACK),
                     (5, 5, WHITE), (5, 6, WHITE), (5, 7, WHITE))
        self.assertEqual(set(answer_points(b, BLACK).split()), {"F5", "F9"})

    def test_冲四只报那一点(self):
        # 白 F7 G7 H7 I7，J7 端被黑堵死 ⇒ 只剩 E7 一个成五点
        b = board_of((9, 6, BLACK), (5, 5, BLACK), (2, 2, BLACK),
                     (5, 6, WHITE), (6, 6, WHITE), (7, 6, WHITE), (8, 6, WHITE))
        self.assertEqual(answer_points(b, BLACK), "E7")

    def test_活四不报点(self):
        """一手堵不住的东西没有「必堵点」，报出来只会误导。"""
        b = board_of((5, 5, BLACK), (6, 4, BLACK), (2, 2, BLACK),
                     (6, 6, WHITE), (7, 6, WHITE), (8, 6, WHITE), (9, 6, WHITE))
        self.assertIsNone(answer_points(b, BLACK))

    def test_对方没有先手时不报(self):
        b = board_of((5, 5, BLACK), (6, 4, BLACK), (2, 2, BLACK),
                     (8, 8, WHITE), (9, 9, WHITE))
        self.assertIsNone(answer_points(b, BLACK))


if __name__ == "__main__":
    unittest.main()
