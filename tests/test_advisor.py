"""建议层挪线程的契约：采集侧永不因问答停拍，过期答案不出口。

回归的是 2026-09-11 实盘首局测出来的唯一真缺陷 —— `on_tick` 原来在采集线程里同步调
`engine.ask()`，一手冷灌 1.3–1.5s 就把 0.3s 的节奏拖成 ~1.7s，
每一拍必然读到「我方落子 + 对手应手」两手。
"""

from __future__ import annotations

import threading
import time
import unittest

from gomoku.__main__ import advise_lines
from gomoku.advisor import RESTART_CAP, Advisor, Endgame, question_position, worth_asking
from gomoku.board import BLACK, EMPTY, WHITE, Change, parse_colour
from gomoku.config import load
from gomoku.engine import EngineError, Plan
from gomoku.state import empty_board

CFG = load()
SIZE = CFG.board.size
PLAN = Plan(move=(7, 7), score=445.0, depth=21, pv=("H8", "G7"), elapsed_ms=1500.0)


def board_of(*stones):
    return empty_board(SIZE).with_changes([Change(x, y, EMPTY, c) for x, y, c in stones])


B1 = empty_board(SIZE).with_changes([Change(7, 7, EMPTY, BLACK)])
B2 = B1.with_changes([Change(8, 8, EMPTY, WHITE)])
B3 = B2.with_changes([Change(6, 6, EMPTY, BLACK)])


def 等到(条件, 超时: float = 3.0):
    deadline = time.perf_counter() + 超时
    while time.perf_counter() < deadline:
        if 条件():
            return True
        time.sleep(0.005)
    return False


class 卡住的引擎:
    """ask() 停在闸门前等放行；用来冒充「引擎正在搜索」的那 1.5s。

    `fails` = 接下来多少次 ask 抛 EngineError（冒充引擎失稳）；`hold=False` 时不等闸门，
    给「连着问」的用例省掉反复放行。
    """

    def __init__(self, plan: Plan | None = PLAN, fails: int = 0, hold: bool = True):
        self.engine_colour = WHITE
        self.plan = plan
        self.asks: list = []
        self.fails = fails
        self.hold = hold
        self.restarts = 0
        self.entered = threading.Event()
        self.release = threading.Event()

    def _maybe_fail(self):
        if self.fails:
            self.fails -= 1
            raise EngineError("引擎进程已退出")
        return self.plan

    def ask(self, board):
        self.asks.append(board)
        if not self.hold:
            return self._maybe_fail()
        self.entered.set()
        self.release.wait(5)
        return self._maybe_fail()

    def restart(self):
        self.restarts += 1


class 收条:
    def __init__(self):
        self.lock = threading.Lock()
        self.plans: list = []
        self.notes: list[str] = []

    def on_plan(self, board, plan, mover, rolled):
        with self.lock:
            self.plans.append((board, plan.move, mover, rolled))

    def on_note(self, text):
        with self.lock:
            self.notes.append(text)


class Worker契约(unittest.TestCase):
    def 装(self, engine):
        got = 收条()
        adv = Advisor(engine, on_plan=got.on_plan, on_note=got.on_note)
        self.addCleanup(adv.close)
        return adv, got

    def test_递交局面不阻塞调用线程(self):
        """引擎卡在 1.5s 里，采集线程的 submit 也必须立刻返回。"""
        eng = 卡住的引擎()
        adv, got = self.装(eng)
        t0 = time.perf_counter()
        adv.submit(B1)
        cost_ms = (time.perf_counter() - t0) * 1000
        self.assertTrue(eng.entered.wait(2), "worker 没起来")
        self.assertLess(cost_ms, 50, f"submit 被引擎拖了 {cost_ms:.0f}ms")
        self.assertEqual(got.plans, [])
        eng.release.set()
        self.assertTrue(等到(lambda: len(got.plans) == 1), "放行后建议没出口")
        self.assertEqual(eng.asks, [B1])

    def test_忙时只保留最新盘面(self):
        """问答期间连来三手，出口的那条建议必须是最盘面，旧的不排队、不补答。"""
        eng = 卡住的引擎()
        adv, got = self.装(eng)
        adv.submit(B1)
        self.assertTrue(eng.entered.wait(2))
        eng.entered.clear()
        adv.submit(B2)
        adv.submit(B3)
        eng.release.set()
        self.assertTrue(等到(lambda: len(got.plans) == 1), "只该回一条建议")
        self.assertEqual(got.plans[0][0], B3)
        self.assertEqual(got.plans[0][2], WHITE)
        self.assertEqual(eng.asks, [B1, B3])          # B1 已问过，镜像照常推进
        self.assertEqual(got.notes, ["建议已过期（盘面又变了），丢弃"])

    def test_引擎异常自愈一次并把同一问补回来(self):
        """失稳不能等到下一手才有建议：换进程重启，同一个盘面立刻再问一次。"""
        eng = 卡住的引擎(fails=1)
        adv, got = self.装(eng)
        adv.submit(B1)
        self.assertTrue(eng.entered.wait(2))
        eng.release.set()
        self.assertTrue(等到(lambda: len(got.plans) == 1), "重启后同一问没补回来")
        self.assertEqual(eng.asks, [B1, B1])
        self.assertEqual(eng.restarts, 1)
        self.assertEqual(got.notes, ["引擎异常: 引擎进程已退出 —— 重启引擎，这一问补回来"])
        self.assertEqual(adv._restarts, 0, "问成功就该把重启计数归零，不能攒着")

        adv.submit(B2)                                # 之后照常干活
        self.assertTrue(等到(lambda: len(got.plans) == 2), "报过一次错就不干活了")
        self.assertEqual(eng.asks, [B1, B1, B2])

    def test_重启次数有上限(self):
        """每次都失稳时不能每拍起一个新进程：自愈满 RESTART_CAP 次后只报一行。"""
        eng = 卡住的引擎(fails=99, hold=False)
        adv, got = self.装(eng)
        expect = 0
        for k in range(1, RESTART_CAP + 2):
            adv.submit(B1)
            expect += 2 if k <= RESTART_CAP else 1     # 重启那次两句，之后只句「要重启进程才有」
            self.assertTrue(等到(lambda: len(got.notes) == expect),
                            f"第 {k} 次提问后提示数不对: {got.notes}")
        self.assertEqual(eng.restarts, RESTART_CAP)
        self.assertEqual(adv._restarts, RESTART_CAP)
        self.assertEqual(got.plans, [])
        self.assertIn("推荐要重启进程才有", got.notes[-1])

    def test_镜像已覆盖时安静跳过(self):
        eng = 卡住的引擎(plan=None)
        adv, got = self.装(eng)
        adv.submit(B1)
        self.assertTrue(eng.entered.wait(2))
        eng.release.set()
        time.sleep(0.1)
        self.assertEqual((got.plans, got.notes), ([], []))

    def test_关闭后在途答案不再出口(self):
        eng = 卡住的引擎()
        adv, got = self.装(eng)
        adv.submit(B1)
        self.assertTrue(eng.entered.wait(2))
        adv.close(join_s=0.2)                         # worker 卡在引擎里，不等它
        eng.release.set()
        time.sleep(0.15)
        self.assertEqual(got.plans, [])
        self.assertEqual(got.notes, [])
        self.assertFalse(adv.alive)

    def test_撤过手的那颗子要原样到出口(self):
        """撤过手的问题问的是「用户点下去之前」那一盘，出口不带这颗子就会指错人。"""
        eng = 卡住的引擎()
        adv, got = self.装(eng)
        撤 = Change(6, 6, EMPTY, BLACK)
        adv.submit(B1, 撤)
        self.assertTrue(eng.entered.wait(2))
        eng.release.set()
        self.assertTrue(等到(lambda: len(got.plans) == 1))
        self.assertEqual(eng.asks, [B1])
        self.assertEqual(got.plans[0][3], 撤)             # 撤掉的那颗子必须原样到出口


class 该问哪个局面(unittest.TestCase):
    def test_平子时轮到黑(self):
        self.assertEqual(empty_board(SIZE).side_to_move(), BLACK)
        self.assertEqual(B1.side_to_move(), WHITE)           # 黑1白0 → 轮到白
        self.assertEqual(B2.side_to_move(), BLACK)           # 黑1白1 → 轮到黑

    def test_已经轮到我就直接问(self):
        pos, 撤 = question_position(B2, [Change(8, 8, EMPTY, WHITE)], BLACK)
        self.assertEqual((pos, 撤), (B2, None))

    def test_实盘形状要撤掉我刚落的子(self):
        """一拍提交「对手白子 + 我方黑子」后盘面恒为 黑=白+1，那是对手的回合。"""
        ch = board_of((7, 7, BLACK), (8, 8, WHITE), (6, 6, BLACK))
        changes = [Change(8, 8, EMPTY, WHITE), Change(6, 6, EMPTY, BLACK)]
        self.assertEqual(ch.side_to_move(), WHITE)
        pos, 撤 = question_position(ch, changes, BLACK)
        self.assertEqual(撤, Change(6, 6, EMPTY, BLACK))     # 撤的就是我刚落的那颗
        self.assertEqual(pos, B2)                            # 撤掉 G7 黑子
        self.assertEqual(pos.side_to_move(), BLACK)          # 回到「轮到我」的局面

    def test_撤不出唯一解释时宁可不问(self):
        """同一拍里出现两颗我方子（或一颗都没有）就说不清哪颗是刚落的手。"""
        two = board_of((7, 7, BLACK), (8, 8, BLACK))
        changes = [Change(7, 7, EMPTY, BLACK), Change(8, 8, EMPTY, BLACK)]
        self.assertEqual(question_position(two, changes, BLACK), (None, None))
        none_ = board_of((7, 7, BLACK), (8, 8, BLACK), (6, 6, WHITE))
        self.assertEqual(none_.side_to_move(), WHITE)
        self.assertEqual(
            question_position(none_, [Change(6, 6, EMPTY, WHITE)], BLACK), (None, None))

    def test_颜色词解析(self):
        for word in ("黑", "b", "BLACK", " 执黑 "):
            self.assertEqual(parse_colour(word), BLACK, word)
        for word in ("白", "w", "white", "白棋"):
            self.assertEqual(parse_colour(word), WHITE, word)
        with self.assertRaises(ValueError):
            parse_colour("红")


class 什么时候值得问(unittest.TestCase):
    """回归「刚锁定那一拍被吞掉」——栽过两次：空盘的第一手、中途启动时的中盘。"""

    def test_中途启动基准盘已轮到我(self):
        """进程在 B2（黑1白1，轮到黑）中途起来，那一拍 changes 恒为空，仍必须问。"""
        self.assertTrue(worth_asking(B2, [], None))

    def test_同一盘面不许反复问(self):
        self.assertFalse(worth_asking(B2, [], B2))

    def test_盘面推进要问(self):
        self.assertTrue(worth_asking(B3, [Change(6, 6, EMPTY, BLACK)], B2))

    def test_重同步换盘即使没提交落子也要问(self):
        """基准盘整盘被换掉时 changes 可能为空，那是「刚见到」的一个新局面。"""
        self.assertTrue(worth_asking(B3, [], B2))


class 终局闸(unittest.TestCase):
    """回归 2026-09-11 实盘：获胜连五整排闪烁，闪灭那一拍曾在已赢的盘上印出「必败」。"""

    五连 = [(x, 7, BLACK) for x in (9, 10, 11, 12, 13)]          # J8..N8，实盘赢的就是这条
    白四 = [(x, 9, WHITE) for x in (2, 3, 4, 5)]                 # 凑成合法形状：黑=白+1

    def test_连五一出现就不再问且只报一次(self):
        gate = Endgame()
        self.assertTrue(gate.admit(board_of(*self.白四))[0])     # 中盘：照问
        self.assertEqual(gate.admit(board_of(*self.五连, *self.白四)), (False, BLACK))
        self.assertEqual(gate.admit(board_of(*self.五连, *self.白四)), (False, None))

    def test_闪灭那一拍不许把闸打开(self):
        """闪掉 5 颗同色子 = 黑10白14 那种不可能的形状，不是「局面变小所以重开了」。"""
        gate = Endgame()
        gate.admit(board_of(*self.五连, *self.白四))
        for _ in range(3):                                       # 闪烁会连着好几十拍
            self.assertEqual(gate.admit(board_of(*self.白四)), (False, None))

    def test_重开一局要放开闸(self):
        gate = Endgame()
        gate.admit(board_of(*self.五连, *self.白四))
        self.assertEqual(gate.admit(board_of((7, 7, BLACK))), (True, None))


class 出口行(unittest.TestCase):
    """`advise_lines`：终端上那几行的全部生成逻辑，2026-09-28 三处静默都锁在这里。

    ① 复盘那一支必须也审用户实际落点（原来只审引擎推荐的那一手，整局 ⚠ 0 次）；
    ② 引擎推荐的那一手已经堵不住时要说「堵不住」；③ 零进度回手不许一律写成「一手制胜」。
    """

    # 白 F6 F7 F8 竖活三（必堵 F5 F9）：2026-09-11 16:07 被秒的那张盘
    活三 = board_of((7, 7, BLACK), (8, 6, BLACK), (3, 7, BLACK),
                    (5, 5, WHITE), (5, 6, WHITE), (5, 7, WHITE))
    # 白 G7 H6 I5 J4 斜活四，两端 F8 K3 全空：2026-09-28 21:21 那一帧
    活四 = board_of((0, 0, BLACK), (14, 0, BLACK), (0, 14, BLACK), (14, 14, BLACK),
                    (6, 6, WHITE), (7, 5, WHITE), (8, 4, WHITE), (9, 3, WHITE))

    def test_复盘对用户实际落点也报警(self):
        首选 = Plan(move=(5, 4), score=120.0, depth=20, elapsed_ms=1300.0)   # F5，堵活三
        偏离 = Change(4, 6, EMPTY, BLACK)                                     # 用户落了 E7
        行 = advise_lines(self.活三, 首选, BLACK, BLACK, 偏离, self.活三)
        self.assertTrue(行[0].startswith("→ 复盘：你落 E7 之前那一步，引擎首选 F5"))
        self.assertTrue(any(行.startswith("  ⚠ E7 没处理对方先手") for 行 in 行), 行)
        self.assertIn("  → 必堵 F5 F9", 行)
        self.assertEqual(sum(1 for 行 in 行 if 行.startswith("  ⚠")), 1, "引擎首选 F5 本身不该报")

    def test_照建议落但堵不住时照样报(self):
        """引擎 0ms 回了 K3（活四只能降级成冲四），出口不能把它念成一条普通建议。"""
        建议 = Plan(move=(10, 2), elapsed_ms=0.4)                             # 零进度、无 depth
        行 = advise_lines(self.活四, 建议, BLACK, BLACK, None, self.活四)
        self.assertIn("棋型层判已无解", 行[0])
        self.assertTrue(any(行.startswith("  ⚠ K3 堵不住") for 行 in 行), 行)
        self.assertFalse(any("必堵" in 行 for 行 in 行), "一手堵不住的东西没有必堵点，报了指错地方")

    def test_零进度但不该标成已无解(self):
        开局 = Plan(move=(7, 7), elapsed_ms=3.0)                               # 开局库直接回手
        行 = advise_lines(empty_board(SIZE), 开局, BLACK, BLACK, None, None)
        self.assertIn("未给进度行（一手制胜或开局库）", 行[0])
        self.assertEqual(len(行), 1)

    def test_进度行照旧带盘指纹(self):
        行 = advise_lines(self.活三, PLAN, BLACK, BLACK, None, self.活三)
        self.assertTrue(行[0].startswith("→ 你的下一手 H8（1500ms 盘3:3） depth=21 eval=445"))
        self.assertIn("pv=H8 G7", 行[0])

    def test_颜色不对那一拍只说无建议(self):
        行 = advise_lines(self.活三, PLAN, WHITE, BLACK, None, self.活三)
        self.assertEqual(行, ["→ 这一拍引擎回答的是白方 H8，不是你的回合，本拍无建议"])


if __name__ == "__main__":
    unittest.main()
