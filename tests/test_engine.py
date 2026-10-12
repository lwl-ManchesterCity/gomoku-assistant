"""引擎适配层测试。

解析用例纯离线；问答用例跑 vendor/rapfi 里的真二进制（缺文件就整体跳过），
锁的是三条实测事实：坐标不镜像、`board` 编辑态按石子数奇偶决定引擎执哪方、
以及 ask() **无论跟随还是偏离推荐都整盘重灌**（一行 `turn` 都不发），
问答之后维持「引擎镜像 == 真实盘面 + 它刚回的那一手」。
"""

from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path

from gomoku.board import BLACK, EMPTY, WHITE, Change, coord_name
from gomoku.config import load
from gomoku.engine import MATE, EngineError, Rapfi, parse_eval, parse_move, parse_progress
from gomoku.state import empty_board

CFG = load()
SIZE = CFG.board.size
ENGINE = Path(__file__).resolve().parents[1] / CFG.engine.binary
FAST = replace(CFG, engine=replace(CFG.engine, timeout_ms=400))

# 黑四连 A1-A5 列（x=0,y=0..3）：补盘点既是黑方的五连点，也是白方唯一的防点。
BLACK_FOUR = [(0, 0, BLACK), (0, 1, BLACK), (0, 2, BLACK), (0, 3, BLACK)]
WHITE_SCATTER = [(10, 10, WHITE), (11, 11, WHITE), (12, 12, WHITE)]
BLOCK_SQUARE = (0, 4)


def build(*stones) -> "object":
    return empty_board(SIZE).with_changes([Change(x, y, EMPTY, c) for x, y, c in stones])


def spy(engine: Rapfi) -> list[str]:
    """录下适配器写给引擎的每一行，用来分辨走了 turn 还是走了 board 重灌。"""
    sent: list[str] = []
    original = engine._command

    def record(line: str, want_ok: bool = True) -> None:
        sent.append(line)
        original(line, want_ok=want_ok)

    engine._command = record   # type: ignore[method-assign]
    return sent


class Parse(unittest.TestCase):
    def test_裸坐标(self):
        self.assertEqual(parse_move("7,7"), (7, 7))
        self.assertEqual(parse_move(" 0,14 "), (0, 14))
        for line in ("OK", "MESSAGE Depth 2-3 | Eval 88", "COORDS 7,7", "H8", "pass"):
            self.assertIsNone(parse_move(line), line)

    def test_杀棋分数(self):
        self.assertEqual(parse_eval("445"), 445)
        self.assertEqual(parse_eval("-1096"), -1096)
        self.assertEqual(parse_eval("+M3"), MATE - 3)
        self.assertEqual(parse_eval("-M14"), -(MATE - 14))
        for junk in ("", "M", "abc", "12.5"):
            self.assertIsNone(parse_eval(junk), junk)

    def test_进度行取深度评分和pv(self):
        got = parse_progress("MESSAGE Depth 2-3 | Eval 88 | Time 1ms | E5 E7")
        self.assertEqual(got, (2, 88.0, ("E5", "E7")))

    def test_收尾Speed行没有pv段(self):
        got = parse_progress("MESSAGE Speed 596K | Depth 26-10 | Eval -M14 | Node 293K | Time 493ms")
        self.assertEqual(got, (26, -(MATE - 14), ()))

    def test_非进度行返回None(self):
        for line in ("MESSAGE OptiTime 1323ms | MaxTime 1470ms",
                     "MESSAGE mix9svq nnue: load weight from mix9svqfreestyle_bsmix.bin.lz4",
                     "OK"):
            self.assertIsNone(parse_progress(line), line)


@unittest.skipIf(not ENGINE.exists(), f"没有本地引擎 {ENGINE}")
class RapfiProtocol(unittest.TestCase):
    def test_空盘回天元(self):
        with Rapfi(CFG) as e:
            plan = e.set_position(empty_board(SIZE))
            self.assertEqual(plan.move, (7, 7), "开局不在天元说明坐标被镜像或换序了")
            self.assertEqual(plan.name, coord_name(*plan.move), "H8")
            self.assertEqual(e.engine_colour, BLACK, "空盘该由引擎执先")

    def test_奇偶决定引擎执哪方(self):
        with Rapfi(CFG) as e:
            plan = e.set_position(build(*BLACK_FOUR, *WHITE_SCATTER))
            self.assertEqual(e.engine_colour, WHITE, "4黑3白 → 轮到白")
            self.assertEqual(plan.move, BLOCK_SQUARE, "白方只有这一个防点")
        with Rapfi(CFG) as e:
            plan = e.set_position(build(*BLACK_FOUR, *WHITE_SCATTER, (13, 13, WHITE)))
            self.assertEqual(e.engine_colour, BLACK, "4黑4白 → 轮到黑")
            self.assertEqual(plan.move, BLOCK_SQUARE, "黑方该补成五连")
            self.assertIsNone(plan.score, "一手成五时引擎直接回手，不给 Depth/Eval 进度行")
            self.assertEqual((plan.depth, plan.pv), (None, ()))

    def test_放宽预算只改发给引擎那一档(self):
        """实盘 1.5s 是为时效；离线复盘要问得比它深，就得单独给 `info timeout_turn` 报一个数。"""
        e = Rapfi(CFG, timeout_ms=6000)
        sent = spy(e)                       # open() 之前挂上才录得到握手那三行
        e.open()
        try:
            self.assertIn("info timeout_turn 6000", sent)
            self.assertNotIn(f"info timeout_turn {CFG.engine.timeout_ms}", sent)
            self.assertEqual(e._budget_s, 6.0 + 6, "读回手的时限要跟着放宽，否则引擎还在想我们就收线")
        finally:
            e.close()

    def test_不指定就照配置那一档走(self):
        self.assertEqual(Rapfi(CFG)._ms, CFG.engine.timeout_ms)

    def test_编辑态报错会以异常抛出(self):
        e = Rapfi(CFG).open()
        try:
            e._command("board", want_ok=False)
            e._command("99,99,1", want_ok=False)   # 越界坐标
            e._command("done", want_ok=False)
            with self.assertRaises(EngineError):
                e._await_move()
        finally:
            e.close()

    def test_跟随推荐也整盘重灌(self):
        """引擎只缺对手一颗子时也不许走 `turn`：那条回手是上一轮 pv 的下一格，评分还是继承的。"""
        with Rapfi(FAST) as e:
            first = e.set_position(empty_board(SIZE))
            sent = spy(e)
            board = build((first.move[0], first.move[1], BLACK), (5, 5, WHITE))
            second = e.ask(board)
            self.assertEqual(sent[0], "board", "增量通道已停用，每一问都得重灌")
            self.assertFalse([s for s in sent if s.startswith("turn")], [sent])
            self.assertEqual(board.cell(*second.move), EMPTY, "回手必须是空点")
            self._assert_mirror(e, board)

    def test_偏离推荐触发整盘重灌(self):
        with Rapfi(FAST) as e:
            first = e.set_position(empty_board(SIZE))
            sent = spy(e)
            # 用户没走天元那手，自己落在 A15；AI 应了一手 K11 —— 此刻 1黑1白，轮到黑
            board = build((0, 14, BLACK), (10, 10, WHITE))
            plan = e.ask(board)
            self.assertIn("board", sent, "镜像里那颗预测子落空了，必须重灌")
            self.assertEqual(e.engine_colour, BLACK)
            self.assertNotEqual(plan.move, first.move, "重灌后按新局面重新给建议")
            self._assert_mirror(e, board)

    def test_镜像已覆盖就不问(self):
        with Rapfi(FAST) as e:
            e.set_position(build(*BLACK_FOUR, *WHITE_SCATTER))
            sent = spy(e)
            self.assertIsNone(e.ask(e.model), "没有新信息时不该重播旧结论")
            self.assertEqual(sent, [], "而且一个字都不该发给引擎")

    def _assert_mirror(self, e: Rapfi, board) -> None:
        """不变式：引擎镜像 == 真实盘面 + 它自己刚回的那一手。"""
        extra = board.diff(e.model)
        self.assertEqual(len(extra), 1, [str(c) for c in extra])
        self.assertEqual(extra[0].after, e.engine_colour)
        self.assertEqual(extra[0].before, EMPTY)
