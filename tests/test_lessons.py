"""经验层测试：全部离线 —— 引擎问答用注入的假 `ask`，文档一律写在临时目录里。

契约有两半。一半是**对照的正确性**：一手棋要归到它落下去之前那个盘面上，归错了整份
经验就是假的（`position_before` 的三种拍形）。另一半是**文档能往返**：写进去的条目
下次启动必须原样读回来，人手写的笔记不能把解析打崩。
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from gomoku.board import BLACK, EMPTY, WHITE, Board, Change, coord_name
from gomoku.engine import Plan
from gomoku.lessons import (
    Lesson,
    append,
    default_at,
    for_prompt,
    frames_from_jsonl,
    frames_from_text,
    latest_log,
    load,
    position_before,
    read_frames,
    result_of,
    round_facts,
    todo_count,
    xy,
)
from gomoku.state import empty_board

SIZE = 15


def board_of(*stones) -> Board:
    return empty_board(SIZE).with_changes([Change(x, y, EMPTY, c) for x, y, c in stones])


# 2026-09-11 实盘那形状：白 F6 F7 F8 是活三（必堵 F5 F9），黑 H8 I7 D8 三子分散。
白活三 = ((5, 5, WHITE), (5, 6, WHITE), (5, 7, WHITE))
黑三子 = ((7, 7, BLACK), (8, 6, BLACK), (3, 7, BLACK))
开局 = board_of(*白活三, *黑三子)


def ask_fixed(*names):
    """按调用顺序返回固定首选的假引擎；用完就回 None，等于引擎没回手。"""
    队列 = list(names)

    def ask(board: Board) -> Plan | None:
        if not 队列:
            return None
        return Plan(move=xy(队列.pop(0)), elapsed_ms=100.0)

    return ask


class 日志读帧(unittest.TestCase):
    def test_JSONL按每行的grid还原盘面(self):
        rec = {"t": "2026-09-28T18:35:23.689", "mode": "ROI",
               "grid": ["".join(row) for row in 开局.grid]}
        out = frames_from_jsonl(json.dumps(rec) + "\n" + '{"t":"x","error":"标定失败"}\n')
        self.assertEqual(len(out), 1, "报错行没有 grid，不该被当成一个局面")
        self.assertEqual(out[0][0], "2026-09-28T18:35:23.689")
        self.assertEqual(out[0][1].diff(开局), [])

    def test_终端日志按表头下的十五行还原盘面(self):
        """他从终端整屏粘出来的就是这种文本，09-28 三份样本全按它读。"""
        头 = "   " + " ".join(coord_name(x, 0)[:-1] for x in range(SIZE))
        体 = "\n".join(f"{y + 1:>2} " + " ".join(row) for y, row in enumerate(开局.grid))
        out = frames_from_text(f"[18:35:23.689] [ROI 143ms] 落子 1 手\n{头}\n{体}\n")
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0][0], "18:35:23.689", "时间戳从盘面上方最近的方括号行取")
        self.assertEqual(out[0][1].diff(开局), [])

    def test_形状不对的一块直接丢弃(self):
        self.assertEqual(frames_from_text("   A B C\n 1 . . O\n 2 . . ."), [])

    def test_连续重复的局面只留一份(self):
        """0.3s 一拍，同一盘面会连印好几拍；复盘要的是局面序列不是拍数。"""
        text = json.dumps({"t": "1", "grid": ["".join(r) for r in 开局.grid]})
        变 = json.dumps({"t": "2",
                         "grid": ["".join(r) for r in 开局.with_changes([Change(5, 4, EMPTY, BLACK)]).grid]})
        with TemporaryDirectory() as d:
            p = Path(d) / "events.jsonl"
            p.write_text("\n".join([text, text, 变, 变]), encoding="utf-8")
            out = read_frames(p)
        self.assertEqual([t for t, _ in out], ["1", "2"])


class 一手归到哪个局面(unittest.TestCase):
    """`position_before`：一手棋必须归到它落下去**之前**的那个盘面，归错整份经验就是假的。"""

    def test_只出现我这颗子时问上一拍(self):
        后 = 开局.with_changes([Change(5, 4, EMPTY, BLACK)])          # 黑4白3，F5 是应一手
        self.assertEqual(position_before(开局, 后, BLACK), (开局, "F5"))

    def test_我这手在先时仍问上一拍(self):
        """同一拍里出现我这手加对手一应，上一拍轮到我 ⇒ 我先落，要问的就是上一拍那个盘。"""
        后 = 开局.with_changes([Change(5, 4, EMPTY, BLACK), Change(10, 10, EMPTY, WHITE)])
        盘, 名 = position_before(开局, 后, BLACK)
        self.assertEqual((盘, 名), (开局, "F5"), "对手的 K11 不该进我要问的局面")

    def test_对手先落时撤掉我这颗才是要问的局面(self):
        """上一拍轮不到我 ⇒ 这一拍里对手先落，我按下去时的局面 = 现盘撤掉我的子。"""
        白先 = board_of(*白活三, *黑三子, (10, 10, WHITE))              # 黑3白4：轮到黑=我
        后 = 白先.with_changes([Change(5, 4, EMPTY, BLACK)])            # 黑4白4
        盘, 名 = position_before(白先, 后, BLACK)
        self.assertEqual((名, 盘.counts()[BLACK], 盘.counts()[WHITE]), ("F5", 3, 4))
        self.assertEqual(盘.diff(后), [Change(5, 4, EMPTY, BLACK)], "撤掉的只能是黑子")

    def test_说不清哪颗是我落的就不对照(self):
        后 = 开局.with_changes([Change(5, 4, EMPTY, BLACK), Change(5, 8, EMPTY, BLACK)])
        self.assertIsNone(position_before(开局, 后, BLACK))

    def test_第一拍没有上一手(self):
        self.assertIsNone(position_before(None, 开局, BLACK))


class 逐手对照(unittest.TestCase):
    def _两拍(self, 我落: str):
        x, y = xy(我落)
        return [("18:35:00.000", 开局),
                ("18:35:06.000", 开局.with_changes([Change(x, y, EMPTY, BLACK)]))]

    def test_占到必堵点的那手记成一致且没判词(self):
        局 = round_facts(self._两拍("F5"), BLACK, ask_fixed("F5"))
        self.assertEqual(len(局.turns), 1)
        t = 局.turns[0]
        self.assertEqual((t.pick, t.play, t.at), ("F5", "F5", "18:35:06.000"))
        self.assertTrue(t.followed)
        self.assertEqual(t.must, "F5 F9")
        self.assertTrue(t.blocked)
        self.assertEqual(t.note, "")

    def test_该堵没堵要单独成一条(self):
        """2026-09-11 被秒那一手：白活三必堵 F5 F9，引擎和用户都往别处落。"""
        局 = round_facts(self._两拍("J10"), BLACK, ask_fixed("J10"))
        t = 局.turns[0]
        self.assertFalse(t.blocked)
        self.assertIn("没处理对方先手", t.note)
        self.assertEqual([x.play for x in 局.misses], ["J10"])

    def test_偏离按坐标名比而不是按棋型(self):
        局 = round_facts(self._两拍("J10"), BLACK, ask_fixed("F5"))
        self.assertEqual(局.dev_line(), "1 处（J10≠F5）")
        self.assertEqual([t.play for t in 局.deviations], ["J10"])

    def test_没有先手要应时不能算成漏堵(self):
        空 = empty_board(SIZE)
        局 = round_facts([("t1", 空), ("t2", 空.with_changes([Change(7, 7, EMPTY, BLACK)]))],
                         BLACK, ask_fixed("H8"))
        self.assertIsNone(局.turns[0].blocked)
        self.assertEqual(局.misses, [])

    def test_引擎没回手的那拍计入skipped(self):
        局 = round_facts(self._两拍("F5"), BLACK, ask_fixed())
        self.assertEqual((len(局.turns), 局.skipped), (0, 1))

    def test_对不出局面的拍不算漏也不计skipped(self):
        """对手落子的那一拍没有我的子，本来就没得对照，不能记成「说不清」。"""
        白应 = 开局.with_changes([Change(5, 4, EMPTY, BLACK), Change(10, 10, EMPTY, WHITE)])
        局 = round_facts([("t1", 开局), ("t2", 开局.with_changes([Change(5, 4, EMPTY, BLACK)])),
                          ("t3", 白应)], BLACK, ask_fixed("F5"))
        self.assertEqual((len(局.turns), 局.skipped), (1, 0))

    def test_事实行把判词与必堵都带上(self):
        局 = round_facts(self._两拍("J10"), BLACK, ask_fixed("F5"))
        行 = 局.fact_lines()[0]
        self.assertIn("我落 J10，引擎首选 F5（偏离）", 行)
        self.assertIn("必堵 F5 F9", 行)
        self.assertIn("没占到", 行)
        self.assertNotIn("depth", 行)                    # 评分与深度不进复盘，一分都不给


class 进度与总数(unittest.TestCase):
    """10 分钟一局有 20~30 问、放宽预算后要跑两三分钟 —— 不报总数用户不知道还要等多久。"""

    白应 = 开局.with_changes([Change(5, 4, EMPTY, BLACK), Change(10, 10, EMPTY, WHITE)])

    def 三拍(self):
        return [("t0", 开局),
                ("t1", 开局.with_changes([Change(5, 4, EMPTY, BLACK)])),      # 我落 F5
                ("t2", self.白应),                                            # 对手应一手
                ("t3", self.白应.with_changes([Change(6, 5, EMPTY, BLACK)]))]  # 我落 G6

    def test_总数和真正问出来的手数一致(self):
        手 = todo_count(self.三拍(), BLACK)
        局 = round_facts(self.三拍(), BLACK, ask_fixed("F5", "G6"))
        self.assertEqual((手, len(局.turns)), (2, 2))

    def test_对手落子那一拍不进总数(self):
        """`todo_count` 必须走同一套「对不出局面就不问」的判据，否则进度条永远走不满。"""
        只对手落 = [("t1", 开局), ("t2", 开局.with_changes([Change(10, 10, EMPTY, WHITE)]))]
        self.assertEqual(todo_count(只对手落, BLACK), 0)
        self.assertEqual(len(round_facts(只对手落, BLACK, ask_fixed("F5")).turns), 0)

    def test_每成一手续报一次(self):
        报: list[int] = []
        round_facts(self.三拍(), BLACK, ask_fixed("F5", "G6"), 报.append)
        self.assertEqual(报, [1, 2])

    def test_引擎没回手时不报进度(self):
        报: list[int] = []
        round_facts(self.三拍(), BLACK, ask_fixed(), 报.append)
        self.assertEqual(报, [])


class 找最新一局(unittest.TestCase):
    def 落(self, runs: Path, 名: str, 内容: str) -> Path:
        runs.mkdir(exist_ok=True)
        p = runs / 名
        p.write_text(内容, encoding="utf-8")
        return p

    def 一行(self) -> str:
        return json.dumps({"t": "2026-09-28T18:35:23.689",
                           "grid": ["".join(r) for r in 开局.grid]}) + "\n"

    def test_按文件名里的时间戳取最新(self):
        """`run_log` 把日期时间写进文件名且定长，所以字典序就是时间序。"""
        with TemporaryDirectory() as d:
            runs = Path(d)
            for 名 in ("advise-2026-09-28-183523.jsonl", "advise-2026-09-29-093011.jsonl",
                       "advise-2026-09-28-201010.jsonl"):
                self.落(runs, 名, self.一行())
            self.assertEqual(latest_log(runs).name, "advise-2026-09-29-093011.jsonl")

    def test_空壳文件不算一局(self):
        with TemporaryDirectory() as d:
            runs = Path(d)
            self.落(runs, "advise-2026-09-28-183523.jsonl", self.一行())
            self.落(runs, "advise-2026-09-29-093011.jsonl", "")
            self.assertEqual(latest_log(runs).name, "advise-2026-09-28-183523.jsonl")

    def test_只有报错行的那一份也不能挡路(self):
        """屏上没盘面的那次启动，每行都是 error：有大小却没 grid，比空壳更挡路。"""
        with TemporaryDirectory() as d:
            runs = Path(d)
            self.落(runs, "advise-2026-09-28-183523.jsonl", self.一行())
            self.落(runs, "advise-2026-09-29-093011.jsonl",
                    json.dumps({"t": "2026-09-29T09:30:11.5", "error": "标定失败"}) + "\n")
            self.assertEqual(latest_log(runs).name, "advise-2026-09-28-183523.jsonl")

    def test_目录不存在时返回None(self):
        with TemporaryDirectory() as d:
            self.assertIsNone(latest_log(Path(d) / "没有这个目录"))


class 收官判定(unittest.TestCase):
    def test_我方五连判胜(self):
        盘 = board_of(*[(x, 7, BLACK) for x in range(9, 14)], (0, 0, WHITE), (1, 0, WHITE))
        self.assertEqual(result_of([盘], BLACK), "胜")
        self.assertEqual(result_of([盘], WHITE), "负")   # 同一盘换个人看就是对方已成五

    def test_对方五连判负(self):
        # 白 C8..G8 五连；黑三颗子刻意全避开第 8 行，否则会把对方那条链打断
        盘 = board_of((2, 7, WHITE), (3, 7, WHITE), (4, 7, WHITE), (5, 7, WHITE), (6, 7, WHITE),
                      (10, 10, BLACK), (11, 10, BLACK), (14, 14, BLACK))
        self.assertEqual(result_of([盘], BLACK), "负")
        self.assertEqual(result_of([盘], WHITE), "胜")

    def test_都没五连就是未完(self):
        self.assertEqual(result_of([开局], BLACK), "未完")
        self.assertEqual(result_of([], BLACK), "未完")


class 经验文档(unittest.TestCase):
    条 = Lesson(at="2026-09-28 18:35:23", result="胜", mine="黑", moves=14,
               dev="0 处（全程照引擎首选落）", why="关键在 I7 占到必堵点。")

    def test_写完能原样读回(self):
        with TemporaryDirectory() as d:
            p = Path(d) / "sub" / "lessons.md"          # 父目录不存在也要建
            append(p, self.条)
            回 = load(p)
            self.assertEqual(回, [self.条])

    def test_追写不覆盖旧条目(self):
        with TemporaryDirectory() as d:
            p = Path(d) / "l.md"
            append(p, self.条)
            append(p, Lesson(at="2026-09-29 09:00:00", result="负", mine="白", moves=20,
                             dev="1 处（D5≠E6）", why="漏堵。"))
            回 = load(p)
            self.assertEqual([x.result for x in 回], ["胜", "负"])

    def test_多行原因压成一行(self):
        with TemporaryDirectory() as d:
            p = Path(d) / "l.md"
            append(p, Lesson(at="2026-09-28 18:35:23", result="胜", mine="黑", moves=1,
                             dev="0", why="第一行\n   第二行  带空格的"))
            self.assertEqual(load(p)[0].why, "第一行 第二行 带空格的")

    def test_按对局时间排而不是写入顺序(self):
        """重放旧日志会追写到文件末尾，喂回讲解的必须是最近几局。"""
        with TemporaryDirectory() as d:
            p = Path(d) / "l.md"
            append(p, Lesson(at="2026-09-28 16:52:42", result="负", mine="黑", moves=22,
                             dev="1", why="早该局"))
            append(p, Lesson(at="2026-09-29 09:00:00", result="胜", mine="白", moves=11,
                             dev="0", why="后该局"))
            self.assertEqual([x.why for x in load(p)], ["早该局", "后该局"])
            self.assertIn("后该局", for_prompt(load(p), limit=1))

    def test_手写笔记不打崩解析(self):
        with TemporaryDirectory() as d:
            p = Path(d) / "l.md"
            p.write_text("# 标题\n\n我自己记的一笔，格式不认识\n\n"
                         "## 2026-09-28 18:35:23 · 胜 · 执黑 · 14手\n- 偏离: 0\n- 原因: ok\n",
                         encoding="utf-8")
            回 = load(p)
            self.assertEqual(len(回), 1)
            self.assertEqual(回[0].why, "ok")

    def test_没有原因的小节不算一条(self):
        with TemporaryDirectory() as d:
            p = Path(d) / "l.md"
            p.write_text("## 2026-09-28 18:35:23 · 胜 · 执黑 · 14手\n- 偏离: 0\n", encoding="utf-8")
            self.assertEqual(load(p), [])

    def test_文件不存在时读到空表(self):
        with TemporaryDirectory() as d:
            self.assertEqual(load(Path(d) / "没有这个文件.md"), [])


class 喂回讲解(unittest.TestCase):
    条 = [Lesson(at=f"2026-09-2{i} 18:35:23", result="胜", mine="黑", moves=10 + i,
                 dev="0", why=f"第 {i} 条") for i in (1, 2, 3, 4)]

    def test_没有经验时返回空串(self):
        self.assertEqual(for_prompt([]), "")

    def test_只带最近几条(self):
        段 = for_prompt(self.条, limit=2)
        self.assertIn("第 4 条", 段)
        self.assertIn("第 3 条", 段)
        self.assertNotIn("第 2 条", 段)
        self.assertNotIn("第 1 条", 段)

    def test_护栏必须明说不是本局依据(self):
        """读回来的是一局的总结，不写清这句，模型会拿它解释当前这一手。"""
        段 = for_prompt(self.条)
        self.assertIn("不是本局的事实", 段)
        self.assertIn("禁止拿它们推断本局的棋型、评分或落点", 段)


class 标题时间(unittest.TestCase):
    def test_有日志时间戳就补日期(self):
        self.assertEqual(default_at("18:35:23.689", "2026-09-28"), "2026-09-28 18:35:23")

    def test_JSONL自带日期就不必再说一遍(self):
        """advise 现在默认落 JSONL，重放几天前的那一局若不读日志里的日期就会记错日子。"""
        self.assertEqual(default_at("2026-09-28T18:35:23.689", ""), "2026-09-28 18:35:23")

    def test_显式给的日期压过日志自带的(self):
        self.assertEqual(default_at("2026-09-28T18:35:23.689", "2026-09-11"),
                         "2026-09-11 18:35:23")

    def test_没有日期用今天没有时刻也用现在(self):
        串 = default_at("", "")
        self.assertRegex(串, r"^\d{4}-\d\d-\d\d \d\d:\d\d:\d\d$")


if __name__ == "__main__":
    unittest.main()
