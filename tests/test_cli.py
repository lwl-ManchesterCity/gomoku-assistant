"""命令行这一层的契约：一局日志落在哪、谁来读回来。

`advise` 要起引擎和采集线程，这里不跑它本身，只钉住「默认落盘 → review 自己找得到」
这条链上纯离线的那两头：

* `run_log` 的命名 —— `latest_log` 按文件名字典序取最新，靠的就是时间戳定长写在名字里；
  更要紧的是它**不再默认写 `logs/events.jsonl`**，那份 09-10 的样本是唯一还能重放的旧证据，
  而日志是以 append 打开的。
* `review --note` 这一支既不问引擎也不调模型，所以整条 CLI（读日志 → 判收官 → 追写文档）
  能在这里跑真，只把引擎和密钥绕开。
"""

from __future__ import annotations

import contextlib
import io
import json
import unittest
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory

from gomoku import __main__ as cli
from gomoku.board import BLACK, EMPTY, WHITE, Change
from gomoku.lessons import load
from gomoku.state import empty_board

SIZE = 15


def 一格不空的盘面():
    """黑在 A1..E1 连成五，白四子散落 —— 收官该判「胜」，且黑=5 白=4 形状合法。"""
    子 = [(x, 0, BLACK) for x in range(5)] + [(x, 3, WHITE) for x in range(4)]
    return empty_board(SIZE).with_changes([Change(x, y, EMPTY, c) for x, y, c in 子])


def 写一份日志(path: Path, 时刻: str = "2026-09-28T16:52:42.703") -> Path:
    盘 = 一格不空的盘面()
    rows = ["".join(r) for r in 盘.grid]
    path.write_text(json.dumps({"t": 时刻, "mode": "ROI", "grid": rows}) + "\n",
                    encoding="utf-8")
    return path


class 日志落在哪(unittest.TestCase):
    def test_文件名带定长时间戳所以按名字排就是按时间排(self):
        got = cli.run_log(datetime(2026, 9, 29, 9, 7, 3))
        self.assertEqual(Path(got).name, "advise-2026-09-29-090703.jsonl")
        self.assertEqual(Path(got).parent.name, "runs")

    def test_不带参数就用当下且总是同一个目录(self):
        got = cli.run_log()
        self.assertRegex(Path(got).name, r"^advise-\d{4}-\d\d-\d\d-\d{6}\.jsonl$")
        self.assertEqual(Path(got).parent, cli.RUNS)

    def test_默认路径绝不落在旧样本那个文件上(self):
        """`logs/events.jsonl` 是 09-10 那 85 行唯一还能重放的样本，append 一次就废了。"""
        self.assertNotEqual(Path(cli.run_log()).name, "events.jsonl")


class 复盘命令行(unittest.TestCase):
    def run_cli(self, argv) -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            码 = cli.main(argv)
        return 码, out.getvalue(), err.getvalue()

    def test_note这一支不碰引擎也能写完一条经验(self):
        with TemporaryDirectory() as d:
            日志 = 写一份日志(Path(d) / "run.jsonl")
            doc = Path(d) / "lessons.md"
            码, out, err = self.run_cli(["review", str(日志), "--me", "黑",
                                         "--note", "手写的原因", "--lessons", str(doc)])
            self.assertEqual(码, 0, err)
            条 = load(doc)
            self.assertEqual(len(条), 1)
            self.assertEqual((条[0].result, 条[0].mine, 条[0].moves), ("胜", "黑", 5))
            self.assertEqual(条[0].dev, "未对照（没问引擎）")
            self.assertEqual(条[0].why, "手写的原因")
            self.assertIn("已写进", out)

    def test_标题时间取自日志而不是跑命令的今天(self):
        with TemporaryDirectory() as d:
            日志 = 写一份日志(Path(d) / "run.jsonl", "2026-09-28T16:52:42.703")
            doc = Path(d) / "lessons.md"
            self.run_cli(["review", str(日志), "--me", "白", "--note", "x",
                          "--lessons", str(doc)])
            self.assertEqual(load(doc)[0].at, "2026-09-28 16:52:42")

    def test_不给日志就读runs里最新一份(self):
        with TemporaryDirectory() as d:
            目录 = Path(d) / "runs"
            目录.mkdir()
            写一份日志(目录 / "advise-2026-09-28-165242.jsonl")
            写一份日志(目录 / "advise-2026-09-29-093011.jsonl", "2026-09-29T09:30:11.5")
            旧 = cli.RUNS
            cli.RUNS = 目录
            self.addCleanup(setattr, cli, "RUNS", 旧)
            doc = Path(d) / "lessons.md"
            码, out, err = self.run_cli(["review", "--me", "黑", "--note", "x",
                                         "--lessons", str(doc)])
            self.assertEqual(码, 0, err)
            self.assertIn("advise-2026-09-29-093011.jsonl", out)
            self.assertEqual(load(doc)[0].at, "2026-09-29 09:30:11")

    def test_没有可复盘的日志时报清楚去哪儿找(self):
        with TemporaryDirectory() as d:
            旧 = cli.RUNS
            cli.RUNS = Path(d) / "runs"
            self.addCleanup(setattr, cli, "RUNS", 旧)
            码, out, err = self.run_cli(["review", "--me", "黑", "--note", "x"])
            self.assertEqual(码, 2)
            self.assertIn("没有可复盘的日志", err)

    def test_读不出盘面的日志要报错而不是写一条空经验(self):
        with TemporaryDirectory() as d:
            日志 = Path(d) / "run.jsonl"
            日志.write_text('{"t":"2026-09-29T09:30:11.500","error":"标定失败"}\n',
                            encoding="utf-8")
            码, out, err = self.run_cli(["review", str(日志), "--me", "黑", "--note", "x",
                                         "--lessons", str(Path(d) / "lessons.md")])
            self.assertEqual(码, 2)
            self.assertIn("读不出任何盘面", err)
            self.assertFalse((Path(d) / "lessons.md").exists())


class 帮助文本(unittest.TestCase):
    """`--help` 是用户看出「默认就落盘、review 不必给路径」这两条改变的地方。"""

    def 帮助(self, 子命令: str) -> str:
        """把 --help 里的换行与空格全抹掉再比：argparse 按终端宽度折行，留着空白就没法断句。"""
        out = io.StringIO()
        with self.assertRaises(SystemExit), contextlib.redirect_stdout(out):
            cli.main([子命令, "--help"])
        return "".join(out.getvalue().split())

    def test_advise有开关能关掉日志但默认是写(self):
        帮 = self.帮助("advise")
        self.assertIn("--no-log", 帮)
        self.assertIn("advise-<日期-时间>.jsonl", 帮)

    def test_review的日志路径变成可选(self):
        帮 = self.帮助("review")
        self.assertIn("[log]", 帮, "位置参数一旦必填（usage 里不带方括号），自动找最新一局那条路就等于没接上")
        self.assertIn("--engine-ms", 帮)


if __name__ == "__main__":
    unittest.main()
