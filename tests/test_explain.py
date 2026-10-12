"""讲解层测试：全部离线，靠一个本地假 HTTP 服务把降级分支跑真。

契约有两半。一半是 prompt：模型禁止自行判断局面，所以棋型、坐标、评分必须由
确定性代码给全，且不能出现会被它按国际象棋惯例读反的「上/下」。另一半是链路：
超时、鉴权失败、结构不认识都必须降级成一行提示，且**绝不把密钥漏进异常信息**。
"""

from __future__ import annotations

import contextlib
import json
import os
import threading
import time
import unittest
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory

from gomoku import explain
from gomoku.board import BLACK, Change, EMPTY, WHITE
from gomoku.config import load
from gomoku.engine import MATE, Plan
from gomoku.explain import (ExplainUnavailable, Explainer, build_prompt, build_review_prompt,
                            format_eval, read_key, request_chat)
from gomoku.lessons import Game, Lesson, Turn, for_prompt
from gomoku.state import empty_board

CFG = load()
SIZE = CFG.board.size
SECRET = "sk-test-DO-NOT-LEAK-0123456789"


def patch_env_file(case: unittest.TestCase, path: Path) -> None:
    """把密钥文件指到临时目录，用例结束自动复原模块全局。"""
    original = explain.ENV_FILE
    explain.ENV_FILE = path
    case.addCleanup(setattr, explain, "ENV_FILE", original)


@contextlib.contextmanager
def 清掉环境变量(name: str):
    """本机可能已有同名变量，用例不能受它影响，退出时原样还原。"""
    had, old = name in os.environ, os.environ.pop(name, None)
    try:
        yield
    finally:
        if had:
            os.environ[name] = old


def board_of(*stones):
    return empty_board(SIZE).with_changes([Change(x, y, EMPTY, c) for x, y, c in stones])


PLAN = Plan(move=(7, 7), score=445.0, depth=21, pv=("H8", "G7", "I9"), elapsed_ms=1500.0)


class 假服务(unittest.TestCase):
    """一个端点三种行为：正常回文本 / 401 / 拖到超时。"""

    class Handler(BaseHTTPRequestHandler):
        mode = "ok"

        def do_POST(self):  # noqa: N802
            body = self.rfile.read(int(self.headers["Content-Length"]))
            type(self).seen = {"path": self.path, "auth": self.headers.get("Authorization"),
                               "body": json.loads(body)}
            if self.mode == "slow":
                time.sleep(2.0)
            if self.mode == "401":
                payload = json.dumps({"error": {"message": "Invalid Authentication api key"}}).encode()
                self.send_response(401)
            elif self.mode == "junk":
                payload = b"<html>not json at all</html>"
                self.send_response(200)
            elif self.mode == "empty":       # reasoning 吃光预算时 DeepSeek 就这么回
                payload = json.dumps({"choices": [{"message": {"content": "   "}}]}).encode()
                self.send_response(200)
            else:
                payload = json.dumps(
                    {"choices": [{"message": {"content": "  这一步冲四，对手必须应。  "}}]}).encode()
                self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *a):
            pass

    def setUp(self):
        self.Handler.seen = None
        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), self.Handler)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.cfg = replace(CFG, llm=replace(CFG.llm, base_url=f"http://127.0.0.1:{self.srv.server_address[1]}"))

    def tearDown(self):
        self.srv.shutdown()
        self.srv.server_close()

    # ---------- 请求与响应 ----------

    def test_正常回话并把首尾空白去掉(self):
        self.Handler.mode = "ok"
        self.assertEqual(request_chat(self.cfg, SECRET, "p"), "这一步冲四，对手必须应。")
        seen = self.Handler.seen
        self.assertEqual(seen["path"], "/chat/completions")
        self.assertEqual(seen["auth"], f"Bearer {SECRET}")     # 密钥确实带上了
        self.assertEqual(seen["body"]["model"], CFG.llm.model)  # 模型 id 走配置，没写死

    def test_必须关掉reasoning否则回话是空的(self):
        """真机实测：v4 系不关推理时 max_tokens 会被 reasoning_content 整预算吃光。"""
        self.Handler.mode = "ok"
        request_chat(self.cfg, SECRET, "p")
        self.assertEqual(self.Handler.seen["body"]["thinking"], {"type": "disabled"})

    def test_回话全空白也算降级(self):
        self.Handler.mode = "empty"
        with self.assertRaises(ExplainUnavailable) as cm:
            request_chat(self.cfg, SECRET, "p")
        self.assertIn("回话为空", str(cm.exception))

    def test_鉴权失败降级且不漏密钥(self):
        self.Handler.mode = "401"
        with self.assertRaises(ExplainUnavailable) as cm:
            request_chat(self.cfg, SECRET, "p")
        note = str(cm.exception)
        self.assertIn("HTTP 401", note)
        self.assertNotIn(SECRET, note)
        self.assertNotIn("Authorization", note)

    def test_响应结构不认识时降级(self):
        self.Handler.mode = "junk"
        with self.assertRaises(ExplainUnavailable):
            request_chat(self.cfg, SECRET, "p")

    def test_超时降级(self):
        self.Handler.mode = "slow"
        cfg = replace(self.cfg, llm=replace(self.cfg.llm, timeout_ms=250))
        t0 = time.perf_counter()
        with self.assertRaises(ExplainUnavailable) as cm:
            request_chat(cfg, SECRET, "p")
        self.assertLess(time.perf_counter() - t0, 1.5, "没按 timeout_ms 收口，会拖死后台线程")
        self.assertNotIn(SECRET, str(cm.exception))

    # ---------- 异步与丢弃 ----------

    def test_不可用时一行提示且不发帖(self):
        got: list[str] = []
        e = Explainer(CFG, on_text=got.append, key=None, enabled=False)
        e.submit("p")
        self.assertEqual(got, ["讲解已关闭"])
        self.assertFalse(e.available)

    def test_缺密钥时降级不抛异常(self):
        got: list[str] = []
        with TemporaryDirectory() as d:
            patch_env_file(self, Path(d) / "nope.env")
            e = Explainer(replace(CFG, llm=replace(CFG.llm, api_key_env="DEFINITELY_NOT_SET_XYZ")),
                          on_text=got.append)
            e.submit("p")
        self.assertEqual(got, ["讲解降级: 缺密钥"])

    def test_排队中的旧请求被新请求挤掉(self):
        """submit 在投递前先把队列清空，所以一次连投只有最后一问会发出去。"""
        self.Handler.mode = "ok"
        got: list[str] = []
        e = Explainer(self.cfg, on_text=got.append, key=SECRET)
        e._jobs.put((1, "旧的"))                     # 先塞一个还没被消费的旧请求
        e._gen = 1
        e.submit("新的")
        deadline = time.time() + 5
        while not got and time.time() < deadline:
            time.sleep(0.05)
        e.close()
        self.assertEqual(len(got), 1, got)
        self.assertIn("冲四", got[0])

    def test_服务挂了也只降级不抛给主循环(self):
        self.Handler.mode = "slow"
        got: list[str] = []
        cfg = replace(self.cfg, llm=replace(self.cfg.llm, timeout_ms=200))
        e = Explainer(cfg, on_text=got.append, key=SECRET)
        e.submit("p")
        deadline = time.time() + 5
        while not got and time.time() < deadline:
            time.sleep(0.05)
        e.close()
        self.assertEqual(len(got), 1)
        self.assertIn("讲解降级", got[0])

    def test_回调形式的拼装发生在本层线程(self):
        """submit 收零参回调时，拼装要到讲解线程里才做。

        回归的是引擎 worker 的时间账：`build_prompt` 实测 148ms，调用方是 worker 线程，
        在那儿拼就等于把下一次提问往后推 148ms。
        """
        self.Handler.mode = "ok"
        got: list[str] = []
        e = Explainer(self.cfg, on_text=got.append, key=SECRET)
        拼过: list[str] = []

        def 拼装():
            拼过.append(threading.current_thread().name)
            time.sleep(0.4)                           # 冒充 build_prompt 那 148ms，放大到能测
            return "冲四"

        t0 = time.perf_counter()
        e.submit(拼装)
        cost = time.perf_counter() - t0
        deadline = time.time() + 5
        while not got and time.time() < deadline:
            time.sleep(0.05)
        e.close()
        self.assertLess(cost, 0.1, f"submit 被拼装拖了 {cost * 1000:.0f}ms，还是在调用线程上拼的")
        self.assertEqual(len(拼过), 1)
        self.assertNotEqual(拼过[0], threading.current_thread().name)
        self.assertIn("冲四", got[0])


class Prompt契约(unittest.TestCase):
    def test_棋型与引擎字段都进了prompt(self):
        b = board_of((5, 5, BLACK), (6, 5, BLACK), (7, 5, BLACK), (8, 5, BLACK))
        p = build_prompt(b, PLAN, BLACK)
        self.assertIn("黑方：活四，走 E6 J6", p)
        self.assertIn("建议 H8", p)
        self.assertIn("以走子方为正的评分 +445", p)
        self.assertIn("H8 G7 I9", p)
        self.assertIn("F6 G6 H6 I6", p)                 # 黑子逐点列出，模型不用自己数

    def test_禁止自行判断局面的措辞在位(self):
        p = build_prompt(board_of((0, 0, BLACK)), PLAN, BLACK)
        self.assertIn("你不得自行判断局面", p)
        self.assertIn("原样照抄", p)

    def test_行号方向说死(self):
        """A/B 实测模型按国际象棋惯例把行号大读成上方，且是系统性一致的错。"""
        p = build_prompt(board_of((0, 0, BLACK)), PLAN, BLACK)
        self.assertIn("行号沿屏幕向下增大", p)
        self.assertIn("行号增大方向", p)

    def test_无棋型时也给出明确的一句(self):
        p = build_prompt(empty_board(SIZE), PLAN, BLACK)
        self.assertIn("双方均无成活、成四或成三的棋型", p)

    def test_走之前与走之后的棋型都给出(self):
        """只给「走之前」却问「这一步属于什么性质」，模型就只能凭空造一条。"""
        b = board_of((5, 5, BLACK), (6, 5, BLACK), (7, 5, BLACK))
        p = build_prompt(b, Plan(move=(8, 5)), BLACK)
        before, _, after = p.partition("【走 I6 之后的盘面】")
        self.assertIn("棋型: 黑方：活三，走 E6 I6 任一点可成活四", before)
        self.assertIn("黑方：活四，走 E6 J6 任一点即成五", after)
        self.assertIn("引擎未给评分", after)

    def test_执白时措辞跟着换(self):
        p = build_prompt(board_of((0, 0, BLACK)), PLAN, WHITE)
        self.assertIn("轮到 白 走", p)

    def test_有变化线时应手以pv第二步为据(self):
        p = build_prompt(board_of((0, 0, BLACK)), PLAN, BLACK)
        self.assertIn("照抄主要变化线里 G7", p)
        self.assertNotIn("不要猜测对手会应哪一手", p)

    def test_无变化线时明确禁止猜应手(self):
        """实测：一手制胜那类回手不带 pv，模型就把对手自己的棋型落点当成防点说出来。"""
        p = build_prompt(board_of((0, 0, BLACK)), replace(PLAN, pv=(), score=None), BLACK)
        self.assertIn("不要猜测对手会应哪一手", p)
        self.assertIn("未给出变化线。", p)
        self.assertNotIn("照抄主要变化线里", p)

    def test_只回一手时两段口径一致(self):
        """09-28 实盘出现过 pv=("J4",)：那时一段印「主要变化线: J4。」另一段说没给变化线，
        模型拿到的是自相矛盾的指令。判据只能有一处。"""
        p = build_prompt(board_of((0, 0, BLACK)), replace(PLAN, pv=("J4",)), BLACK)
        self.assertNotIn("主要变化线:", p)
        self.assertIn("未给出变化线。", p)
        self.assertIn("不要猜测对手会应哪一手", p)

    def test_不走它会怎样这一句有槽位(self):
        """真调用 2/2 都只讲了「走之后」，因为任务里压根没给「不走」留位置。"""
        p = build_prompt(board_of((0, 0, BLACK)), PLAN, BLACK)
        self.assertIn("不走这一点会怎样", p)
        self.assertIn("对手当下没有成型的先手", p)      # 没威胁时的照说话术也得给死
        self.assertIn("不得编造上面没列出的威胁", p)


class 分数文字化(unittest.TestCase):
    def test_三档都成句(self):
        self.assertEqual(format_eval(None), "引擎未给评分（实测出现在一步制胜或开局库命中时）")
        self.assertEqual(format_eval(MATE - 3), "引擎判走子方必胜，其搜索内已证再 3 手成五")
        self.assertEqual(format_eval(-(MATE - 14)), "引擎判对手必胜，其搜索内已证再 14 手成五")
        self.assertEqual(format_eval(445), "以走子方为正的评分 +445")


class 密钥读取(unittest.TestCase):
    def test_环境变量优先其次env文件(self):
        with TemporaryDirectory() as d:
            patch_env_file(self, Path(d) / ".env")
            explain.ENV_FILE.write_text(
                "# 注释\nOTHER=1\nDEEPSEEK_API_KEY = \"sk-from-file\" \n", encoding="utf-8")
            with 清掉环境变量("DEEPSEEK_API_KEY"):
                self.assertEqual(read_key("DEEPSEEK_API_KEY"), "sk-from-file")
                self.assertIsNone(read_key("ABSENT_XYZ"))
                os.environ["DEEPSEEK_API_KEY"] = "sk-from-env"
                self.assertEqual(read_key("DEEPSEEK_API_KEY"), "sk-from-env")

    def test_env文件里没有这一项就返回None(self):
        with TemporaryDirectory() as d:
            patch_env_file(self, Path(d) / ".env")
            explain.ENV_FILE.write_text("FOO=bar\n", encoding="utf-8")
            with 清掉环境变量("DEEPSEEK_API_KEY"):
                self.assertIsNone(read_key("DEEPSEEK_API_KEY"))


class 经验段(unittest.TestCase):
    """读回来的历史经验只进讲解，且必须带着「不是本局依据」那句护栏。"""

    def test_给了经验就拼进prompt(self):
        段 = for_prompt([Lesson(at="2026-09-28 16:52:42", result="负", mine="黑", moves=22,
                                dev="2 处（J5≠J4）", why="白活三先占它的落点")])
        p = build_prompt(board_of((0, 0, BLACK)), PLAN, BLACK, 段)
        self.assertIn("白活三先占它的落点", p)
        self.assertIn("不是本局的事实", p)
        self.assertLess(p.index("【历史经验"), p.index("【任务】"),
                        "经验要在任务之前，否则模型会拿它当结论")

    def test_没给经验时一个字都不出现(self):
        p = build_prompt(board_of((0, 0, BLACK)), PLAN, BLACK)
        self.assertNotIn("历史经验", p)


class 复盘prompt(unittest.TestCase):
    """复盘写出来的那句要进文档、下次再喂回模型，所以事实边界比讲解层更要紧。"""

    def 局(self, *turns):
        return Game(mine=BLACK, result="负", moves=2, turns=list(turns))

    def test_逐手事实与偏离计数都进了prompt(self):
        p = build_review_prompt(self.局(
            Turn(at="16:52:34", pick="J4", play="J5", must="J4", note="J5 没处理对方先手")))
        self.assertIn("我落 J5，引擎首选 J4（偏离）", p)
        self.assertIn("偏离引擎首选 1 处：J5(首选 J4)", p)
        self.assertIn("该堵没堵 1 处：J5", p)
        self.assertIn("J5 没处理对方先手", p)

    def test_不许出现评分与深度(self):
        """09-28 实盘一手就能成五的点被判成 -999982，那种数字一进复盘就写出反向教训。"""
        p = build_review_prompt(self.局(Turn(at="t", pick="F5", play="F5", elapsed_ms=1234.0)))
        self.assertIn("不得引用评分、搜索深度或变化线", p)
        self.assertNotIn("1234", p, "耗时也不给：它只会让模型写「想得久的更可信」")
        self.assertNotIn("depth", p)
        self.assertNotIn("必胜", p)
        self.assertNotIn("必败", p)

    def test_没有可对照的手时也要说清(self):
        p = build_review_prompt(self.局())
        self.assertIn("没有一手参与对照", p)
        self.assertIn("偏离引擎首选 0 处", p)

    def test_禁止编坐标与给新落点(self):
        p = build_review_prompt(self.局(Turn(at="t", pick="F5", play="F5")))
        self.assertIn("不得推算新坐标", p)
        self.assertIn("不得提出新的落点建议", p)


if __name__ == "__main__":
    unittest.main()
