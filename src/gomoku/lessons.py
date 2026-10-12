"""经验层（lessons）：一局打完把胜/负的原因写进一份持久文档，下次启动读回来。

两份职责，边界刻意切开：

1. **事实由确定性代码算**（`read_frames` / `round_facts`）。从日志里把逐拍盘面读回来，
   对每一手「我方实际落的点」和「引擎在该局面下的首选」做对照，再叠上棋型层已经算好的
   必堵点与安全网判词。这一半不看评分。
2. **原因才交给模型**（`explain.build_review_prompt`）。它拿到的只有第 1 半算好的事实。

为什么不参与选点：文档里那句「原因」是模型写的散文，不是一条判据 —— 让它能改落点，
等于把一次措辞失误永久装进决策链。而且「引擎首选」这件事本身随预算档漂：
`tools/probe_review_budget.py` 在 09-28 两份日志上并排跑了 1500ms（实盘档）和 6000ms
（review 档）两遍，胜 21 手那局偏离计数 6 → 5，负 22 手那局计数同为 3 但有 2 手的首选
换了点（16:50:12 的 K5 与 16:50:30 的 H5 在两档下各自「一致」一次）。要往选点走，
得先固定一档预算、在历史日志上离线重放过、拿「胜负 + 该堵没堵处数」判过「不更差」。
评分的**符号**这一头倒是已经不含糊：`tools/probe_eval_side.py` 证过 Eval 跟着走子方翻、
不跟着颜色翻（两局「走子方被双四杀掉」的形状，黑白各一次，都回 -999998），
所以旧版那句「杀棋分会反」是误读 —— 那次 -999982 是引擎在说走子方真的要被杀。

日志有两个来源，都得能吃：`--log` 写的 JSONL（每拍一行含 `grid`），以及用户直接
从终端粘出来的整屏日志（`A B C …` 表头 + 15 行）。09-28 那三份实盘样本就是后者。
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .board import BLACK, EMPTY, WHITE, Board, Change
from .engine import Plan
from .pattern import answer_points, five_on_board, safety_note

DEFAULT_DOC = Path("/Users/lwl/Documents/Qoder/2026-09-10/gomoku-lessons.md")
TITLE = "# Gomoku 盯屏助手 · 复盘经验"

WIN = "胜"
LOSE = "负"
UNFINISHED = "未完"

WHO = {BLACK: "黑", WHITE: "白"}
TS_RE = re.compile(r"\[(\d\d:\d\d:\d\d\.\d\d\d)\]")
HEADER_RE = re.compile(r"^## (\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) · (胜|负|未完) · 执(.) · (\d+)手$")
FIELD_RE = re.compile(r"^- (偏离|原因): (.*)$")


def other(colour: str) -> str:
    return WHITE if colour == BLACK else BLACK


def xy(name: str) -> tuple[int, int]:
    """坐标名 → (x, y)：与 `tests` 里同样的写法，集中一处免得每处各算一遍。"""
    return ord(name[0]) - 65, int(name[1:]) - 1


# ---------------------------------------------------------------- 日志 → 逐拍盘面

def _board_of_rows(rows: Sequence[str]) -> Board | None:
    """把 15 个等长字符串拼成盘面；形状不对就返回 None，不猜。"""
    n = len(rows)
    if n < 5 or any(len(r) != n for r in rows):
        return None
    if any(ch not in (EMPTY, BLACK, WHITE) for r in rows for ch in r):
        return None
    return Board(tuple(tuple(r) for r in rows))


def frames_from_jsonl(text: str) -> list[tuple[str, Board]]:
    """`--log` 写的 JSONL：取每行的 `grid`，`t` 当时间戳，报错行自然没有 grid。"""
    out: list[tuple[str, Board]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        rows = rec.get("grid")
        if not isinstance(rows, list):
            continue
        board = _board_of_rows([str(r).strip() for r in rows])
        if board is not None:
            out.append((str(rec.get("t", "")), board))
    return out


def frames_from_text(text: str) -> list[tuple[str, Board]]:
    """终端粘出来的日志：`A B C …` 表头下面紧跟 N 行 `行号 + N 格`。

    时间戳往上找最近的一行 `[HH:MM:SS.mmm]` —— 盘面是在那一拍印出来的，复盘要按它排序。
    """
    lines = text.splitlines()
    out: list[tuple[str, Board]] = []
    for i, line in enumerate(lines):
        head = line.strip()
        if not head.startswith("A B C"):
            continue
        size = len(head.split())
        rows: list[str] = []
        for follow in lines[i + 1:]:
            parts = follow.split()
            if len(parts) != size + 1:
                break
            rows.append(" ".join(parts[1:]).replace(" ", ""))
            if len(rows) == size:
                break
        board = _board_of_rows(rows)
        if board is None:
            continue
        stamp = ""
        for j in range(i, max(-1, i - 30), -1):
            m = TS_RE.search(lines[j])
            if m:
                stamp = m.group(1)
                break
        out.append((stamp, board))
    return out


def latest_log(runs: Path) -> Path | None:
    """最新一份**读得出盘面**的日志；从文件名最新的开始倒着扫，扫到能用的就停。

    两个坑都得挡：
    * 启动即失败那一次会留下 0 字节的壳；
    * 屏上没有棋盘面时跑的那一次每一行都是 `error`，有大小却没有 `grid`。
      按大小挑的话这一份会把真正要复盘的上一局挡掉。
    文件名带定长时间戳，所以字典序就是时间序。
    """
    for p in sorted(Path(runs).glob("*.jsonl"), reverse=True):
        try:
            if read_frames(p):
                return p
        except OSError:
            continue
    return None


def read_frames(path: Path) -> list[tuple[str, Board]]:
    """读一份日志，返回逐拍 (时间戳, 可信盘面)，并去掉连续重复的拍。

    0.3s 一拍里同一盘面会连印好几拍，复盘要的是「局面序列」而不是拍数。
    """
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    first = next((ln for ln in text.splitlines() if ln.strip()), "")
    frames = frames_from_jsonl(text) if first.lstrip().startswith("{") else frames_from_text(text)
    out: list[tuple[str, Board]] = []
    for stamp, board in frames:
        if not out or out[-1][1] != board:
            out.append((stamp, board))
    return out


def result_of(frames: Sequence[Board], mine: str) -> str:
    """收官结果按最后一个盘面判，且必须盘上真有一条五连。

    不按「轮到谁」判：赢的那一拍之后 App 让五连整排闪烁，闪灭那拍少 5 子，
    按奇偶会读成「没结束」。
    """
    if not frames:
        return UNFINISHED
    last = frames[-1]
    if five_on_board(last, mine):
        return WIN
    if five_on_board(last, other(mine)):
        return LOSE
    return UNFINISHED


# ---------------------------------------------------------------- 逐手对照

@dataclass(frozen=True)
class Turn:
    """我方的一手：面对的局面、引擎首选、实际落点，以及棋型层给的两条判词。"""

    at: str
    pick: str                       # 引擎首选坐标名
    play: str                       # 我方实际落点坐标名
    must: str = ""                  # 对方先手的必堵点（空格分隔），无则空串
    note: str = ""                  # 落在实际点上的安全网判词，没毛病则空串
    elapsed_ms: float = 0.0

    @property
    def followed(self) -> bool:
        return self.pick == self.play

    @property
    def blocked(self) -> bool | None:
        """实际落点有没有占到必堵点之一。没有先手要应时给 None，不能算成「漏堵」。"""
        if not self.must:
            return None
        return self.play in self.must.split()


@dataclass
class Game:
    mine: str
    result: str
    moves: int                      # 我方落在盘上的子数
    turns: list[Turn] = field(default_factory=list)
    skipped: int = 0                # 说不清「哪颗是我刚落的」因而没对照的拍

    @property
    def deviations(self) -> list[Turn]:
        return [t for t in self.turns if not t.followed]

    @property
    def misses(self) -> list[Turn]:
        """该堵没堵的那几手 —— 文档里最该被读回来的一条。"""
        return [t for t in self.turns if t.blocked is False]

    def dev_line(self) -> str:
        dev = self.deviations
        if not dev:
            return "0 处（全程照引擎首选落）"
        点 = " ".join(f"{t.play}≠{t.pick}" for t in dev)
        return f"{len(dev)} 处（{点}）"

    def fact_lines(self) -> list[str]:
        """喂给模型的确定性事实。一条一手，模型不用自己数子。"""
        out = []
        for t in self.turns:
            行 = f"{t.at} 我落 {t.play}，引擎首选 {t.pick}" + ("（一致）" if t.followed else "（偏离）")
            if t.must:
                行 += f"；该局面必堵 {t.must}，我{'占到了' if t.blocked else '没占到'}"
            if t.note:
                行 += f"；安全网判词：{t.note}"
            out.append(行)
        return out


def position_before(prev: Board | None, cur: Board, mine: str) -> tuple[Board, str] | None:
    """还原「我方这一手之前面对的局面」并给出那一手的坐标名，说不清就返回 None。

    两种拍形都得处理：只出现我这颗子（对手那手早在一拍前读到了），以及我这手和对手的
    应手挤在同一拍。后者按石子奇偶定先后 —— 上一拍就轮到我，说明我先落，要问的局面是
    上一拍那个盘；上一拍轮不到我，说明对手先落，把盘上我这颗撤掉才是我按下时的局面。
    说不清（同一拍出现两颗我的子、撤完还是轮不到我）就不对照，绝不猜。
    """
    if prev is None:
        return None
    changed = prev.diff(cur)
    mine_new = [ch for ch in changed if ch.after == mine]
    if len(mine_new) != 1:
        return None
    name = mine_new[0].name
    if prev.side_to_move() == mine:
        return prev, name
    撤 = cur.with_changes([Change(ch.x, ch.y, mine, EMPTY) for ch in mine_new])
    return (撤, name) if 撤.side_to_move() == mine else None


def todo_count(frames: Sequence[tuple[str, Board]], mine: str) -> int:
    """这一局会问引擎几手：和 `round_facts` 用同一套「对不出局面就不问」的判据，只是不提问。

    10 分钟一局有 20~30 问、放宽预算后要跑两三分钟，先报总数才知道还要等多久。
    """
    return sum(1 for i in range(1, len(frames))
               if position_before(frames[i - 1][1], frames[i][1], mine))


def round_facts(frames: Sequence[tuple[str, Board]], mine: str,
                ask: Callable[[Board], Plan | None],
                on_progress: Callable[[int], None] | None = None) -> Game:
    """把逐拍盘面走一遍，逐手对照引擎。`ask` 由调用方注入，所以本函数能离线测。

    第一帧没有「上一拍」，那份局面不是我落出来的，直接从跳过之列开始对照。

    `on_progress(已对照手数)` 每成一手续报一次：10 分钟一局有 20~30 问，放宽预算后
    整局要跑两三分钟，不报进度看上去就像卡在引擎里。
    """
    boards = [b for _, b in frames]
    turns: list[Turn] = []
    skipped = 0
    for i in range(1, len(frames)):
        got = position_before(frames[i - 1][1], frames[i][1], mine)
        if got is None:
            if any(ch.after == mine for ch in boards[i - 1].diff(boards[i])):
                skipped += 1               # 有我落的子，但对不出该问哪个盘
            continue
        faced, play = got
        plan = ask(faced)
        if plan is None:
            skipped += 1
            continue
        turns.append(Turn(
            at=frames[i][0], pick=plan.name, play=play,
            must=answer_points(faced, mine) or "",
            note=safety_note(faced, mine, xy(play)) or "",
            elapsed_ms=plan.elapsed_ms))
        if on_progress:
            on_progress(len(turns))
    return Game(mine=mine, result=result_of(boards, mine),
                moves=boards[-1].counts()[mine] if boards else 0,
                turns=turns, skipped=skipped)


# ---------------------------------------------------------------- 经验文档

@dataclass(frozen=True)
class Lesson:
    at: str                         # "2026-09-28 19:11:26"
    result: str                     # 胜 / 负 / 未完
    mine: str                       # 黑 / 白
    moves: int
    dev: str                        # 偏离摘要，来自 Game.dev_line()
    why: str                        # 原因，一行（多行会被写成一行的空格串）

    def block(self) -> str:
        return (f"## {self.at} · {self.result} · 执{self.mine} · {self.moves}手\n"
                f"- 偏离: {self.dev}\n"
                f"- 原因: {_one_line(self.why)}")


def _one_line(text: str) -> str:
    """文档格式是「一条经验两行」，模型爱写换行，这里压成一行。"""
    return " ".join(text.split())


def append(path: Path, lesson: Lesson) -> Path:
    """追写一条经验。文件不存在就先写标题，父目录不存在就建。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    body = f"{lesson.block()}\n\n"
    if p.exists():
        with p.open("a", encoding="utf-8") as fh:
            fh.write(body)
    else:
        p.write_text(f"{TITLE}\n\n{body}", encoding="utf-8")
    return p


def load(path: Path) -> list[Lesson]:
    """读回全部经验，按对局时间正序。读不认识的小节直接跳过，手写的笔记不会把解析打崩。

    排序按标题里的时间而不是文件顺序：重放几天前的旧日志会追写到文件末尾，
    那样「最近几条」（`for_prompt`）就会挑错，而喂回讲解的恰恰只能是最近几局。
    """
    p = Path(path)
    if not p.exists():
        return []
    out: list[Lesson] = []
    head: re.Match | None = None
    fields: dict[str, str] = {}

    def flush() -> None:
        if head and fields.get("原因"):
            out.append(Lesson(at=head.group(1), result=head.group(2), mine=head.group(3),
                              moves=int(head.group(4)), dev=fields.get("偏离", ""),
                              why=fields["原因"]))

    for line in p.read_text(encoding="utf-8").splitlines():
        h = HEADER_RE.match(line.strip())
        if h:
            flush()
            head, fields = h, {}
            continue
        f = FIELD_RE.match(line.strip())
        if f and head:
            fields[f.group(1)] = f.group(2)
    flush()
    return sorted(out, key=lambda x: x.at)


def for_prompt(lessons: Sequence[Lesson], limit: int = 3) -> str:
    """最近 limit 条压成讲解 prompt 里的一段。没有经验时返回空串，调用方据此不拼这一段。

    护栏是必须的：这一段是**上一局**的总结，模型拿到它就会拿它解释本局，
    所以明说「不得据此判断或评价当前落点」。
    """
    if not lessons:
        return ""
    条 = lessons[-limit:]
    body = "\n".join(f"- {x.at} {x.result}（执{x.mine}，偏离{x.dev}）：{x.why}" for x in 条)
    return ("【历史经验（上一局的总结，不是本局的事实）】\n"
            f"{body}\n"
            "这些只作讲解时的背景参考。禁止拿它们推断本局的棋型、评分或落点，"
            "上面【引擎结论】和盘面字段才是本局唯一依据。")


def default_at(stamp: str | None = "", date: str | None = "") -> str:
    """文档标题里的日期时间。`date` 允许是 None —— argparse 没给值时就是这个。

    日期按 `--date` > 日志自带 > 今天 取：JSONL 的 `t` 是完整 ISO 时间戳，重放几天前的
    那份日志不必再说一遍日期；终端粘出来的整屏日志只有时分秒（那一列本来就是
    `[18:35:23.689]`），那种来源才必须 `--date`。
    """
    m = re.search(r"(\d\d:\d\d:\d\d)", stamp or "")
    时刻 = m.group(1) if m else datetime.now().strftime("%H:%M:%S")
    自带 = re.search(r"(\d{4}-\d\d-\d\d)", stamp or "")
    日 = (date or "").strip() or (自带.group(1) if 自带 else datetime.now().strftime("%Y-%m-%d"))
    return f"{日} {时刻}"
