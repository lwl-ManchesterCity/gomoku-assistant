"""建议层（advise）：把引擎问答从采集线程里挪出来。

动机是 2026-09-11 实盘首局（20 手，`/tmp/advise_live2.log`）测出来的唯一真缺陷：
`on_tick` 直接在采集线程里同步调 `engine.ask()`，冷灌一手要 1.3–1.5s，于是 0.3s 的节奏
被拖成 ~1.7s，每一拍都必然把「我方落子 + 对手应手」读成两手（10/10 拍如此，
单手的拍一次都没出现过），消抖的 `confirm_ticks` 因此也测不到。

六条约束：

1. **引擎子进程只能有一个使用者**。Gomocup 协议是一发一收的，两个线程同时喂会互相
   吃掉对方的回手，所以问答全在唯一的 worker 线程里做，采集线程只负责递交局面。
2. **最新者胜**。worker 忙时新盘面直接覆盖待办槽位，不排队 —— 过期盘面的建议没有意义。
   问答返回时若盘面又被挤掉，就只丢弃这条建议：每一问都是整盘重灌，下一次提问本来就
   按新盘面重算，浪费的只是一次问答的时间。
3. **引擎异常不许把采集带崩**。异常只回一行提示，并按第 6 条自愈一次。
4. **问哪个局面由用户声明的颜色定，不从石子奇偶反推「用户是谁」**。两局实盘都是同一个
   形状：你的落子和人机的应手挤在同一拍，于是提交完的盘面永远是「黑=白+1」判轮到对手，
   直接拿它去问，引擎回的是对手的一手（`question_position` 负责把这种局面撤掉用户刚落的
   那颗子再问）。早期之所以**每一拍**都撞上，是我们自己的实心度判据把人机的子整拍挡掉；
   判据 2026-09-11 已删，现在只剩落子间隔短于两拍时才会退成这种形状。
5. **撤过手的问题要把撤掉的那颗子带到出口**（`submit(..., rolled)` → `on_plan` 第四个参数）。
   撤过手的问题属于「用户点下去之前」的那一盘，而打印时的盘面已经含着那颗子了；第二局实盘
   就是这里读坏了：引擎对撤手盘回了 J9，屏幕上 J9 已经是用户的子，用户看到的是
   「你的下一手 J9」＝「它把我的子当成对手落的，又认定我是后手」。
6. **引擎失稳自己修一次**。`board` 重灌把引擎打进 ERROR 或空转时，旧进程的搜索树已经不可信，
   而它不认 `NEW GAME`（实测三条清树命令全被拒），所以唯一的干净办法是换进程 —— 重启后把
   同一问补回去，用户不用等下一手才有建议。重启次数设上限（`RESTART_CAP`），超了就只报一行：
   几轮重启都救不回来，说明是那个盘面本身把引擎问崩了，再重启就是每拍起一个进程空转。

形状照搬讲解层（`explain.Explainer`），因为那套「后台单线程 + 过期即丢」已实盘验过。
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

from .board import BLACK, WHITE, Board, Change, EMPTY
from .engine import EngineError, Plan, Rapfi
from .pattern import five_on_board

RESTART_CAP = 3        # 连续自愈重启的次数上限，问成功一次就归零


@dataclass
class Endgame:
    """终局闸：已经分出胜负的那一局不再送引擎。

    必须 latch 住，不能逐拍现判。2026-09-11 实盘：App 让获胜那五子整排闪烁（~2Hz，
    每个态持续 1–3 拍），闪灭那一拍凭空少 5 子 —— 按当拍盘面会判成「还没结束」，
    于是拿着 黑10白14 这种不可能的形状去问引擎，在用户已经赢了的盘上印出「必败」。
    只有盘面比收局面还小（= 重开了一局）才放开闸。
    """

    stones: int | None = None        # 结局面上的子数

    def admit(self, board: Board) -> tuple[bool, str | None]:
        """返回 (这一拍该不该问引擎, 刚判出终局时的胜方颜色码 —— 只在第一拍给一次)。

        放开闸的条件是「子数比收官时少」且「形状仍然合法」（黑先手 ⇒ 黑=白 或 黑=白+1）。
        光看子数变小不行：闪灭那一拍正是少 5 子的盘，而 5 子同色拿掉之后形状必然非法
        （黑10白14 这种），所以非法形状只能是被闪掉的旧局面，重开一局和悔棋都保持合法。
        """
        counts = board.counts()
        black, white = counts[BLACK], counts[WHITE]
        total = black + white
        if self.stones is not None and total < self.stones and black in (white, white + 1):
            self.stones = None
        ended: str | None = None
        for colour in (BLACK, WHITE):
            if five_on_board(board, colour):
                if self.stones is None:
                    self.stones = total
                    ended = colour
                break
        return self.stones is None, ended


def worth_asking(board: Board, changes, prev: Board | None) -> bool:
    """这一拍值不值得问引擎。

    有新确认的落子就问；没有新落子但看到的是**刚锁定**的那个局面也问 —— `prev is None`
    是进程刚启动或标定失败后重新锁定，`prev != board` 是基准盘整盘被换掉（重同步）。
    这个时机已栽两次：吞掉空盘那一拍 ⇒ 先手的第一手只能出「复核」；吞掉中途启动时已轮到
    我方的基准盘 ⇒ 全程零建议。轮不到我方时 `question_position` 会自己拦掉，多问一次无害。
    """
    return bool(changes) or prev is None or prev != board


def question_position(board: Board, changes, mine: str) -> tuple[Board | None, Change | None]:
    """这一拍该拿哪个局面去问引擎。返回 (要问的盘面, 被撤掉的那颗子)。

    你的落子和人机的应手挤在同一拍时，提交完的盘面是「黑=白+1」，`side_to_move()` 判轮到
    对手 —— 那个盘面上引擎回答的是**对手要下哪儿**，不是给用户的一手。把用户这一拍刚落的
    子撤掉，才回到用户当时面对的那个局面。

    第二个返回值非空时，它带着「这条建议归属到哪一手」的信息（`Advisor.submit` 一路传到出口），
    撤过手的那一盘和当前盘面差着这颗子，不说清楚就会指着用户自己占的点让他再下一遍。
    """
    if board.side_to_move() == mine:
        return board, None
    mine_new = [ch for ch in changes if ch.after == mine]
    if len(mine_new) != 1:
        return None, None           # 说不清哪颗是用户刚落的，撤出来的盘面不可信
    reverted = board.with_changes([Change(ch.x, ch.y, mine, EMPTY) for ch in mine_new])
    return reverted, mine_new[0]


@dataclass
class Advisor:
    """后台单线程问答：最新盘面挤掉待办的旧盘面，在途答案过期就丢。"""

    engine: Rapfi
    on_plan: object = None        # callable(board, plan, mover, rolled) -> None
    on_note: object = None        # callable(str) -> None
    _pending: "tuple[Board, Change | None] | None" = field(default=None, repr=False)
    _gen: int = field(default=0, repr=False)
    _cv: threading.Condition = field(default=None, repr=False)
    _thread: threading.Thread | None = field(default=None, repr=False)
    _closed: bool = field(default=False, repr=False)
    _restarts: int = field(default=0, repr=False)   # 连续自愈重启的次数，问成功即归零

    def __post_init__(self):
        if self._cv is None:
            self._cv = threading.Condition()

    @property
    def alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def submit(self, board: Board, rolled: Change | None = None) -> None:
        """递交要问的盘面并立刻返回。排队中的旧盘面一律作废。

        `rolled` = 问之前从盘面上撤掉的那颗子，即用户这一拍刚落的一手。撤过手的问题是
        「用户点下去之前」的那一盘，出口必须带着它才说得出这条建议归属哪一盘 ——
        否则会把用户刚落的子当成空位再推荐一遍。
        """
        with self._cv:
            self._gen += 1
            self._pending = (board, rolled)
            self._cv.notify_all()
            if not self.alive and not self._closed:
                self._thread = threading.Thread(target=self._work, daemon=True)
                self._thread.start()

    def _work(self) -> None:
        while True:
            with self._cv:
                while self._pending is None and not self._closed:
                    self._cv.wait(0.5)
                if self._closed:
                    return
                job, gen = self._pending, self._gen
                self._pending = None
            board, rolled = job
            plan = self._ask(board)
            if plan is None:
                continue
            with self._cv:
                stale = gen != self._gen
            if stale:
                self._note("建议已过期（盘面又变了），丢弃")
                continue
            if self.on_plan:
                self.on_plan(board, plan, self.engine.engine_colour, rolled)

    def _ask(self, board: Board) -> Plan | None:
        """问一次；引擎失稳就换进程把同一问补回来，最多一次。"""
        for attempt in (1, 2):
            try:
                plan = self.engine.ask(board)
            except EngineError as exc:
                err = str(exc)
            except Exception as exc:      # 兜底：适配层没预料到的错也不能把 worker 带走
                err = type(exc).__name__
            else:
                self._restarts = 0
                return plan
            if attempt == 2 or self._restarts >= RESTART_CAP:
                self._note(f"引擎异常: {err} —— 采集继续，推荐要重启进程才有")
                return None
            self._restarts += 1
            self._note(f"引擎异常: {err} —— 重启引擎，这一问补回来")
            try:
                self.engine.restart()
            except Exception as exc:
                self._note(f"引擎重启失败: {type(exc).__name__} —— 采集继续，推荐要重启进程才有")
                return None
        return None

    def _note(self, text: str) -> None:
        if not self._closed and self.on_note:
            self.on_note(text)

    def close(self, join_s: float = 2.0) -> None:
        """不再采纳任何在途结果；worker 正卡在引擎回答里时不等它。"""
        with self._cv:
            self._closed = True
            self._pending = None
            self._gen += 1
            self._cv.notify_all()
        thread, self._thread = self._thread, None
        if thread and thread.is_alive():
            thread.join(timeout=join_s)
