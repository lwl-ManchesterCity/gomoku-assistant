"""RapFi 引擎适配层（Gomocup 变体）。

协议是 2026-09-10 用 tools/rapfi_session.py 实测出来的，不是照规范猜的：

* 握手：`START` 只吐 `MESSAGE`，**唯一一个 `OK` 挂在后面的棋盘规格行 `15` 上**（规范说 OK 该回给 START）。
* 坐标 0 基、x=列 y=行，**没有镜像**：pv 里的 `I8` 与裸输出 `8,7` 自洽
  （列 A=0、行标签 = y+1），和 board.coord_name 的约定一模一样，两侧都不必换序。
  出厂 `config.toml` 里 `coord_conversion_mode = "none"` 正是这个含义。
* 回手是**裸坐标** `8,7`，不带规范要求的 `COORDS ` 前缀；中间夹大量进度行
  `MESSAGE Depth 12-24 | Eval 445 | Time 50ms | I7 F7 I8`，顺手把评分和 pv 取出来。
  Eval 以**走子方**为正，杀棋写成 `+M3` / `-M14`。这条 2026-09-29 用
  `tools/probe_eval_side.py` 钉死了：摆两局「握双四的一方必杀先手方」，黑白各一次，
  两次都回 `-999998`（走子方被杀、余 2 手）—— 符号跟着**谁在走**翻，不跟着颜色翻，
  所以 `engine_colour` 按奇偶取「轮到的一方」和引擎自己是同一个约定，出口那句
  「必胜/必败」读法成立。
* **`turn` 增量问答已停用**（2026-09-11 16:07 实盘被秒）：同一个局面，`turn` 回了 E7 并连报
  必胜(余21)→(余11)→(余3)→必败(余2)，而 E7 正是上一轮 `pv=G9 E7 F7 F9 …` 的第二手 ——
  它没重新搜索，把旧 pv 往下播了一格，那个 mate 分是从上一轮继承的幻影；`board` 重灌
  4/4 稳定回 F5（真的成五落点）。所以现在**每一问都整盘重灌**，奇偶仍然自带「引擎执哪方」。
* 整盘重同步：`board` 进入位置编辑态并**清空旧子**，每行 `x,y,color`（color 1=黑 2=白 3=空），
  `done` 结束编辑并立即搜索回手。**引擎执哪一方由石子数奇偶推出**：
  3黑3白时它执黑把 C3-C5 扩成 C2（回 `2,1`），加一颗黑子变 4黑3白后它改执白去扩自己的 K3-K5（回 `10,1`）。
  所以重同步不需要「我方颜色」这个配置项，奇偶本身就带着这个信息。

三条实测拿到的防御性约束：
* `END` 会真的结束进程，编辑态里则被当成一行坐标（报 Coord 错但不退出）。
* **`takeback` / `restart` 会把引擎打进死循环**（第二次 takeback 后连吐 9478 行
  `Unknown command:`，CPU 空转）。这里一律不用回退命令：局面错位靠 `board` 重灌解决，
  且读取侧设洪水上限，超了就判引擎失稳。
* 编辑态里除 `x,y,color` 之外的行一律报 `Coord is not valid or empty`，
  所以拼错一行不会静默丢掉一颗子，会以 ERROR 形式在 `_await_move` 里抛出。
"""

from __future__ import annotations

import queue
import re
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from .board import BLACK, EMPTY, WHITE, Board, Change, coord_name
from .config import Config

ROOT = Path(__file__).resolve().parents[2]
Coord = tuple[int, int]
MOVE_RE = re.compile(r"^(\d{1,2}),(\d{1,2})$")
EVAL_RE = re.compile(r"^([+-]?)(M?)(\d+)$")
FLOOD_LIMIT = 600          # 一次问答允许吞掉的最大行数，超了就是引擎在空转
MATE = 1_000_000           # 杀棋分数量级；|score| >= MATE 表示必胜/必负，余步数 = MATE - |score|
CODES = {BLACK: 1, WHITE: 2}


class EngineError(RuntimeError):
    pass


@dataclass(frozen=True)
class Plan:
    move: Coord
    score: float | None = None
    depth: int | None = None
    pv: tuple[str, ...] = ()
    elapsed_ms: float = 0.0

    @property
    def name(self) -> str:
        return coord_name(*self.move)


def parse_move(line: str) -> Coord | None:
    m = MOVE_RE.match(line.strip())
    return (int(m.group(1)), int(m.group(2))) if m else None


def parse_eval(token: str) -> float | None:
    """`445` → 445.0，`+M3` → MATE-3，`-M14` → -(MATE-14)。"""
    m = EVAL_RE.match(token)
    if not m:
        return None
    value = int(m.group(3)) * (-1 if m.group(1) == "-" else 1)
    return float(value) if not m.group(2) else (MATE - abs(value)) * (1 if value > 0 else -1)


def parse_progress(line: str) -> tuple[int, float, tuple[str, ...]] | None:
    """`MESSAGE Depth 12-24 | Eval 445 | Time 50ms | I7 F7 I8` → (12, 445.0, (I7,F7,I8))。"""
    if "Depth " not in line or " Eval " not in line:
        return None
    depth = score = None
    pv: tuple[str, ...] = ()
    # 首段带 MESSAGE 前缀，去掉后才能按字段名分派，否则 Depth 永远读不到。
    for stripped in line.split("|"):
        chunk = stripped.strip().removeprefix("MESSAGE").strip()
        if chunk.startswith("Depth "):
            depth = int(chunk[6:].split("-")[0])
        elif chunk.startswith("Eval "):
            score = parse_eval(chunk[5:].split()[0])
        elif not chunk.startswith(("Time", "OptiTime", "MaxTime", "Speed", "Node")):
            pv = tuple(t for t in chunk.split() if re.fullmatch(r"[A-Za-z]\d{1,2}", t))
    if depth is None or score is None:
        return None
    return depth, score, pv


def _pump(proc: subprocess.Popen, out: "queue.Queue[str]") -> None:
    for line in iter(proc.stdout.readline, ""):
        out.put(line.rstrip("\n"))
    out.put("")   # 空串 = 进程结束，等待方据此立刻报错而不是干等超时


@dataclass
class Rapfi:
    """常驻一个 RapFi 子进程，每一问都用 `board` 整盘重灌（增量 `turn` 已停用，见模块头）。"""

    cfg: Config
    binary: Path | None = None
    timeout_ms: int | None = None         # 覆盖 [engine].timeout_ms；离线复盘不必守实盘那一档
    model: Board | None = None            # 引擎此刻以为的局面，含它自己刚回的那一手
    engine_colour: str | None = None      # 引擎执哪方 = 盘面上「轮到走」的一方，由最近一次重灌的奇偶定
    proc: subprocess.Popen = field(default=None, repr=False)
    _out: "queue.Queue[str]" = field(default=None, repr=False)
    _budget_s: float = field(default=0.0, repr=False)

    def __post_init__(self):
        path = Path(self.cfg.engine.binary) if self.binary is None else Path(self.binary)
        self.binary = path if path.is_absolute() else ROOT / path
        self._out = queue.Queue()
        self._ms = self.timeout_ms or self.cfg.engine.timeout_ms
        self._budget_s = self._ms / 1000 + 6

    # ---------- 进程生命周期 ----------

    def open(self) -> "Rapfi":
        if not self.binary.exists():
            raise EngineError(f"找不到引擎: {self.binary}")
        self.proc = subprocess.Popen(
            [str(self.binary)], cwd=str(self.binary.parent),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, encoding="utf-8",
        )
        threading.Thread(target=_pump, args=(self.proc, self._out), daemon=True).start()
        # 实测唯一的 OK 挂在规格行上而不是 START 之后（单发 START 等 10s 收不到 OK），
        # 所以 START 不等回音，它那两行 MESSAGE 由下一次 _command 顺手排掉。
        self._command("START", want_ok=False)
        self._command(str(self.cfg.board.size))
        self._command(f"info timeout_turn {self._ms}", want_ok=False)
        return self

    def restart(self) -> "Rapfi":
        """换个新进程。探针实测 `NEW GAME`/`NEW` 全被拒，搜索树清不干净就只能换进程。

        `model` / `engine_colour` 必须一起作废：新进程肚子里是空的，留着旧镜像会让
        `ask()` 判成「镜像已覆盖盘面」而永远不再灌盘，等于重启完就不出建议了。
        """
        self.close()
        self.model = None
        self.engine_colour = None
        return self.open()

    def close(self) -> None:
        proc, self.proc = self.proc, None
        if proc is None:
            return
        try:
            if proc.poll() is None:
                proc.stdin.write("END\n")
                proc.stdin.flush()
        except (BrokenPipeError, ValueError, AttributeError):
            pass
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.terminate()
        # 管道要等进程收干净再关：pump 线程还在 readline，提前关会让它抛 ValueError。
        for fh in (proc.stdin, proc.stdout):
            try:
                fh.close()
            except (OSError, ValueError):
                pass

    def __enter__(self) -> "Rapfi":
        return self.open()

    def __exit__(self, *exc) -> None:
        self.close()

    # ---------- 一发一收 ----------

    def _command(self, line: str, want_ok: bool = True) -> None:
        if self.proc is None or self.proc.poll() is not None:
            raise EngineError("引擎进程不在")
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()
        if not want_ok:
            return
        deadline = time.time() + 10
        while time.time() < deadline:
            got = self._get(deadline)
            if got == "OK":
                return
            if got.startswith("ERROR"):
                raise EngineError(f"{line!r} 被引擎拒绝: {got}")
            if got == "":
                raise EngineError(f"{line!r} 之后引擎进程退出")
        raise EngineError(f"{line!r} 之后 10s 内没等到 OK")

    def _get(self, deadline: float) -> str:
        try:
            return self._out.get(timeout=min(0.5, max(0.05, deadline - time.time())))
        except queue.Empty:
            return "\0idle"

    def _await_move(self) -> Plan:
        t0 = time.perf_counter()
        deadline = t0 + self._budget_s
        depth = score = None
        pv: tuple[str, ...] = ()
        for n in range(FLOOD_LIMIT):
            if time.perf_counter() > deadline:
                break
            got = self._get(deadline)
            if got == "\0idle":
                continue
            if got == "":
                raise EngineError("引擎进程已退出")
            if got.startswith("ERROR"):
                raise EngineError(f"引擎报错: {got}")
            p = parse_progress(got)
            if p:
                depth, score = p[0], p[1]
                pv = p[2] or pv     # 收尾的 Speed 行没有 pv 段，别让它在最后一拍把 pv 清空
                continue
            mv = parse_move(got)
            if mv:
                return Plan(move=mv, score=score, depth=depth, pv=pv,
                            elapsed_ms=(time.perf_counter() - t0) * 1000)
        raise EngineError(f"{self._budget_s:.0f}s / {FLOOD_LIMIT} 行内未回手，判定引擎失稳")

    # ---------- 提问 ----------

    def set_position(self, board: Board) -> Plan:
        """整盘灌进引擎并立刻拿到轮到一方的最佳手；这是唯一的提问手段。"""
        self._command("board", want_ok=False)
        for y in range(board.size):
            for x in range(board.size):
                code = CODES.get(board.cell(x, y))
                if code:
                    self._command(f"{x},{y},{code}", want_ok=False)
        self._command("done", want_ok=False)
        plan = self._await_move()
        self.engine_colour = board.side_to_move()
        self.model = board.with_changes(
            [Change(plan.move[0], plan.move[1], EMPTY, self.engine_colour)])
        return plan

    def ask(self, board: Board) -> Plan | None:
        """每一问都整盘重灌。引擎镜像已经覆盖真实盘面时返回 None —— 没有新信息，也不该重播旧结论。

        曾经的规则是「只缺对手的子就走 `turn` 增量」，2026-09-11 16:07 实盘为这条规则被秒：
        增量回手是上一轮搜索 pv 的下一格，没重新算过，评分还是从上一轮继承的 mate 幻影，
        于是一手没挡对方活三的 E7 顶着「必胜(余11)」送出去了。重灌慢 1s 上下，
        但问的盘面和答的盘面必然是同一个。

        **回答的是「轮到的一方」，不一定是我方**：引擎执哪方完全由盘石子数奇偶推出，
        在「我方刚落子、对手还没应」的时刻问，它就改执对手那一方（实测 2黑1白 时
        engine_colour 翻成 O 并回了 F5）。调用方必须把 engine_colour 一起说出去，
        不能把结论当成给用户的建议。
        """
        if self.model is not None and not self.model.diff(board):
            return None                   # 镜像已经含下盘面每一颗子，问了也是白问
        return self.set_position(board)
