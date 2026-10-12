"""问引擎两个「走子方必被杀」的局面，看那个 Eval 的符号到底站在哪一方。

`engine_colour` 是我们按石子奇偶**假设**的，Gomocup 那一头没有任何一行让 RapFi 声明
它执哪色；而 `explain.format_eval`、`__main__._eval_text` 和出口那句「必胜/必败」
全都把进度行的 Eval 读成**走子方为正**。这两件事只要有一件不成立，实盘印出来的
「必胜」就可能是「必败」—— 2026-09-28 那局 21:20 重问时同一形状把 +M3 判给了白方，
正是这条链上没闭的一段。

所以这里绕开适配层，把原始行整条打出来：符号、回手坐标、pv 首手三样并排看，
才知道该不该让引擎自己声明颜色。

跑法：PYTHONPATH=src .venv/bin/python tools/probe_eval_side.py
"""

from __future__ import annotations

import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gomoku.board import BLACK, EMPTY, WHITE, Board, Change, coord_name   # noqa: E402
from gomoku.config import load                                            # noqa: E402
from gomoku.engine import CODES, MOVE_RE, parse_progress                  # noqa: E402
from gomoku.lessons import read_frames                                    # noqa: E402
from gomoku.pattern import five_on_board                                  # noqa: E402
from gomoku.state import empty_board                                      # noqa: E402

CFG = load()
BIN = Path(CFG.engine.binary)
if not BIN.is_absolute():
    BIN = ROOT / CFG.engine.binary
BUDGET_MS = 6000
SAMPLE = ROOT / "samples" / "2026-09-28-负-22手.txt"


def build(*stones) -> Board:
    return empty_board(CFG.board.size).with_changes(
        [Change(x, y, EMPTY, c) for x, y, c in stones])


def transcript(board: Board, wait_s: float = 45.0) -> tuple[list[str], str | None]:
    """新进程整盘灌进去，返回（收到的每一行原始输出, 那手裸坐标）。

    一次性把所有命令写完再读，读到手为止。旧探针按「每发一条等到一个 OK/MESSAGE」收线，
    结果在第一条进度行上就退出了，从没拿到过真正的那手棋。
    """
    cmds = ["START", str(board.size), f"info timeout_turn {BUDGET_MS}", "board"]
    for y in range(board.size):
        for x in range(board.size):
            code = CODES.get(board.cell(x, y))
            if code:
                cmds.append(f"{x},{y},{code}")
    cmds.append("done")

    lines: "queue.Queue[str]" = queue.Queue()

    def pump(proc) -> None:
        for raw in iter(proc.stdout.readline, ""):
            got = raw.strip()
            if got:
                lines.put(got)
        lines.put("")        # 空串 = 进程结束

    proc = subprocess.Popen([str(BIN)], cwd=str(BIN.parent), stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1, encoding="utf-8")
    threading.Thread(target=pump, args=(proc,), daemon=True).start()
    for c in cmds:
        proc.stdin.write(c + "\n")
    proc.stdin.flush()

    out: list[str] = []
    move: str | None = None
    deadline = time.time() + wait_s
    while time.time() < deadline:
        try:
            got = lines.get(timeout=0.5)
        except queue.Empty:
            continue
        if got == "":
            break
        out.append(got)
        if MOVE_RE.match(got):
            move = got
            break
    try:
        proc.stdin.write("END\n")
        proc.stdin.flush()
    except OSError:
        pass
    proc.wait(timeout=5)
    return out, move


槽位 = [(x, y) for y in (2, 5, 8) for x in (8, 10, 12)]   # 互不相邻的散子位，别在右下角再攒出一个四


def 一手成五局(持四: str) -> Board:
    """第 0 行摆四连（成五点 E1 空着），谁走谁一手成五 —— 两例只差颜色互换。

    散子只用来把先后交到手握四连的那一家：黑=白 ⇒ 轮到黑；黑=白+1 ⇒ 轮到白。
    """
    对家 = WHITE if 持四 == BLACK else BLACK
    持四散, 对家散 = (1, 5) if 持四 == BLACK else (1, 6)
    子 = [(x, 0, 持四) for x in range(4)]
    子 += [(*xy, 持四) for xy in 槽位[:持四散]]
    子 += [(*xy, 对家) for xy in 槽位[持四散:持四散 + 对家散]]
    return build(*子)


def 双四局(持四: str) -> Board:
    """两条贴边冲四摆在同一家手里 ⇒ 对面一手挡不干净，**先走那一家必输**且无可争辩。

    这一例才问得出问题。甲/乙两局里「走子方」和「赢面方」是反的，所以：
    Eval 跟着走子方 ⇒ 两例同号（都该是负，走子方被杀）；跟着固定颜色 ⇒ 两例反号；
    跟着赢面方 ⇒ 两例同为正。之前试过「整盘换色」和「加一颗远角子」两种镜像，都不成立：
    换色会连带把先手方在攻防里的地位换掉，加子那一例白方本来就有 J4 自己的成四点，
    谁赢说不死，符号也就推不出来。
    """
    对家 = WHITE if 持四 == BLACK else BLACK
    四 = [(x, 0, 持四) for x in range(4)]                       # A1-D1，成五点只有 E1
    四 += [(x, 14, 持四) for x in (11, 12, 13, 14)]             # L15-O15，成五点只有 J15
    散: list[tuple[int, int, str]] = [(x, 13, 对家) for x in range(0, 15, 2)]   # 隔一路摆，攒不出四
    return build(*四, *(散[:7] if 持四 == BLACK else 散))


def 报(标题: str, board: Board) -> None:
    c = board.counts()
    轮 = board.side_to_move()
    print(f"\n### {标题}")
    print(f"    盘 黑={c[BLACK]} 白={c[WHITE]}  按棋规轮到 {'黑' if 轮 == BLACK else '白'}")
    for 色 in (BLACK, WHITE):
        点 = [coord_name(x, y) for y in range(board.size) for x in range(board.size)
             if board.cell(x, y) == EMPTY and five_on_board(
                 board.with_changes([Change(x, y, EMPTY, 色)]), 色)]
        if 点:
            print(f"    {'黑' if 色 == BLACK else '白'}方一手成五的点: {' '.join(点)}")
    线, 手 = transcript(board)
    名 = coord_name(*(int(v) for v in 手.split(","))) if 手 else "-"
    print(f"    回手: {手} = {名}")
    进度 = [parse_progress(x) for x in 线 if parse_progress(x)]
    if 进度:
        最后 = 进度[-1]
        print(f"    最后一条进度行: depth={最后[0]} Eval={最后[1]} pv={' '.join(最后[2])}")
    else:
        print("    没给进度行（一手成五那类直接回手，实测连 Eval 都不给）")
    for x in 线:
        if not x.startswith("MESSAGE"):
            print("    raw:", x)


if __name__ == "__main__":
    print(f"引擎 {BIN}｜每问预算 {BUDGET_MS}ms")
    print("判据：甲乙两局的「走子方」都是被双四杀掉的那一方。"
          "\n  同号且为负 ⇒ Eval 跟着走子方，`format_eval`/出口的「必胜/必败」读法成立；"
          "\n  一正一负 ⇒ Eval 钉死在某个颜色上，那个读法有一半是反的；"
          "\n  同号且为正 ⇒ Eval 跟着赢面方，实盘印出来的「必败」就全错了。")
    报("甲 黑握双四（E1/K15），白先手 ⇒ 白被杀", 双四局(BLACK))
    报("乙 白握双四（E1/K15），黑先手 ⇒ 黑被杀", 双四局(WHITE))
    报("丙 一手成五，黑先手（对照：这类回手实测不给 Eval）", 一手成五局(BLACK))
    if SAMPLE.exists():
        帧 = [b for _, b in read_frames(SAMPLE)
              if b.counts()[BLACK] == 20 and b.counts()[WHITE] == 20]
        if not 帧:
            print(f"\n（{SAMPLE.name} 里没有 20:20 那一帧）")
            raise SystemExit(0)
        报("丁 09-28 那一帧原样（黑=白 ⇒ 轮到黑；实盘这里回 J4 +M3）", 帧[0])
    else:
        print(f"\n（没有 {SAMPLE}，跳过真实局面那一例）")
