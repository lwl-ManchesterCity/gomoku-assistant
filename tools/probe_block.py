"""离线探针：给 RapFi 摆出「对方有活三、轮到我」的局面，看它回的是堵还是一手无关棋。

只走 engine.Rapfi 这一层，不碰采集，用来把「引擎不堵」和「盘面上根本没送到」分开。
"""
from __future__ import annotations

import sys
from dataclasses import replace

sys.path.insert(0, "src")

from gomoku.board import BLACK, EMPTY, WHITE, Change, coord_name
from gomoku.config import load
from gomoku.engine import MATE, Rapfi
from gomoku.pattern import Scan, safety_note
from gomoku.state import empty_board

CFG = load()
SIZE = CFG.board.size


def build(stones) -> "object":
    b = empty_board(SIZE)
    return b.with_changes([Change(x, y, EMPTY, c) for x, y, c in stones])


def report(tag: str, engine: Rapfi, board) -> None:
    plan = engine.set_position(board)
    mover = board.side_to_move()
    scan = Scan(board)
    opp = WHITE if mover == BLACK else BLACK
    print(f"\n=== {tag} ===")
    print(f"轮到我方 = {'黑' if mover == BLACK else '白'}  "
          f"对方先手: {scan.forcing(opp)}  我方棋型: {scan.four(mover)} {scan.three(mover)}")
    score = "None" if plan.score is None else (
        f"必胜(余{MATE - int(abs(plan.score))})" if abs(plan.score) >= MATE - 1000
        else f"{plan.score:.0f}"
    )
    print(f"引擎回 {plan.name}  depth={plan.depth} eval={score} {plan.elapsed_ms:.0f}ms")
    print(f"pv={' '.join(plan.pv[:8])}")
    note = safety_note(board, mover, plan.move)
    print(f"safety_note: {note if note else '无告警'}")
    print(f"期望的堵点: " + " ".join(
        coord_name(x, y) for x, y in scan.fives(opp)))


def main() -> int:
    # A. 对方一个活三，我方三颗散子（毫无棋型）：标准「必堵」局
    A = build([(5, 6, WHITE), (6, 6, WHITE), (7, 6, WHITE),
               (1, 1, BLACK), (13, 1, BLACK), (1, 13, BLACK)])
    # B. 对方活三 + 我方一个眠三（端点已被堵死）：引擎仍该先处理对方先手
    B = build([(5, 6, WHITE), (6, 6, WHITE), (7, 6, WHITE),
               (10, 10, BLACK), (11, 11, BLACK), (13, 13, BLACK)])
    # C. 对方活三 + 我方活三：对攻局，谁快，允许不堵
    C = build([(5, 6, WHITE), (6, 6, WHITE), (7, 6, WHITE),
               (10, 10, BLACK), (11, 11, BLACK), (12, 12, BLACK)])
    # D. 我方一步就能成活四（双用点）：这类「直接能赢」它该不该走
    D = build([(5, 6, WHITE), (6, 6, WHITE), (7, 6, WHITE),
               (10, 10, BLACK), (11, 11, BLACK), (12, 12, BLACK), (13, 13, WHITE)])
    # E. 我方有冲四（一手成五）：绝对该立刻走
    E = build([(7, 7, BLACK), (8, 7, BLACK), (9, 7, BLACK), (10, 7, BLACK),
               (5, 6, WHITE), (6, 6, WHITE), (7, 6, WHITE)])

    for tag, timeout in (("timeout=1500ms（当前配置）", CFG.engine.timeout_ms),
                         ("timeout=4000ms", 4000)):
        cfg = replace(CFG, engine=replace(CFG.engine, timeout_ms=timeout))
        print(f"\n########## {tag} ##########")
        eng = Rapfi(cfg).open()
        try:
            for name, board in (("A 必堵局", A), ("B 我方眠三", B), ("C 对攻局", C),
                                ("D 我方成活四", D), ("E 我方冲四", E)):
                report(name, eng, board)
        finally:
            eng.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
