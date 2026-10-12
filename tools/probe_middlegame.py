"""离线探针 2：整盘重灌在「对方有活三/冲四」的中盘局上到底干不干净。

三件事一起测：
1. 一批接近实盘的形状（活三在中央/贴边、跳三、抱三、双活三、冲四、我方有反将），
   逐条用 pattern.safety_note 判「引擎这手有没有处理对方先手」。
2. 同一个局面：换新进程问 vs 在同一进程里连着问，答案是否一致（`board` 重灌不清 TT 的话这里会露）。
3. 引擎认不认 `NEW GAME`（决定「每问清一次搜索树」这条优化能不能做）。
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

CFG = replace(load(), engine=replace(load().engine, timeout_ms=2000))
SIZE = CFG.board.size
B, W = BLACK, WHITE


def build(stones):
    return empty_board(SIZE).with_changes([Change(x, y, EMPTY, c) for x, y, c in stones])


def W_(name):   # "F6" -> (x, y)
    return ord(name[0]) - ord("A"), int(name[1:]) - 1


X_, O_ = B, W
CASES = [
    # 09-11 16:07 那局原样：白 F6 F7 F8 活三，黑 H8 I7 D8
    ("被秒原局", [(*W_(c), X_) for c in ("H8", "I7", "D8")] + [(*W_(c), O_) for c in ("F6", "F7", "F8")]),
    # 中央活三，我方只有散子
    ("中央活三", [(*W_(c), X_) for c in ("H8", "D4", "M12")] + [(*W_(c), O_) for c in ("F8", "G8", "I8")]),
    # 贴边活三（一端已在盘外，实际是眠三）——正确做法是不堵
    ("边线眠三", [(*W_(c), X_) for c in ("H8", "D4", "M12")] + [(*W_(c), O_) for c in ("A8", "B8", "C8")]),
    # 跳三 F7 . H7 I7：成四点是 G7，堵点也是 G7
    ("跳三", [(*W_(c), X_) for c in ("H8", "D4", "M12")] + [(*W_(c), O_) for c in ("F7", "H7", "I7")]),
    # 抱三 F7 G7 . I7
    ("抱三", [(*W_(c), X_) for c in ("H8", "D4", "M12")] + [(*W_(c), O_) for c in ("F7", "G7", "I7")]),
    # 斜向活三 + 我方一个眠三
    ("斜活三", [(*W_(c), X_) for c in ("H6", "D4", "M12")] + [(*W_(c), O_) for c in ("F6", "G7", "H8")]),
    # 双活三（十字）：一手堵不完，看它怎么走
    ("双活三", [(*W_(c), X_) for c in ("D4", "M12", "A1", "O1", "A15", "O15")] + [(*W_(c), O_) for c in
                ("F7", "G7", "H7", "G6", "G8", "G9")]),
    # 对方冲四（一手成五），必须应
    ("对方冲四", [(*W_(c), X_) for c in ("H8", "D4", "M12")] + [(*W_(c), O_) for c in ("E7", "F7", "G7", "H7")]),
    # 对方活三 + 我方一手成活四（允许对攻不堵）
    ("我方反将", [(*W_(c), X_) for c in ("E10", "F10", "G10")] + [(*W_(c), O_) for c in ("F7", "G7", "H7")]),
]

TAG = {B: "黑", W: "白"}


def run(eng: Rapfi, fresh_note: str) -> list[tuple[str, str]]:
    out = []
    for name, stones in CASES:
        board = build(stones)
        mover = board.side_to_move()
        opp = WHITE if mover == BLACK else BLACK
        forcing = Scan(board).forcing(opp)
        plan = eng.set_position(board)
        note = safety_note(board, mover, plan.move)
        score = ("None" if plan.score is None else
                 (f"{'必胜' if plan.score > 0 else '必败'}(余{MATE - int(abs(plan.score))})"
                  if abs(plan.score) >= MATE - 1000 else f"{plan.score:.0f}"))
        flag = "报警" if note else ("无先手" if not forcing else "OK")
        print(f"  [{fresh_note}] {name:<6} 对方先手={forcing and str(forcing) or '无':<28} "
              f"回手={plan.name:<4} {score:<10} d{plan.depth} {plan.elapsed_ms:.0f}ms -> {flag}"
              + (f"\n         ⚠ {note}" if note else ""))
        out.append((name, plan.name))
    return out


def main() -> int:
    print("=== 1) 同一批局面：换新进程问 vs 同进程连问 ===")
    a = []
    for name, stones in CASES:
        eng = Rapfi(CFG).open()
        try:
            board = build(stones)
            mover = board.side_to_move()
            plan = eng.set_position(board)
            a.append((name, plan.name, safety_note(board, mover, plan.move) is None))
        finally:
            eng.close()
    eng = Rapfi(CFG).open()
    try:
        b = run(eng, "同进程")
    finally:
        eng.close()
    a_map = {n: m for n, m, _ in a}
    差异 = [(n, a_map[n], m) for n, m in b if a_map[n] != m]
    print(f"\n换进程 vs 同进程连问：{len(a)} 局里不一致 {len(差异)} 局 {差异}")
    漏防 = [(n, m) for (n, m, ok), (n2, m2) in zip(a, b) if not ok]
    print(f"新进程下 safety_note 报警 {len(漏防)}/{len(a)} 局 {漏防}")

    print("\n=== 2) 引擎认不认 NEW GAME（清搜索树用）===")
    eng = Rapfi(CFG).open()
    for cmd in ("NEW GAME", "new game", "NEW"):
        try:
            eng._command(cmd)
            print(f"  {cmd!r} -> OK，支持")
        except Exception as exc:
            print(f"  {cmd!r} -> {type(exc).__name__}: {exc}")
    eng.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
