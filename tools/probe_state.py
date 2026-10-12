"""离线分析：`confirm_ticks` / `max_changes` 这两个默认值在实盘日志上到底是什么代价。

只吃 `logs/events.jsonl` 里记录下来的**原始** grid（消抖前的读数），所以重放不受当时
参数影响。要回答的是三个能数的量：

1. 一闪而过的读数有几处 —— 它决定 confirm_ticks 该不该 >1；
2. 换档后同一串读数提交了几手、末拍还压着几处待复验 —— 每加一档就多等一拍；
3. 单拍变化数超过 max_changes 的有几拍，那些拍随后是不是稳定住了 —— 决定整拍丢弃
   挡住的是误读，还是把真落子也丢了。

样本量小时结论只能写成「这份日志里没观察到」，不能写成「不需要」。
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, "src")

from gomoku import config
from gomoku.board import Board, EMPTY
from gomoku.state import Tracker

CFG = config.load()
PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("logs/events.jsonl")


def main() -> int:
    recs = [json.loads(line) for line in PATH.read_text(encoding="utf-8").splitlines() if line]
    grids = [r["grid"] for r in recs if "grid" in r]
    ms = [r["ms"] for r in recs if "ms" in r]
    errs = Counter(r["error"].split("，")[0][:24] for r in recs if "error" in r)
    print(f"### {PATH}  共 {len(recs)} 拍，其中有盘面读数的 {len(grids)} 拍")
    if ms:
        print(f"    单拍耗时 中位 {sorted(ms)[len(ms) // 2]:.0f}ms  最慢 {max(ms):.0f}ms"
              f"   采样间隔 {CFG.capture.interval_s}s")
    if errs:
        print(f"    失败拍 {sum(errs.values())} 拍：{dict(errs)}")
    if len(grids) < 2:
        print("    样本不足以定参数。")
        return 0

    boards = [Board(tuple(tuple(row) for row in g)) for g in grids]
    size = boards[0].size

    # 1) 每处读数连续稳定了几拍；1 拍 = 一闪而过，正是 confirm_ticks>1 要挡的东西。
    #    第 0 拍是初始局面、不是「新出现的一手」，不计。
    appear: Counter[int] = Counter()     # 某个点读成有子，持续几拍
    vanish: Counter[int] = Counter()     # 某个点读成空（子没了），持续几拍
    for y in range(size):
        for x in range(size):
            seq = [b.cell(x, y) for b in boards]
            i = 1
            while i < len(seq):
                if seq[i] == seq[i - 1]:
                    i += 1
                    continue
                j = i
                while j + 1 < len(seq) and seq[j + 1] == seq[i]:
                    j += 1
                (appear if seq[i] != EMPTY else vanish)[j - i + 1] += 1
                i = j + 1
    print("\n    一处「新读出的子」连续稳定几拍（1 = 一闪而过，2 = 只活了半秒）:")
    for k in sorted(appear):
        print(f"      出现并保持 {k:>2} 拍: {appear[k]} 处")
    if vanish:
        print("    一处「子又读回空格」持续几拍（正常棋局不该有，有就是误读）:")
        for k in sorted(vanish):
            print(f"      消失并保持 {k:>2} 拍: {vanish[k]} 处")
    print(f"      → 只出现 1 拍的 {appear[1]} 处，只出现 2 拍的 {appear[2]} 处")

    # 2) 换 confirm_ticks 重放
    print("\n    同一串读数换 confirm_ticks 重放：")
    for ct in (1, 2, 3, 4):
        cfg = replace(CFG, state=replace(CFG.state, confirm_ticks=ct))
        tr = Tracker(cfg)
        committed = 0
        pending_last = 0
        for i, b in enumerate(boards):
            s = tr.update(b)
            committed += len(s.changes)
            pending_last = len(s.held)
        c = tr.board.counts()
        print(f"      confirm_ticks={ct}: 提交 {committed:>2} 手，末拍待复验 {pending_last} 处，"
              f"收尾盘面 黑={c['X']} 白={c['O']}（最后一拍原始读数 黑={boards[-1].counts()['X']}"
              f" 白={boards[-1].counts()['O']}）")

    # 3) max_changes
    print("\n    单拍原始变化数分布：")
    diffs = [len(boards[i - 1].diff(boards[i])) for i in range(1, len(boards))]
    print("      " + "  ".join(f"{n}处:{c}拍" for n, c in sorted(Counter(diffs).items())))
    over = [(i, n) for i, n in enumerate(diffs) if n > CFG.state.max_changes]
    print(f"      超过 max_changes={CFG.state.max_changes} 的：{len(over)} 拍")
    for i, n in over:
        same = 1
        for j in range(i + 2, len(boards)):
            if boards[j].diff(boards[i + 1]) == []:
                same += 1
            else:
                break
        tag = "真局面，重同步会采纳" if same >= CFG.state.resync_ticks else "随后就变了 = 过渡帧/误读，整拍丢弃是对的"
        print(f"        第 {i + 1} 拍 {n} 处变化：之后连续 {same} 拍不变 → {tag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
