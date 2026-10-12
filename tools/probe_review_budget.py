"""同一份日志问两遍引擎，只换每问的预算 —— 看「引擎首选」稳不稳。

复盘那一支每一手都要问一次引擎，而 2026-09-28 那份输棋日志连跑两次给出过不一样的
偏离计数，所以写经验文档之前得先知道：那到底是引擎在预算边界上换首选，还是我们对
「哪一手算偏离」的判据在抖。两档预算 = 实盘那一档（`[engine].timeout_ms`）和
`review --engine-ms` 默认那一档，逐手并排打出来，不一样的那几行标 `≠`。

跑法：PYTHONPATH=src .venv/bin/python tools/probe_review_budget.py [日志路径 ...]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gomoku.board import BLACK, WHITE                        # noqa: E402
from gomoku.config import load                               # noqa: E402
from gomoku.engine import Rapfi                              # noqa: E402
from gomoku.lessons import read_frames, round_facts          # noqa: E402
from gomoku.__main__ import REVIEW_ENGINE_MS, RUNS  # noqa: E402

CFG = load()
DEFAULTS = [ROOT / "samples" / "2026-09-28-负-22手.txt",
            ROOT / "samples" / "2026-09-28-胜-21手.txt"]


def 跑一遍(日志: Path, mine: str, ms: int):
    引擎 = Rapfi(CFG, timeout_ms=ms).open()
    try:
        return round_facts(read_frames(日志), mine, 引擎.ask)
    finally:
        引擎.close()


def 比(日志: Path, mine: str) -> None:
    实盘 = 跑一遍(日志, mine, CFG.engine.timeout_ms)
    放宽 = 跑一遍(日志, mine, REVIEW_ENGINE_MS)
    print(f"\n### {日志.name}｜执{'黑' if mine == BLACK else '白'}"
          f"｜{CFG.engine.timeout_ms}ms 档 {len(实盘.turns)} 问 / "
          f"{REVIEW_ENGINE_MS}ms 档 {len(放宽.turns)} 问")
    if len(实盘.turns) != len(放宽.turns):
        print("    手数就对不上，两档走的路径不同，下面只能按手序号并排看")
    抖 = 0
    for a, b in zip(实盘.turns, 放宽.turns):
        同 = a.pick == b.pick
        抖 += not 同
        print(f"    {a.at} 我落 {a.play}  {a.pick}{'=' if 同 else '≠'}{b.pick}"
              f"｜{a.elapsed_ms:.0f}ms vs {b.elapsed_ms:.0f}ms"
              + ("" if 同 else "   ← 首选随预算换了"))
    for 档, 局 in ((f"{CFG.engine.timeout_ms}ms（实盘档）", 实盘),
                   (f"{REVIEW_ENGINE_MS}ms（review 档）", 放宽)):
        print(f"    {档}: 偏离 {len(局.deviations)} 处、该堵没堵 {len(局.misses)} 处")
    print(f"    首选不一致的手数：{抖}/{min(len(实盘.turns), len(放宽.turns))}")


if __name__ == "__main__":
    日志们 = [Path(a) for a in sys.argv[1:] if not a.startswith("-")] or DEFAULTS
    t0 = time.perf_counter()
    for 日志 in 日志们:
        比(日志, BLACK)
    print(f"\n两档加起来跑了 {(time.perf_counter() - t0) / 60:.1f} 分钟。"
          "自动落的逐拍日志在 " + str(RUNS))
