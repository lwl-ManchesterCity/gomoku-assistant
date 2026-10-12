"""Gomocup 会话驱动器：从 stdin 读命令逐条喂引擎，需要回手的命令就等它落子。

用法：
    printf 'START\\n15\\ninfo timeout_turn 1000\\nBEGIN\\nYOU 0,3\\nEND\\n' \\
      | .venv/bin/python tools/rapfi_session.py

MOVE_WORDS 里的命令会触发一次搜索，等直到出现裸坐标 `x,y` 或超时；其余只短暂等一下。
"""

from __future__ import annotations

import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENGINE_DIR = ROOT / "vendor" / "rapfi"
ENGINE = ENGINE_DIR / "pbrain-rapfi-macos-apple-silicon"
MOVE_WORDS = ("begin", "turn", "takeback", "restart", "move")
SLOW_WORDS = ("board", "about", "show", "status", "yxboard")
SETTLE_S = 0.4
SLOW_WAIT_S = 3.0
MOVE_WAIT_S = 12.0


def is_move(line: str) -> bool:
    parts = line.strip().split(",")
    return len(parts) == 2 and all(p.lstrip("-").isdigit() for p in parts)


def main() -> int:
    proc = subprocess.Popen(
        [str(ENGINE)], cwd=str(ENGINE_DIR),
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, encoding="utf-8",
    )
    out: queue.Queue[str] = queue.Queue()

    def pump():
        for line in iter(proc.stdout.readline, ""):
            out.put(line.rstrip("\n"))
        out.put("<EOF>")

    threading.Thread(target=pump, daemon=True).start()

    def collect(deadline: float, stop_on_move: bool = False) -> list[str]:
        got: list[str] = []
        while time.time() < deadline:
            try:
                line = out.get(timeout=0.05)
            except queue.Empty:
                continue
            got.append(line)
            if stop_on_move and is_move(line):
                break
        return got

    log = open("/tmp/rapfi_session.log", "w", encoding="utf-8")

    def emit(text: str) -> None:
        print(text, flush=True)
        log.write(text + "\n")
        log.flush()

    def wait_for(cmd: str) -> tuple[float, bool]:
        words = cmd.split()
        head = words[0].lower() if words else ""
        if head in MOVE_WORDS:
            return MOVE_WAIT_S, True
        if head in SLOW_WORDS:
            return SLOW_WAIT_S, False
        return SETTLE_S, False

    for raw in sys.stdin:
        cmd = raw.rstrip("\n")
        if cmd.lstrip().startswith("#"):
            continue   # 空行不跳过：board 位置录入可能靠空行结束
        if cmd.strip().lower().startswith("sleep:"):
            time.sleep(float(cmd.split(":", 1)[1]))
            continue
        leftover = collect(time.time() + 0.05)
        for line in leftover:
            emit(f"   [延后自上一条] {line}")
        try:
            proc.stdin.write(cmd + "\n")
            proc.stdin.flush()
        except BrokenPipeError:
            emit(f"!! 引擎已退出，写 {cmd!r} 失败")
            break
        wait_s, want_move = wait_for(cmd)
        emit(f">> {cmd}")
        deadline = time.time() + wait_s
        move, noise = None, 0
        while time.time() < deadline and move is None:
            try:
                line = out.get(timeout=0.05)
            except queue.Empty:
                continue
            if line == "<EOF>":
                emit("   <引擎进程结束>")
                move = "<EOF>"
                break
            noise += 1
            if want_move and is_move(line):
                move = line
            else:
                emit(f"   {line}")
        if want_move:
            emit(f"   ⇒ {cmd!r} 回手 {move or f'<无回手，收到 {noise} 行>'}")
    log.close()
    if proc.poll() is None:
        proc.terminate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
