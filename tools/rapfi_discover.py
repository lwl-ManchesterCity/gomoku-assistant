"""命令词表发现：挨个试候选输入，区分「命令不认」「命令认了但参数不认」「完全接受」。

用法：.venv/bin/python tools/rapfi_discover.py

坑：引擎报错时会把命令词**大写**回显（`Unknown command: NAME`），
所以比对必须大小写无关，否则全部会被误判成「已接受」。
"""

from __future__ import annotations

import queue
import subprocess
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENGINE_DIR = ROOT / "vendor" / "rapfi"
ENGINE = ENGINE_DIR / "pbrain-rapfi-macos-apple-silicon"

PREFIX = ["START", "15", "info timeout_turn 500"]
CANDIDATES = [
    ("裸坐标(已试过=未知)", "0,3"),
    ("裸坐标空格", "0 3"),
    ("GTP式", "h8"),
    ("YOU 逗号", "you 0,3"),
    ("YOU 空格", "you 0 3"),
    ("usermove 逗号", "usermove 0,3"),
    ("usermove 字母", "usermove h8"),
    ("move", "move 0,3"),
    ("otherposition", "otherposition 7,7 0,3"),
    ("position", "position 7,7 0,3"),
    ("startpos", "startpos"),
    ("force", "force"),
    ("new", "new"),
    ("undo", "undo"),
    ("hint", "hint"),
    ("quit", "quit"),
    ("help", "help"),
    ("about", "about"),
    ("name", "name"),
    ("version", "version"),
    ("result", "result 1-0"),
    ("ping", "ping 1"),
    ("protover", "protover 2"),
    ("xboard", "xboard"),
    ("board", "board"),
    ("show", "show"),
    ("status", "status"),
    ("edit", "edit"),
    ("accepted", "accepted"),
    ("maxmemory", "maxmemory 256"),
    ("time", "time 2000"),
    ("otim", "otim 2000"),
]


def send_and_collect(lines: list[str]) -> list[str]:
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
    got: list[str] = []
    for line in lines:
        try:
            proc.stdin.write(line + "\n")
            proc.stdin.flush()
        except BrokenPipeError:
            got.append("<进程已退出>")
            break
        deadline = time.time() + 1.5
        while time.time() < deadline:
            try:
                got.append(out.get(timeout=0.05))
            except queue.Empty:
                continue
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
    return got


def classify(cmd: str, got: list[str]) -> tuple[str, str]:
    errs = [g for g in got if "nknown command" in g]
    head = cmd.split()[0].upper()
    for e in errs:
        tok = e.split(":", 1)[-1].strip().upper() if ":" in e else ""
        if tok == head:
            return "命令不认", e
        if tok:
            return "命令认了/参数不认", e
    other = [g for g in got if g.upper().startswith(("ERROR", "UNKNOWN"))]
    return ("接受", other[-1] if other else "无报错")


def main() -> None:
    buckets: dict[str, list[str]] = {}
    for label, cmd in CANDIDATES:
        got = send_and_collect(PREFIX + [cmd])
        kind, detail = classify(cmd, got)
        buckets.setdefault(kind, []).append(label or cmd)
        print(f"  {cmd:<28} {kind:<12} {detail}")
    print()
    for kind, items in buckets.items():
        print(f"{kind}: {len(items)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
