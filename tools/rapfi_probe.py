"""一次性探针：实测 RapFi 的 Gomocup 会话格式，并验证坐标映射没有被镜像。

用法：.venv/bin/python tools/rapfi_probe.py

已实测到的事实（写进适配器）：
* Gomocup 会话是 START → N → INFO → (BEGIN | YOU x,y)，`END` 会真的结束进程，
  所以每个场景必须另起一个进程。
* 落子是**裸坐标** `7,7` 输出，不带规范要求的 `COORDS ` 前缀，解析器两种都得收。
* 其他行全是 `MESSAGE ...` 日志，要能忽略。
"""

from __future__ import annotations

import subprocess
import sys
import threading
import queue
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENGINE_DIR = ROOT / "vendor" / "rapfi"
ENGINE = ENGINE_DIR / "pbrain-rapfi-macos-apple-silicon"
TIMEOUT_MS = 2000


def session(lines: list[str], settle: float = 1.0) -> list[str]:
    """起一个引擎进程，按顺序喂 lines，收回所有输出后结束。"""
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
            got.append(f"<进程已退出，写 {line!r} 失败>")
            break
        print(f">> {line}")
        deadline = time.time() + settle + TIMEOUT_MS / 1000
        while time.time() < deadline:
            try:
                item = out.get(timeout=0.1)
            except queue.Empty:
                continue
            got.append(item)
            print(f"   {item}")
            if "\n" not in item and ("," in item or item.startswith("OK")):
                break
    try:
        proc.stdin.close()
    except Exception:
        pass
    if proc.poll() is None:
        proc.terminate()
        proc.wait(timeout=5)
    return got


def main() -> int:
    if not ENGINE.exists():
        print(f"找不到引擎: {ENGINE}", file=sys.stderr)
        return 1
    head = ["START", "15", f"INFO timeout_turn {TIMEOUT_MS}", "INFO timeout_match 60000"]

    print("=== 1. 空盘先手，应落在天元 H8 = 7,7 ===")
    first = session(head + ["BEGIN"])

    print("\n=== 2. 靠边活三：黑 (0,3)(1,3)(2,3)，白唯一防点是 (3,3) ===")
    a = session(head + [f"YOU {x},3" for x in (0, 1, 2)])

    print("\n=== 3. 同一形状换到 y=11，白的答案行号必须跟过去 ===")
    b = session(head + [f"YOU {x},11" for x in (0, 1, 2)])

    def replies(got: list[str]) -> list[str]:
        return [g for g in got if "," in g and not g.startswith(("MESSAGE", "ERROR", "OK"))]

    print("\n=== 汇总 ===")
    print(f"  y=3  局：白落点 {replies(a)}")
    print(f"  y=11 局：白落点 {replies(b)}")
    print("  判据：最后一手应为 3,3 与 3,11；若两局答案行号互换说明输入被镜像。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
