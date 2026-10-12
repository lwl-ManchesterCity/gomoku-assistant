"""采集循环：全屏定位一次，之后跟随 ROI 高速采集；每拍解析、消抖、打印可信变化。

P1 时这里故意把每拍原始差分摊出来看过渡帧；现在消抖交给 state.Tracker，
本模块只负责采集、标定跟随和汇报。JSONL 仍记录每拍的**原始**盘面网格，
所以真出现误读时可以在离线重放里复原当时的读数，不受消抖逻辑影响。

锁定之后棋盘基本不动，所以标定不必每拍重做：每拍先跑一次门卫校验（实测 2ms），
通过就沿用上次的格点，不通过才回去做全量标定（实测 21ms）。放行过的档位实测重读整盘
一格都不错（空盘/中局盘到 4px 仍放行，压着「＋」标记子的那面 3px 就退回全量标定，
整格平移一律拦得住，见 `calibrate.verify_grid`）——拦得早只多花 19ms，不亏。
每一拍是否沿用了旧格点写进 JSONL 的 `reuse`，误读定责时要能对上。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from itertools import count
from pathlib import Path

from .board import Board, Change
from .calibrate import CalibrationError, Geometry, detect_board, verify_grid
from .capture import Capture, CaptureError, Frame
from .config import Config
from .state import Tracker
from .vision import SampleError, parse_board


@dataclass(frozen=True)
class Tick:
    board: Board           # 状态层提交的可信局面
    raw: Board             # 本拍未经消抖的原始读数
    changes: list[Change]  # 本拍确认提交的变化
    held: list[Change]     # 本拍读到但待下一拍复验
    reason: str | None     # 非空 = 本拍被整拍丢弃或发生重同步
    elapsed_ms: float
    mode: str


def _ts() -> str:
    now = time.time()
    return time.strftime("[%H:%M:%S.", time.localtime(now)) + f"{int(now * 1000) % 1000:03d}]"


def _iso_ms() -> str:
    now = time.time()
    return time.strftime("%Y-%m-%dT%H:%M:%S.", time.localtime(now)) + f"{int(now * 1000) % 1000:03d}"


@dataclass
class Watcher:
    cfg: Config
    log: object = print
    use_roi: bool = True
    events_path: str | None = None
    on_tick: object = None      # callable(Tick) -> None；P2 的建议层挂这里，loop 不认识它
    cap: Capture = field(default=None, repr=False)
    geometry: Geometry | None = field(default=None, repr=False)
    geom_origin: tuple[int, int] | None = field(default=None, repr=False)   # 这份格点是在哪个帧原点标出来的
    screen_bbox: tuple[int, int, int, int] | None = field(default=None, repr=False)
    tracker: Tracker = field(default=None, repr=False)
    events: list[dict] = field(default_factory=list, repr=False)
    fails: int = 0
    locked: bool = False      # 已锁定棋盘，可以走 ROI
    calibrated: bool = False  # 已向用户打印过一面盘
    _fh: object = field(default=None, init=False, repr=False)

    def __post_init__(self):
        if self.cap is None:
            self.cap = Capture()
        if self.tracker is None:
            self.tracker = Tracker(self.cfg)
        if self.events_path:
            Path(self.events_path).parent.mkdir(parents=True, exist_ok=True)
            self._fh = open(self.events_path, "a", encoding="utf-8")

    def _log_event(self, rec: dict) -> None:
        self.events.append(rec)
        if self._fh:
            self._fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self._fh.flush()

    def _mode(self) -> str:
        if self.use_roi and self.locked and self.screen_bbox is not None:
            return "ROI"
        return "全屏"

    def _reusable(self, frame: Frame) -> bool:
        """ROI 帧和标定那次是同一个取样矩形，且格点还压在网格线上 —— 本拍不必重标定。

        放行过的档位实测重读一格都不错：空盘/中局盘到 4px 仍放行（分离度 90→52），5px 起
        掉到阈值以下；压着「＋」标记子的那面 3px 就退回全量标定 —— 放行区间随盘面对比度
        浮动，早退只多花 19ms。整格（一格整数倍）平移时网格图案自相似，但盘面的边界会越出
        缓存框，分离度塌到负值，同样拦得住 —— 那一条要是放过去，状态层连续几拍稳定就会把
        错位盘当真局面采纳。
        """
        if self.geometry is None or self.geom_origin != frame.origin:
            return False
        return verify_grid(frame.rgb, self.geometry, self.cfg)

    def _grab(self) -> tuple[Frame, Geometry, bool]:
        """取一帧并给出该用哪副格点。返回的 bool = 这一拍是否沿用了上次标定。"""
        if self._mode() == "ROI":
            frame = self.cap.grab_rect(self.screen_bbox, self.cfg.capture.roi_expand)
            if self._reusable(frame):
                return frame, self.geometry, True
        else:
            frame = self.cap.grab_full()
        geom = detect_board(frame.rgb, self.cfg)
        x0, y0, x1, y1 = geom.bbox
        ox, oy = frame.origin
        self.geometry = geom
        self.geom_origin = frame.origin
        self.screen_bbox = (ox + x0, oy + y0, ox + x1, oy + y1)
        return frame, geom, False

    def step(self) -> Tick | None:
        """一拍。标定/采集失败时返回 None，连续 N 次后退回全屏重定位。"""
        t0 = time.perf_counter()
        mode = self._mode()
        try:
            frame, geom, reused = self._grab()
            board = parse_board(frame.rgb, geom, self.cfg)
        except (CalibrationError, SampleError, CaptureError) as exc:
            self.fails += 1
            self.calibrated = False
            fell_back = self.locked and self.fails >= self.cfg.capture.roi_relocate_fails
            if fell_back:
                self.locked = False
            self.log(
                f"{_ts()} {mode} 标定失败 x{self.fails}: {exc}"
                + ("  → 退回全屏重定位" if fell_back else "")
            )
            self._log_event({"t": _iso_ms(), "mode": mode, "error": str(exc)})
            return None
        self.fails = 0
        self.locked = True
        sync = self.tracker.update(board)
        elapsed = (time.perf_counter() - t0) * 1000
        self._log_event(
            {
                "t": _iso_ms(),
                "mode": mode,
                "ms": round(elapsed, 1),
                "reuse": reused,     # 下一轮实盘要靠它把「误读」和「沿用旧格点」对上
                "bbox": list(self.screen_bbox),
                "grid": ["".join(row) for row in board.grid],
                "changes": [str(ch) for ch in sync.changes],
                "held": [str(ch) for ch in sync.held],
                "reason": sync.reason,
            }
        )
        return Tick(
            board=sync.board,
            raw=board,
            changes=sync.changes,
            held=sync.held,
            reason=sync.reason,
            elapsed_ms=elapsed,
            mode=mode,
        )

    def run(self, max_ticks: int | None = None) -> None:
        try:
            for i in count():
                if max_ticks is not None and i >= max_ticks:
                    break
                t0 = time.perf_counter()
                tick = self.step()
                if tick is not None:
                    self._report(tick)
                    if self.on_tick is not None:
                        self.on_tick(tick)
                left = self.cfg.capture.interval_s - (time.perf_counter() - t0)
                if left > 0 and (max_ticks is None or i + 1 < max_ticks):
                    time.sleep(left)
        except KeyboardInterrupt:
            self.log(f"{_ts()} 停止")
        finally:
            self.cap.close()
            if self._fh:
                self._fh.close()

    def _report(self, tick: Tick) -> None:
        c = tick.board.counts()
        stamp = f"{_ts()} [{tick.mode} {tick.elapsed_ms:.0f}ms]"
        if not self.calibrated:
            xs, ys = self.geometry.spacing
            self.log(
                f"{stamp} 标定 OK bbox={self.screen_bbox} 格距 {xs:.1f}x{ys:.1f}px "
                f"r={self.cfg.board.sample_radius}"
            )
            self.log(f"{stamp} 盘面 黑={c['X']} 白={c['O']}")
            self.log(tick.board.render())
            self.calibrated = True
        elif tick.changes:
            detail = " ".join(str(ch) for ch in tick.changes)
            self.log(
                f"{stamp} 落子 {len(tick.changes)} 手: {detail} 黑={c['X']} 白={c['O']}"
            )
            self.log(tick.board.render())
        if tick.reason:
            self.log(f"{stamp} {tick.reason}")
        elif tick.held:
            self.log(f"{stamp} 待复验 {len(tick.held)} 处: " + " ".join(str(ch) for ch in tick.held))
