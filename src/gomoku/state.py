"""状态同步：把每拍的原始读数合成可信的逻辑局面，喂给引擎层。

三条从实测来的约束，决定了这里为什么长成这样：

1. **一拍可能有多手。** 2026-09-10 的实盘日志里 14:11:07.218 那拍同时出现
   ``F9:.->X`` 和 ``C12:.->O``（302ms 的采样窗里对手抢了一手）。所以绝不假设双方
   轮流落子，执子色只能取自像素分类结果，不能由棋盘奇偶反推。
2. **落子动画没有可观察的中间帧**（2026-09-11 用 ~152ms 采样跑 563 拍 / 40 颗子实测：
   40/40 都是 "." 一帧变实心盘，「出现又消失」0 处），所以 `confirm_ticks` 默认 1 = 读到即提交。
   这套「连续 N 拍同色才提交」的机制保留着，把配置调回 2 就启用：颜色一改就重新计数。
   它换不来任何过滤（两份样本里 1 拍、2 拍的短暂读数各 0 处），只多花一拍延时，
   而那一拍正压在人机落子→建议出口那条时效链上。
3. **单拍大面积变化一定不是棋。** 正常一手只动 1 个点，超过阈值的整拍丢弃，
   那是标定漂移或半过渡帧。但若同一异常局面连续多拍稳定不变，说明是真变化
   （重开一局、悔棋），此时接受它作为新基准，否则会永久卡死。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .board import Board, Change, EMPTY
from .config import Config


def empty_board(size: int) -> Board:
    return Board(tuple(tuple(EMPTY for _ in range(size)) for _ in range(size)))


@dataclass(frozen=True)
class Sync:
    board: Board               # 提交后的可信局面
    changes: list[Change]      # 本拍确认提交的变化
    held: list[Change]         # 本拍读到但尚未确认，等下一拍复验
    reason: str | None = None  # 非空表示本拍被整拍丢弃或发生了重同步


@dataclass
class Tracker:
    """有状态的一拍一进：raw Board → 可信 Board + 本拍提交的变化。"""

    cfg: Config
    board: Board = field(default=None, repr=False)
    _pending: dict[tuple[int, int], tuple[str, int]] = field(default_factory=dict, repr=False)
    _streak_sig: tuple[str, ...] | None = field(default=None, repr=False)
    _streak: int = 0
    _primed: bool = False

    def __post_init__(self):
        if self.board is None:
            self.board = empty_board(self.cfg.board.size)

    def update(self, raw: Board) -> Sync:
        # 首次读数直接采纳：刚锁定时盘面是静止的，没有中间帧可防，
        # 否则开局已有十几子的局面会撞 max_changes、白等 resync_ticks 拍。
        if not self._primed:
            self._primed = True
            self.board = raw
            return Sync(board=raw, changes=[], held=[])

        st = self.cfg.state
        changes = self.board.diff(raw)
        if not changes:
            self._streak_sig, self._streak = None, 0
            self._pending.clear()
            return Sync(board=self.board, changes=[], held=[])

        if len(changes) > st.max_changes:
            return self._on_bulk_change(raw, changes, st.resync_ticks)

        self._streak_sig, self._streak = None, 0
        committed: list[Change] = []
        held: list[Change] = []
        seen: set[tuple[int, int]] = set()
        for ch in changes:
            key = (ch.x, ch.y)
            seen.add(key)
            code, n = self._pending.get(key, (None, 0))
            n = n + 1 if code == ch.after else 1
            self._pending[key] = (ch.after, n)
            if n >= st.confirm_ticks:
                committed.append(ch)
                del self._pending[key]
            else:
                held.append(ch)
        # 上一拍的候选这一拍没再读到 = 中间帧已消失，撤销候选
        for key in self._pending.keys() - seen:
            del self._pending[key]

        if committed:
            self.board = self.board.with_changes(committed)
        return Sync(board=self.board, changes=committed, held=held)

    def _on_bulk_change(self, raw: Board, changes: list[Change], resync_ticks: int) -> Sync:
        sig = tuple(cell for row in raw.grid for cell in row)
        self._streak = self._streak + 1 if sig == self._streak_sig else 1
        self._streak_sig = sig
        self._pending.clear()
        if self._streak < resync_ticks:
            return Sync(
                board=self.board,
                changes=[],
                held=changes,
                reason=(
                    f"单拍 {len(changes)} 处变化 > {self.cfg.state.max_changes}"
                    f"，整拍丢弃（连续第 {self._streak} 拍）"
                ),
            )
        self._streak_sig, self._streak = None, 0
        self.board = raw
        return Sync(
            board=raw,
            changes=changes,
            held=[],
            reason=f"异常局面连续 {resync_ticks} 拍稳定，按新基准重同步",
        )


