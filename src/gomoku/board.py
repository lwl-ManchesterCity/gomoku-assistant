"""棋盘状态：各模块之间唯一的接口。

屏幕坐标约定：x 为列（0..14，标签 A..O 自左向右），y 为行（0..14，标签 1..15 自上向下）。
接入 Piskvork 引擎时行序要翻转为自下向上，那一层的转换只允许写在引擎适配里。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

COLUMN_LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

EMPTY = "."
BLACK = "X"
WHITE = "O"


def coord_name(x: int, y: int) -> str:
    return f"{COLUMN_LABELS[x]}{y + 1}"


COLOUR_WORDS = {
    "black": BLACK, "b": BLACK, "黑": BLACK, "黑棋": BLACK, "执黑": BLACK,
    "white": WHITE, "w": WHITE, "白": WHITE, "白棋": WHITE, "执白": WHITE,
}


def parse_colour(text: str) -> str:
    """把用户嘴里的颜色换成内部码。认不出就报错，绝不在这里猜。"""
    key = text.strip().lower()
    if key not in COLOUR_WORDS:
        raise ValueError(f"认不出颜色 {text!r}，只认 黑/b/black 或 白/w/white")
    return COLOUR_WORDS[key]


@dataclass(frozen=True)
class Change:
    x: int
    y: int
    before: str
    after: str

    @property
    def name(self) -> str:
        return coord_name(self.x, self.y)

    def __str__(self) -> str:
        return f"{self.name}:{self.before}->{self.after}"


@dataclass(frozen=True)
class Board:
    grid: tuple[tuple[str, ...], ...]

    @property
    def size(self) -> int:
        return len(self.grid)

    def cell(self, x: int, y: int) -> str:
        return self.grid[y][x]

    def stones(self) -> dict[str, str]:
        """{坐标名: 颜色}，只含非空点。"""
        return {
            coord_name(x, y): self.grid[y][x]
            for y in range(self.size)
            for x in range(self.size)
            if self.grid[y][x] != EMPTY
        }

    def counts(self) -> dict[str, int]:
        flat = "".join("".join(row) for row in self.grid)
        return {BLACK: flat.count(BLACK), WHITE: flat.count(WHITE), EMPTY: flat.count(EMPTY)}

    def side_to_move(self) -> str:
        """按「黑先、严格交替」推出该谁走：子少的一方走，平子时轮到黑。

        这只是约定推论，**不能反过来用它认定「用户是哪一方」**——那是只有用户知道的
        事实（见 advisor.question_position）。两局实盘都栽在这个推断上。
        """
        c = self.counts()
        return WHITE if c[BLACK] > c[WHITE] else BLACK

    def diff(self, other: "Board") -> list[Change]:
        if self.size != other.size:
            raise ValueError(f"棋盘规格不同: {self.size} vs {other.size}")
        return [
            Change(x, y, self.grid[y][x], other.grid[y][x])
            for y in range(self.size)
            for x in range(self.size)
            if self.grid[y][x] != other.grid[y][x]
        ]

    def with_changes(self, changes: Iterable[Change]) -> "Board":
        """返回套用这些变化后的新盘面（自身不变）。"""
        rows = [list(row) for row in self.grid]
        for ch in changes:
            rows[ch.y][ch.x] = ch.after
        return Board(tuple(tuple(row) for row in rows))

    def render(self, show_labels: bool = True) -> str:
        header = "   " + " ".join(COLUMN_LABELS[: self.size]) if show_labels else ""
        body = "\n".join(
            (f"{y + 1:>2} " if show_labels else "") + " ".join(row)
            for y, row in enumerate(self.grid)
        )
        return f"{header}\n{body}" if header else body

    def __str__(self) -> str:  # pragma: no cover - 便于调试
        return self.render()
