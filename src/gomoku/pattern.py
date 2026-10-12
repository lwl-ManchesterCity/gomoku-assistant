"""棋型扫描：把 15×15 矩阵翻成确定性标签，喂给 explain 层，并在推荐出口前把关。

为什么要自己写：RapFi 只回一手棋加评分，不吐「活三 / 冲四」这类文字；而 explain
的契约是**模型禁止自己判断局面**（2026-09-10 A/B 实测它推荐过已占交叉点、
还凭空造出一条不共线的斜活二），所以棋型必须由确定性代码产出。

判据是一条自洽的阶梯，每一级都问「这一手落下去能得到什么」：

* 五连：盘上已有 5 颗同色子连成一线。
* 活四 / 冲四：数**补成五连的落点**。含缺口的棋型也正确 —— 只数连续同色子会把
  `黑黑白黑` 的两个断黑当成两气，从而完全看不见这手一步成五的杀。落点 ≥2 个
  必活（对手一手堵不完），恰好 1 个是冲四。
* 活三 / 眠三：对每个候选点试落一手，看新盘面的成五落点数 —— ≥2 个就是活三
  （下一手成活四），恰好 1 个是眠三（只做出冲四）。这样跳三 `X.XX`、抱三 `XX.X`
  和端点已被堵死的假活三都能自动分对，不需要为每种形状开一条分支。
* 活二 / 眠二：退回到形状判据 —— 恰长 2 的连子，两端空点 2 个记活二、1 个记眠二。
  再往下递归试落代价太大，而「二」本身不 forcing，只用于讲解。
* 一方只要出现了五连 / 活四 / 冲四，就**只报这一条**，不再列它的三和二。
  任何四都必然含着一堆三/二 子形（`XXXX.X` 既是冲四也含眠三），一并列出去
  就是让模型自己挑一条讲，正是 A/B 实测里它编造棋型的成因。四三杀的组合
  分量本来就在引擎的评分和 pv 里。

候选点只取紧邻（切比雪夫距离 ≤2）己方子的空点：造出活四/冲四的那一手必然贴着
己方棋型，超过这个距离落子不改变任何棋型的 forcing 等级。

坐标约定同 board.py：x=列（A..O），y=行（1..15，自上向下），**这里不做任何翻转**，
行号增大＝屏幕向下。explain 层的 prompt 必须照此措辞（用「行号增大方向」而不是
「上/下」），因为 DeepSeek 会按国际象棋惯例把行号大读成上方。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .board import BLACK, EMPTY, WHITE, Board, Change, coord_name

Coord = tuple[int, int]
DIRECTIONS = ((1, 0), (0, 1), (1, 1), (1, -1))
FIVE = 5
CANDIDATE_DIST = 2
MAX_CELLS = 6            # 单个标签最多列几个点，防止 prompt 被撑爆
SEVERITY = ("五连", "活四", "冲四", "活三", "眠三", "活二", "眠二")
FORCING_RANK = SEVERITY.index("活三")     # ≤ 这一级（五连/活四/冲四/活三）对手必须先应一手
LIVE4_RANK = SEVERITY.index("活四")       # ≤ 这一级（五连/活四）已经拦不住，不必再谈回防
FOUR_RANK = SEVERITY.index("冲四")        # ≤ 这一级就是「下一手即成五」，轮到对方时已经输了

# 每个标签一句定型措辞：落点写成「走 X」，现有连子写成「连子 X」，不留歧义。
_PHRASES = {
    "活四": lambda t: f"走 {t.names} 任一点即成五，对手一手堵不完",
    "冲四": lambda t: f"只有 {t.names} 这一点可成五，对手必须应",
    "活三": lambda t: f"走 {t.names} 任一点可成活四",
    "眠三": lambda t: f"走 {t.names} 任一点只能成冲四（成不了活四）",
    "活二": lambda t: f"连子 {t.names} 两端皆空，共 {t.count} 处",
    "眠二": lambda t: f"连子 {t.names} 只有一端可延伸，共 {t.count} 处",
}


@dataclass(frozen=True)
class Threat:
    """一方的一个棋型。

    `cells` 的含义分两类，措辞里必须写出来，否则模型会把落点读成已有的子
    （2026-09-10 A/B 实测它就是读串了坐标语义才推荐过已占交叉点）：

    * 五连 / 活四 / 冲四 / 活三 / 眠三：cells 是**空点（落点）**。
    * 活二 / 眠二：cells 是**已有的同色连子**。
    """

    colour: str
    label: str
    cells: tuple[Coord, ...]
    n: int = 0              # 该棋型共几处；0 = 与 cells 数一致

    @property
    def who(self) -> str:
        return "黑" if self.colour == BLACK else "白"

    @property
    def rank(self) -> int:
        return SEVERITY.index(self.label)

    @property
    def count(self) -> int:
        return self.n or len(self.cells)

    @property
    def names(self) -> str:
        shown = " ".join(coord_name(x, y) for x, y in self.cells)
        return shown + " 等" if self.count > len(self.cells) else shown

    def __str__(self) -> str:
        if not self.cells:
            return f"{self.who}已成五连"
        return f"{self.who}{self.label}，{_PHRASES[self.label](self)}"


def inside(board: Board, x: int, y: int) -> bool:
    return 0 <= x < board.size and 0 <= y < board.size


def _placed(board: Board, x: int, y: int, colour: str) -> Board:
    return board.with_changes([Change(x, y, EMPTY, colour)])


def _count(board: Board, x: int, y: int, dx: int, dy: int, colour: str) -> int:
    n = 0
    while inside(board, x, y) and board.cell(x, y) == colour:
        n += 1
        x += dx
        y += dy
    return n


def makes_five(board: Board, x: int, y: int, colour: str) -> bool:
    """在 (x,y) 落一颗 colour 子能否连成五（含中间有缺口的棋型）。"""
    return any(
        1 + _count(board, x + dx, y + dy, dx, dy, colour)
        + _count(board, x - dx, y - dy, -dx, -dy, colour) >= FIVE
        for dx, dy in DIRECTIONS
    )


def five_cells(board: Board, colour: str) -> tuple[Coord, ...]:
    """所有「落下去就成五」的空点（成五落点）。已有五连时返回 ()，由 five_on_board 负责。"""
    return tuple(
        (x, y)
        for y in range(board.size)
        for x in range(board.size)
        if board.cell(x, y) == EMPTY and makes_five(board, x, y, colour)
    )


def five_on_board(board: Board, colour: str) -> bool:
    """盘上是否已有五连 —— 有就是终局，再谈棋型没有意义。"""
    for y in range(board.size):
        for x in range(board.size):
            if board.cell(x, y) != colour:
                continue
            for dx, dy in DIRECTIONS:
                if not inside(board, x - dx, y - dy) or board.cell(x - dx, y - dy) != colour:
                    if _count(board, x, y, dx, dy, colour) >= FIVE:
                        return True
    return False


def candidates(board: Board, colour: str) -> tuple[Coord, ...]:
    """紧邻己方子的空点：一手能改变棋型 forcing 等级的落点全集。"""
    out: set[Coord] = set()
    for y in range(board.size):
        for x in range(board.size):
            if board.cell(x, y) != colour:
                continue
            for dy in range(-CANDIDATE_DIST, CANDIDATE_DIST + 1):
                for dx in range(-CANDIDATE_DIST, CANDIDATE_DIST + 1):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x + dx, y + dy
                    if inside(board, nx, ny) and board.cell(nx, ny) == EMPTY:
                        out.add((nx, ny))
    return tuple(sorted(out))


def _runs(board: Board, colour: str, length: int) -> list[tuple[Coord, ...]]:
    """恰为 `length` 长的极大连续同色子。"""
    out: list[tuple[Coord, ...]] = []
    for y in range(board.size):
        for x in range(board.size):
            if board.cell(x, y) != colour:
                continue
            for dx, dy in DIRECTIONS:
                if inside(board, x - dx, y - dy) and board.cell(x - dx, y - dy) == colour:
                    continue                      # 不是这条连子的起点
                if _count(board, x, y, dx, dy, colour) == length:
                    out.append(tuple((x + i * dx, y + i * dy) for i in range(length)))
    return out


def _open_ends(board: Board, run: tuple[Coord, ...]) -> int:
    """连子两端里在盘内且为空的点数。"""
    (x0, y0), (x1, y1) = run[0], run[-1]
    dx, dy = x1 - x0, y1 - y0                      # _runs 保证是单位步长
    return sum(
        1 for cx, cy in ((x0 - dx, y0 - dy), (x1 + dx, y1 + dy))
        if inside(board, cx, cy) and board.cell(cx, cy) == EMPTY
    )


@dataclass
class Scan:
    """一次扫描的缓存：全盘扫描不便宜，五连判定和成五落点各只算一遍。"""

    board: Board
    _won: dict[str, bool] = field(default_factory=dict)
    _fives: dict[str, tuple[Coord, ...]] = field(default_factory=dict)

    def won(self, colour: str) -> bool:
        if colour not in self._won:
            self._won[colour] = five_on_board(self.board, colour)
        return self._won[colour]

    def fives(self, colour: str) -> tuple[Coord, ...]:
        if colour not in self._fives:
            self._fives[colour] = () if self.won(colour) else five_cells(self.board, colour)
        return self._fives[colour]

    def four(self, colour: str) -> Threat | None:
        if self.won(colour):
            return Threat(colour, "五连", ())
        five = self.fives(colour)
        if len(five) >= 2:
            return Threat(colour, "活四", five[:MAX_CELLS], len(five))
        if len(five) == 1:
            return Threat(colour, "冲四", five)
        return None

    def three(self, colour: str) -> Threat | None:
        """试落每个候选点，按新盘面的成五落点数分活三 / 眠三。"""
        live: list[Coord] = []
        dead: list[Coord] = []
        for x, y in candidates(self.board, colour):
            if makes_five(self.board, x, y, colour):
                continue            # 成五点属于「四」级；落下去是六连，会把这里判成假活三
            made = self.after(colour, x, y)
            if made >= 2:
                live.append((x, y))
            elif made == 1:
                dead.append((x, y))
        if live:
            return Threat(colour, "活三", tuple(sorted(live))[:MAX_CELLS], len(live))
        if dead:
            return Threat(colour, "眠三", tuple(sorted(dead))[:MAX_CELLS], len(dead))
        return None

    def after(self, colour: str, x: int, y: int) -> int:
        """在 (x,y) 落子后，该方还剩几个成五落点。"""
        return len(five_cells(_placed(self.board, x, y, colour), colour))

    def two(self, colour: str) -> Threat | None:
        """形状判的「二」：恰长 2 的极大连子按两端空点数分类，cells 给第一处。"""
        runs = [(r, _open_ends(self.board, r)) for r in _runs(self.board, colour, 2)]
        for label, want in (("活二", 2), ("眠二", 1)):
            hit = [r for r, e in runs if e == want]
            if hit:
                return Threat(colour, label, hit[0], len(hit))
        return None

    def forcing(self, colour: str) -> Threat | None:
        """该方「必须被应一手」的棋型：五连 / 活四 / 冲四 / 活三；眠三及以下返回 None。

        给出口前的安全网用（`safety_note`）。没有四时才会退到 `three()`，那是本层最贵的一步
        （逐候选点试落再扫成五点），所以调用方只在确实需要时才问它。
        """
        t = self.four(colour) or self.three(colour)
        return t if t and t.rank <= FORCING_RANK else None

    def threats(self, colour: str) -> list[Threat]:
        four = self.four(colour)
        if four:
            return [four]           # 四一级已经强制对手应一手；它必然含一堆三/二 子形，列出来只是噪声
        return [t for t in (self.three(colour), self.two(colour)) if t]


def scan(board: Board, mover: str = BLACK) -> list[Threat]:
    """双方棋型，按严重度排序；同级里走子方在前 —— prompt 关心的是「该谁应」。"""
    s = Scan(board)
    out = [*s.threats(BLACK), *s.threats(WHITE)]
    out.sort(key=lambda t: (t.rank, t.colour != mover))
    return out


def _rank(threat: Threat | None) -> int:
    """没有棋型时给一个「比眠二还轻」的等级，让比较能写成一条不等式。"""
    return threat.rank if threat else len(SEVERITY)


def _residual(after: Scan, mover: str, opp: str) -> Threat | None:
    """落完这一手，对方是否还握着「下一手就成五」的点。我方直接成五（先赢）不算。"""
    if after.won(mover):
        return None
    t = after.four(opp)
    return t if t is not None and t.rank <= FOUR_RANK else None


def hopeless(board: Board, mover: str, move: Coord) -> Threat | None:
    """这一手走下去已经堵不住（轮到对方，对方下一手即成五），返回那条拦不住的威胁。

    2026-09-28 实盘 21:21 那一帧：白方斜活四（K3/F8 两端全空），黑堵 K3 只把活四降级成
    冲四，`safety_note` 原来的判据是「等级降了就算在处理」，于是出口全静默，用户看到的
    是一行普通的 `→ 你的下一手 K3`。三及以下还有得应，四及以下没有：那一点在对方手里。
    """
    return _residual(Scan(_placed(board, move[0], move[1], mover)), mover,
                     WHITE if mover == BLACK else BLACK)


def safety_note(board: Board, mover: str, move: Coord) -> str | None:
    """出口前的战术安全网：这一手到底有没有把对方的先手处理掉。没问题返回 None。

    来历是 2026-09-11 16:07 实盘被秒那一手：盘面 黑 H8 I7 D8 / 白 F6 F7 F8，白方是活三
    （走 F5 F9 任一点成活四），引擎回 E7 —— 两边都没堵，下一拍白 F9 成活四就终局。
    引擎那条回答来自被旧 pv 污染的增量通道，评分还是「必胜(余11)」。**所以这里刻意不看评分**：
    要防的正是「引擎说自己必胜、其实被一手杀」，信它的分就等于不信这道网。

    判据只有一条自洽的阶梯：落自己一颗子只能占掉对方的空点，**不可能让对方的强制等级变高**，
    所以「落子后对方的 forcing 等级没降」就等于「这手没在处理先手」。例外是我方反将 ——
    这一手**做出**了五连/活四/冲四，对手得先应我们，回防就不急了；自家四早就摆在那儿、
    这手却不去用，不算反将（那样只是把两条先手一起败掉）。
    在这条之前还有一道更硬的闸（`hopeless`）：落完之后对方仍有四一级的，等级降了也没用，
    那一手不是「在处理」而是「在处理中输掉」，必须照说。
    """
    opp = WHITE if mover == BLACK else BLACK
    was = Scan(board)
    before = was.forcing(opp)
    if before is None:
        return None                     # 对方本来就没有先手，谈不上应不应
    now = Scan(_placed(board, move[0], move[1], mover))
    if now.won(mover):
        return None                     # 这一手直接成五，终局都到了，不必再谈回防
    dead = _residual(now, mover, opp)
    if dead is not None:
        尾 = "盘上对方已连五" if dead.rank == 0 else "下一手是对方的成五点"
        return f"{coord_name(*move)} 堵不住：{dead}；{尾}"
    after = now.forcing(opp)
    if after is None or after.rank > before.rank:
        return None                     # 等级降下来甚至清掉了，这手确实在处理
    mine = now.four(mover)
    if mine and (mine.rank <= LIVE4_RANK or mine.rank < _rank(was.four(mover))):
        return None                     # 一手做成四/成五，攻其必救
    return f"{coord_name(*move)} 没处理对方先手：{after}；这一手必须占它的落点"


def answer_points(board: Board, mover: str) -> str | None:
    """对方先手必须被占掉的点。forcing 级棋型的 `cells` 全是落点，占掉一个才算应。

    给出口用：`safety_note` 报警时把那几个坐标单独印一行，不让用户自己去 ⚠ 那段
    文字里挑。对方没有先手、或已经是活四及以上（一手堵不住）时返回 None —— 报点反而误导。
    """
    opp = WHITE if mover == BLACK else BLACK
    t = Scan(board).forcing(opp)
    if t is None or not t.cells or t.rank <= LIVE4_RANK:
        return None
    return t.names


def describe(board: Board, mover: str = BLACK) -> str:
    """给 prompt 用的一行式棋型摘要。无棋型时明确说「无」，不能让模型自己填。"""
    ts = scan(board, mover)
    if not ts:
        return "双方均无成活、成四或成三的棋型。"
    by: dict[str, list[str]] = {}
    for t in ts:
        by.setdefault(t.who, []).append(str(t).removeprefix(t.who))
    return "；".join(f"{who}方：" + "；".join(v) for who, v in by.items())
