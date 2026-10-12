"""讲解层（explain）：把确定性字段交给 LLM 翻成教学语言。

三条不可让的设计约束，全都来自 2026-09-10 那次同局面 A/B 实测：

1. **模型禁止判断局面**。棋型、候选点、评分一律由 pattern.py / engine.py 算好，
   以文字字段喂进来。实测里它把带标记的白子说成用户刚落的黑子、宣称存在竖直活三
   （其正下方就是白子）、还把推荐点落在已被占据的交叉点上；喂文字棋谱时判对 3/4 条线，
   却仍凭空造出一条 Δ列=1、Δ行=2 根本不共线的「斜线活二」。这些错误全是确定性代码
   零成本能消除的，所以不留给它自由推理。
2. **措辞里不出现上下左右**。模型默认按国际象棋惯例把行号大读成「上方」，把我给的
   「行 1–15（上→下）」系统性读反了。这里只说「行号增大方向」「列标签增大方向」，
   并要求它引用坐标时原样照抄。
3. **永不阻塞、永不改变推荐**。请求在后台线程里发，带硬超时；失败只打一行降级提示。
   同时提交的新请求会挤掉排队中的旧请求，回来的答案若已过期（盘又变了）直接丢弃。

只用 stdlib：本机 PyPI 与镜像都不通，`requests` / `httpx` 装不上。
密钥只从环境变量或仓库根 `.env` 读，**不写进日志、不进异常信息、不进 prompt**。
"""

from __future__ import annotations

import json
import os
import queue
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from .board import BLACK, EMPTY, WHITE, Board, Change, coord_name
from .config import Config
from .engine import MATE, Plan
from .pattern import describe

ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = ROOT / ".env"

WHO_TEXT = {BLACK: "黑", WHITE: "白"}

class ExplainUnavailable(RuntimeError):
    pass


def read_key(env_name: str) -> str | None:
    """环境变量优先，其次仓库根 `.env`。返回 None 表示未配置，调用方须整体降级。"""
    value = os.environ.get(env_name, "").strip()
    if value:
        return value
    if not ENV_FILE.exists():
        return None
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("#") or "=" not in line:
            continue
        name, _, got = line.partition("=")
        if name.strip() == env_name:
            return got.strip().strip('"').strip("'") or None
    return None


def format_eval(score: float | None) -> str:
    """评分换算成走子方视角的文字。引擎的 Eval 以**它自己那一方**为正。

    杀棋分写成「引擎判…」而非直接下结论：M 分只在引擎的搜索深度内成立，
    而且实战还要看用户是否走出这步，不能完全交给模型去断言。
    """
    if score is None:
        return "引擎未给评分（实测出现在一步制胜或开局库命中时）"
    if abs(score) >= MATE - 1000:
        steps = MATE - abs(score)
        return f"引擎判{'走子方' if score > 0 else '对手'}必胜，其搜索内已证再 {steps} 手成五"
    return f"以走子方为正的评分 {score:+.0f}"


def stones_line(board: Board) -> str:
    parts = []
    for colour, who in ((BLACK, "黑"), (WHITE, "白")):
        names = " ".join(
            coord_name(x, y)
            for y in range(board.size)
            for x in range(board.size)
            if board.cell(x, y) == colour
        )
        parts.append(f"{who}: {names or '无子'}")
    return "；".join(parts)


def build_prompt(board: Board, plan: Plan, mover: str, lessons: str = "") -> str:
    """拼出只含既定事实的 prompt。mover = 引擎这次回答的是哪一方。

    棋型给**两份**：走之前，以及走完建议手之后。只给前者却问「这一步属于什么
    性质」，模型就只能凭空造一条 —— A/B 实测里那条不共线的「斜线活二」就是这么来的。

    `lessons` 是上一局写进经验文档的最近几条（`lessons.for_prompt`），空串就不拼这一段。
    它只进讲解、不进选点：文档里那句「原因」是模型自己写的，拿去改落点等于把一次
    误判永久装进决策链。
    """
    who = "黑" if mover == BLACK else "白"
    other = WHITE if mover == BLACK else BLACK
    after = board.with_changes([Change(plan.move[0], plan.move[1], EMPTY, mover)])
    depth = f"搜索深度 {plan.depth}。" if plan.depth is not None else "本次未回进度行（实测出现在一步制胜或开局库命中）。"
    # pv 只有引擎给出变化线时才带依据；实测它在「一手制胜」那类回手里整条不回。
    # 判据只能有一处：09-28 那局出现过 pv=("J4",) 这种只回推荐点自身的回手，当时
    # `line` 按「pv 非空」印成「主要变化线: J4。」、`reply` 按「不足两手」说「本次没有
    # 给出变化线」，同一段里自相矛盾，模型无从照抄。
    有线 = len(plan.pv) >= 2
    reply = (f"最后一句照抄主要变化线里 {plan.pv[1]} 这一步说明对手最可能的应手。"
             if 有线 else
             "本次没有给出变化线，所以**不要猜测对手会应哪一手**，只说明对手必须面对什么威胁。")
    line = f"主要变化线: {' '.join(plan.pv[:6])}。" if 有线 else "未给出变化线。"
    history = f"{lessons}\n\n" if lessons else ""
    return f"""你是五子棋教练，为用户讲解引擎给出的这一步棋。

【坐标约定】列用字母 A..O 自左向右；行用数字 1..15，行号沿屏幕向下增大。
描述位置时只准用「行号增大方向 / 行号减小方向 / 列标签增大方向」这类说法，
禁止使用「上方 / 下方 / 上面 / 下面」，因为你对行号方向的直觉与本项目相反。
引用任何交叉点必须原样照抄下面给出的坐标名，不得改写、不得推算新坐标。

【铁律】你不得自行判断局面。棋型、候选点、评分全部由下面的确定性程序算出并已给定，
你只能引用它们。凡下面没写明的棋型，一律视为不存在；不要补充任何你没被给的判断。

【走之前的盘面】共 {len(board.stones())} 子
{stones_line(board)}
棋型: {describe(board, mover)}

【走 {plan.name} 之后的盘面】共 {len(after.stones())} 子
{stones_line(after)}
棋型: {describe(after, other)}

【引擎结论】轮到 {who} 走，建议 {plan.name}。{format_eval(plan.score)}。{depth} {line}

{history}【任务】用 100 到 180 个中文字讲清这一步：先说它把局面推进成上面【走之后】列出的哪种
棋型，再说它给对手留下什么难题；接着说不走这一点会怎样 —— 这一句的依据只能是上面
【走之前的盘面】那一行「棋型」里列出的**对手**落点；那行没有对手的棋型就照说「对手当下没有成型的先手」，
不得编造上面没列出的威胁。{reply}
只准引用上面出现过的坐标和棋型。不加标题、不分点、不客套。"""


def build_review_prompt(game) -> str:
    """本局复盘的 prompt：事实全部来自 `lessons.round_facts`，模型只负责写成一句「原因」。

    和讲解层同一条铁律，只是这里更要紧 —— 产出的那句原因会被写进经验文档、下次启动
    再喂回模型。所以刻意不给评分和深度（2026-09-28 实盘：一手就能成五的点被判成
    `-999982`，那种数字一进复盘就会写出「引擎说必胜时别防」这种反向教训），
    也不许它自己下棋型结论或给新落点：它只能复述上面列过的坐标和判词。
    """
    逐手 = "\n".join(f"- {行}" for 行 in game.fact_lines()) or "- 没有一手参与对照（日志里对不出局面）"
    漏 = game.deviations
    没堵 = game.misses
    return f"""你是五子棋教练，为一整局棋写一条复盘经验。

【坐标约定】列 A..O 自左向右；行 1..15，行号沿屏幕向下增大。禁止使用「上方/下方」。
引用交叉点必须原样照抄下面给出的坐标名，不得推算新坐标。

【本局结果】我方执{WHO_TEXT[game.mine]}，{game.moves} 子，判为{game.result}。
偏离引擎首选 {len(漏)} 处：{' '.join(f"{t.play}(首选 {t.pick})" for t in 漏) or "无"}。
该堵没堵 {len(没堵)} 处：{' '.join(t.play for t in 没堵) or "无"}。

【逐手事实】由确定性程序算出，你只能引用，不得新增任何判断：
{逐手}

【任务】写 80 到 150 个中文字的一条经验：说明本局{game.result}的关键发生在哪一手或哪几手，
以及下次遇到同样形状时要注意什么。只准引用上面出现过的坐标与判词；
不得引用评分、搜索深度或变化线（上面没给）；不得断言上面没写出的棋型；不得提出新的落点建议。
最后一行只输出这条经验本身，不加标题、不分点、不客套。"""


WHO_TEXT = {BLACK: "黑", WHITE: "白"}


def request_chat(cfg: Config, key: str, prompt: str) -> str:
    """一次 OpenAI 兼容的 chat/completions 调用。异常信息里绝不带密钥。"""
    llm = cfg.llm
    body = json.dumps({
        "model": llm.model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": llm.max_tokens,
        "temperature": 0.4,
        "stream": False,
        # v4 系是 reasoning 模型：不关推理时 max_tokens 会被 reasoning_content 整预算吃光，
        # content 直接回空串（实测 2000 token 全花在推理上）。本层契约本来就禁止它判断局面。
        "thinking": {"type": "disabled"},
    }).encode()
    req = urllib.request.Request(
        llm.base_url.rstrip("/") + "/chat/completions",
        data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=llm.timeout_ms / 1000) as resp:
            payload = json.load(resp)
    except urllib.error.HTTPError as exc:
        # 只取状态码和一小段响应正文；不把 header 或 URL 查询串带出去。
        detail = ""
        try:
            detail = exc.read(400).decode("utf-8", "replace")
        except Exception:
            pass
        raise ExplainUnavailable(f"HTTP {exc.code} {detail}".strip()) from None
    except urllib.error.URLError as exc:
        raise ExplainUnavailable(f"网络不可达: {exc.reason}") from None
    except (TimeoutError, json.JSONDecodeError) as exc:
        raise ExplainUnavailable(f"响应异常: {type(exc).__name__}") from None
    try:
        text = (payload["choices"][0]["message"]["content"] or "").strip()
    except (KeyError, IndexError, TypeError, AttributeError):
        raise ExplainUnavailable(f"响应结构不认识: {str(payload)[:200]}") from None
    if not text:
        raise ExplainUnavailable("回话为空（reasoning 模型把预算全花在推理上时会这样）")
    return text


@dataclass
class Explainer:
    """后台单线程问答：最新请求挤掉排队中的旧请求，过期答案直接丢。"""

    cfg: Config
    on_text: object = None            # callable(str) -> None
    key: str | None = None
    enabled: bool = True
    _jobs: "queue.Queue[tuple[int, str]]" = field(default=None, repr=False)
    _thread: threading.Thread | None = field(default=None, repr=False)
    _gen: int = field(default=0, repr=False)
    _warned: str = field(default="", repr=False)

    def __post_init__(self):
        if self.key is None:
            self.key = read_key(self.cfg.llm.api_key_env)
        self._jobs = queue.Queue()

    @property
    def available(self) -> bool:
        return self.enabled and bool(self.key)

    def submit(self, prompt: "str | Callable[[], str]") -> None:
        """投一次讲解请求。不可用就静默返回；排队中的旧请求一律作废。

        `prompt` 允许是零参回调，那时拼装在 `_work` 里做：`build_prompt` 实测 148ms，
        而调用方正是引擎 worker 线程，那 148ms 就是把下一次提问往后推 148ms；
        本层一次请求要 1s，挪过来等于白赚。
        """
        if not self.available:
            self._notify("讲解已关闭" if not self.enabled else "讲解降级: 缺密钥")
            return
        while not self._jobs.empty():
            try:
                self._jobs.get_nowait()
            except queue.Empty:
                break
        self._gen += 1
        self._jobs.put((self._gen, prompt))
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._work, daemon=True)
            self._thread.start()

    def _work(self) -> None:
        while True:
            try:
                gen, prompt = self._jobs.get(timeout=0.5)
            except queue.Empty:
                return                      # 队列空了就退出，下次 submit 再起线程
            if gen != self._gen:
                continue
            t0 = time.perf_counter()
            try:
                text = request_chat(self.cfg, self.key,
                                    prompt() if callable(prompt) else prompt)
            except ExplainUnavailable as exc:
                self._notify(f"讲解降级: {exc}")
                continue
            except Exception as exc:          # 兜底：讲解层绝不能把主循环带崩
                self._notify(f"讲解降级: {type(exc).__name__}")
                continue
            if gen != self._gen:
                self._notify("讲解结果已过期（盘面又变了），丢弃")
                continue
            if self.on_text:
                self.on_text(f"[{time.perf_counter() - t0:.1f}s] {text}")

    def _notify(self, note: str) -> None:
        """同类提示只说一次，避免降级信息在 300ms 一拍里刷屏。"""
        if note != self._warned and self.on_text:
            self._warned = note
            self.on_text(note)

    def close(self) -> None:
        self._gen += 1                        # 让在途结果作废
        self._jobs.put((-1, ""))              # 唤醒可能阻塞在 get 上的线程
        thread, self._thread = self._thread, None
        if thread and thread.is_alive():
            thread.join(timeout=self.cfg.llm.timeout_ms / 1000 + 1)
