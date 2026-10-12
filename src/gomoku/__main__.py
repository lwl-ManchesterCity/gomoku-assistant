from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

from . import config
from .advisor import Advisor, Endgame, question_position, worth_asking
from .board import BLACK, WHITE, Board, Change, parse_colour
from .calibrate import CalibrationError, detect_board
from .config import Config
from .engine import MATE, EngineError, Plan, Rapfi
from .explain import (Explainer, ExplainUnavailable, WHO_TEXT, build_prompt,
                      build_review_prompt, read_key, request_chat)
from .lessons import (DEFAULT_DOC, Game, Lesson, append, default_at, for_prompt)
from .lessons import load as load_lessons
from .lessons import latest_log, read_frames, result_of, round_facts, todo_count
from .loop import Watcher
from .pattern import answer_points, five_on_board, hopeless, safety_note
from .vision import diagnostics, parse_board


LOGS = config.DEFAULT_PATH.parents[1] / "logs"
RUNS = LOGS / "runs"
REVIEW_ENGINE_MS = 6000      # 复盘那一支单独放宽：实盘 1.5s 是为时效，离线问没有这个约束


def run_log(now: datetime | None = None) -> str:
    """一局一个日志文件，时间戳写进文件名，所以按名字排就是按时间排（`latest_log` 靠这条）。

    默认不再写 `logs/events.jsonl`：那份 09-10 的 85 行样本是唯一还能重放的旧证据，
    而 `Watcher` 是 append 打开的，一旦谁忘了带 `--log /tmp/...` 就把它污染了。
    """
    t = now or datetime.now()
    return str(RUNS / f"advise-{t.strftime('%Y-%m-%d-%H%M%S')}.jsonl")


def load_image(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def _analyze(cfg: Config, rgb: np.ndarray):
    geom = detect_board(rgb, cfg)
    return geom, parse_board(rgb, geom, cfg)


def cmd_parse(args) -> int:
    cfg = config.load(Path(args.config) if args.config else None)
    rgb = load_image(Path(args.image))
    try:
        geom, board = _analyze(cfg, rgb)
    except CalibrationError as exc:
        print(f"标定失败: {exc}", file=sys.stderr)
        return 2

    c = board.counts()
    xs, ys = geom.spacing
    print(f"图像 {rgb.shape[1]}x{rgb.shape[0]}px  棋盘面 bbox={geom.bbox}  格距 {xs:.2f}x{ys:.2f}px")
    print(f"黑={c['X']} 白={c['O']} 空={c['.']}  子位: {sorted(board.stones())}")
    print(board.render())

    if args.stats:
        rows = diagnostics(rgb, geom, cfg)
        stones = [r for r in rows if r["code"] != "."]
        empties = np.array([r["lum"] for r in rows if r["code"] == "."])
        print("\n非空点读数:")
        for r in stones:
            print(
                f"  {r['name']:>3} px={r['px']} lum={r['lum']:.0f} warmth={r['warmth']:.0f} -> {r['code']}"
            )
        p = np.percentile(empties, [0, 1, 50, 99, 100])
        print(
            f"\n空格亮度 n={len(empties)} min={p[0]:.0f} p1={p[1]:.0f} 中位={p[2]:.0f} "
            f"p99={p[3]:.0f} max={p[4]:.0f}"
        )
        print(
            f"判据: 黑 lum<={cfg.board.black_max_lum}  白 lum>={cfg.board.white_min_lum}  "
            f"暖度>={cfg.board.max_warmth} 判空（星位）"
        )
    if args.json:
        Path(args.json).write_text(
            json.dumps(
                {
                    "bbox": geom.bbox,
                    "spacing": [xs, ys],
                    "grid": ["".join(row) for row in board.grid],
                },
                ensure_ascii=False,
                indent=1,
            )
        )
        print(f"写出 {args.json}")
    return 0


def resolve_my_colour(setting: str, flag: str | None) -> str:
    """开局必须问清用户执哪一方。命令行优先于配置，两处都是 auto 才开口问。

    为什么不让程序自己推：两局实盘都按「石子奇偶」反推过，两次都推反 —— App 把人机的子
    压到用户下一次点击才画出来，采样拿到的盘面永远是「黑=白+1」这一个形状，
    看不出坐在对面的是谁。
    """
    for token in (flag, setting):
        if token and token.strip().lower() != "auto":
            return parse_colour(token)
    print("开局确认一次：你执黑还是执白？（黑棋先走）")
    while True:
        try:
            return parse_colour(input("  黑 / 白 > "))
        except ValueError as exc:
            print(f"  {exc}")
        except (EOFError, KeyboardInterrupt):
            print()
            raise SystemExit(1)


def cmd_advise(args) -> int:
    cfg = config.load(Path(args.config) if args.config else None)
    mine = resolve_my_colour(cfg.play.my_colour, args.me)
    日志 = None if args.no_log else (args.log or run_log())
    print(f"你执{'黑' if mine == BLACK else '白'}，只把轮到你走的局面送引擎")
    print("提示：读到对手落子的那一拍就直接问，回的是「你的下一手」；"
          "只有当你这一手和对手的应手挤在同一拍落地时，才会退成复盘。")
    print(f"逐拍日志: {日志 or '关（--no-log，这一局事后 review 不了）'}")
    try:
        engine = Rapfi(cfg).open()
    except EngineError as exc:
        print(f"引擎启动失败: {exc}", file=sys.stderr)
        return 2

    def say(text: str) -> None:
        print(f"  讲解| {text}", flush=True)

    explainer = Explainer(cfg, on_text=say, enabled=not args.no_explain)
    state = ("已关闭" if args.no_explain
             else f"已启用 {cfg.llm.model}" if explainer.available
             else "未启用（缺密钥）")
    print(f"讲解层: {state}")

    经验 = load_lessons(Path(args.lessons))
    段 = for_prompt(经验)
    print(f"经验层: 从 {args.lessons} 读回 {len(经验)} 条，"
          f"最近 {min(len(经验), 3)} 条随讲解一起发出"
          + ("" if 段 else "（文档为空，本局不带历史）"))

    def on_plan(board, plan, mover, rolled) -> None:
        for 行 in advise_lines(board, plan, mover, mine, rolled, 当前盘):
            print(行, flush=True)
        if mover == mine:
            explainer.submit(lambda: build_prompt(board, plan, mover, 段))

    advisor = Advisor(
        engine, on_plan=on_plan, on_note=lambda s: print(f"  {s}", flush=True)
    )

    gate = Endgame()
    上一盘: Board | None = None
    当前盘: Board | None = None

    def advise(tick) -> None:
        nonlocal 上一盘, 当前盘
        当前盘 = tick.board
        if not worth_asking(tick.board, tick.changes, 上一盘):
            return                       # 消抖后没有新落子，也不是刚锁定的局面
        上一盘 = tick.board
        该问, 胜方 = gate.admit(tick.board)
        if not 该问:
            if 胜方:
                print(f"→ 本局已结束：{'黑' if 胜方 == BLACK else '白'}方连五，不再问引擎",
                      flush=True)
            return
        pos, 撤 = question_position(tick.board, tick.changes, mine)
        if pos is None:
            return                       # 撤不出「轮到你」的那个局面，问了也是白问
        advisor.submit(pos, 撤)   # 采集线程只递交局面，引擎在 worker 里想

    watcher = Watcher(
        cfg=cfg, use_roi=not args.full_screen, on_tick=advise, events_path=日志
    )
    try:
        watcher.run()
    finally:
        advisor.close()
        explainer.close()
        engine.close()
    if 日志:
        print(f"这一局的日志: {日志} —— 下一步直接跑 gomoku review 就会读它")
    return 0


def _eval_text(score: float | None) -> str:
    """杀棋分打成「必胜/必败（余 N 手）」，裸的 999997 对用户没有信息量。"""
    if score is None:
        return "-"
    if abs(score) >= MATE - 1000:
        return f"{'必胜' if score > 0 else '必败'}(余{MATE - int(abs(score))})"
    return f"{score:.0f}"


def advise_lines(board: Board, plan: Plan, mover: str, mine: str,
                 rolled: Change | None, 当前盘: Board | None) -> list[str]:
    """一次回手要顶格印的哪几行。只依赖盘面和回手，不碰引擎也不碰讲解层，所以能离线测。

    抽出来是因为 2026-09-28 那局的三处出口静默全都出在这段判断本身：它原先嵌在
    `cmd_advise` 的闭包里，测试要跑它就得起真引擎真屏幕。
    """
    if mover != mine:
        # 走到这里说明颜色声明或奇偶判断出了问题，宁可不说也不能说错一方的一手
        return [f"→ 这一拍引擎回答的是{'黑' if mover == BLACK else '白'}方 {plan.name}，"
                f"不是你的回合，本拍无建议"]
    c = board.counts()
    盘 = f"盘{c[BLACK]}:{c[WHITE]}"

    def 把关(落点: tuple[int, int]) -> list[str]:
        note = safety_note(board, mover, 落点)
        if not note:
            return []
        行 = [f"  ⚠ {note}"]
        pts = answer_points(board, mover)
        if pts:
            行.append(f"  → 必堵 {pts}")
        return 行

    if plan.depth is not None:
        detail = (f"（{plan.elapsed_ms:.0f}ms {盘}） "
                  f"depth={plan.depth} eval={_eval_text(plan.score)}")
        if plan.pv:
            detail += " pv=" + " ".join(plan.pv[:6])
    else:
        # 引擎没回进度行时不猜性质：实测同一种「零进度直接回手」既出现在一手制胜上，
        # 也出现在 2026-09-28 那局已经堵不住的位置上，写成一种就双向骗人。
        尾 = ("棋型层判已无解" if hopeless(board, mover, plan.move)
              else "未给进度行（一手制胜或开局库）")
        detail = f"（{plan.elapsed_ms:.0f}ms {盘}） {尾}"
    # 撤过手的那一盘是你点下去之前的局面，说成「下一手」就会指着你自己刚占的点
    if rolled is None:
        行 = [f"→ 你的下一手 {plan.name}{detail}"]
    elif plan.name == rolled.name:
        行 = [f"→ 复核：你落的 {rolled.name} 正是引擎首选{detail}"]
    else:
        # 复盘这一支是整条出口里唯一看得见「用户实际落点」的时刻。2026-09-28 那局
        # 41 问只偏了一次（建议 J4、落 J5），那一次把白方活三放成活四就直接终局，
        # 而安全网当时只审引擎推荐的那一手，整局 ⚠ 与必堵两行都是 0 次。
        行 = [f"→ 复盘：你落 {rolled.name} 之前那一步，引擎首选 {plan.name}{detail}",
             *把关((rolled.x, rolled.y))]
    行 += 把关(plan.move)
    if rolled is None and 当前盘 is not None and 当前盘 != board:
        n = 当前盘.counts()
        行.append(f"  注：这一条算的是 {盘} 那一盘，屏上已是 盘{n[BLACK]}:{n[WHITE]}，"
                  f"仅作参考")
    return 行


def cmd_review(args) -> int:
    """一局打完：读日志 → 逐手对照引擎 → 模型写一条原因 → 追写进经验文档。

    只读日志、不碰屏幕，所以任何一份历史样本都能离线重跑。引擎在这儿只回答「那一手
    本该怎么落」，它的评分不进 prompt（原因见 `explain.build_review_prompt`：09-28 实盘
    里一手就能成五的点被它判成 `-999982`，那个数字一进复盘就写出反向教训）。
    """
    cfg = config.load(Path(args.config) if args.config else None)
    if args.log:
        日志 = Path(args.log)
    else:
        最新 = latest_log(RUNS)
        if 最新 is None:
            print(f"{RUNS} 里没有可复盘的日志（一份都读不出盘面），把路径写在 review 后面", file=sys.stderr)
            return 2
        日志 = 最新
        print(f"没给日志，按最新一局读: {日志}")
    frames = read_frames(日志)
    if not frames:
        print(f"日志里读不出任何盘面: {日志}", file=sys.stderr)
        return 2
    boards = [b for _, b in frames]
    mine = resolve_my_colour(cfg.play.my_colour, args.me)
    收官 = boards[-1]

    if args.note:
        局 = Game(mine=mine, result=result_of(boards, mine),
                  moves=收官.counts()[mine], turns=[], skipped=0)
        原因, 来源, 偏离 = args.note, "--note 手写", "未对照（没问引擎）"
    else:
        try:
            engine = Rapfi(cfg, timeout_ms=args.engine_ms).open()
        except EngineError as exc:
            print(f"引擎启动失败: {exc}", file=sys.stderr)
            return 2
        总 = todo_count(frames, mine)
        print(f"盘面 {len(frames)} 个不同局面，要问引擎 {总} 手，"
              f"每手预算 {args.engine_ms}ms（实盘那一档是 {cfg.engine.timeout_ms}ms）")

        def 进度(n: int) -> None:
            if n % 5 == 0 and n != 总:
                print(f"  已对照 {n}/{总} 手…", flush=True)

        t0 = time.perf_counter()
        try:
            局 = round_facts(frames, mine, engine.ask, 进度)
        finally:
            engine.close()
        print(f"对照完 {len(局.turns)}/{总} 手，用了 {time.perf_counter() - t0:.0f}s"
              f"（对不出局面的拍 {局.skipped}）：偏离 {len(局.deviations)}，"
              f"该堵没堵 {len(局.misses)}")
        for t in 局.turns:
            堵 = f"  必堵 {t.must}" if t.must else ""
            判 = f"  ⚠ {t.note}" if t.note else ""
            记 = "" if t.followed else "  ← 偏离"
            print(f"  {t.at} 我落 {t.play}  引擎首选 {t.pick}{堵}{判}{记}")
        print(f"结果: {局.result}（执{WHO_TEXT[mine]} {局.moves} 子，收官盘面 "
              f"黑={收官.counts()[BLACK]} 白={收官.counts()[WHITE]}）")
        key = read_key(cfg.llm.api_key_env)
        if not key:
            print(f"缺密钥，写不出原因；要手写就加 --note \"…\"（环境变量名 "
                  f"{cfg.llm.api_key_env}）", file=sys.stderr)
            return 2
        try:
            原因 = request_chat(cfg, key, build_review_prompt(局))
        except ExplainUnavailable as exc:
            print(f"复盘降级: {exc}", file=sys.stderr)
            return 2
        来源, 偏离 = cfg.llm.model, 局.dev_line()

    条 = Lesson(at=default_at(frames[-1][0], args.date), result=局.result,
                mine=WHO_TEXT[mine], moves=局.moves, dev=偏离, why=原因)
    doc = append(Path(args.lessons), 条)
    print(f"\n已写进 {doc}（原因来源: {来源}）\n{条.block()}")
    return 0


def cmd_watch(args) -> int:
    cfg = config.load(Path(args.config) if args.config else None)
    日志 = None if args.log is None else (args.log or run_log())
    watcher = Watcher(
        cfg=cfg, use_roi=not args.full_screen, events_path=日志
    )
    ticks = 1 if args.once else None
    watcher.run(max_ticks=ticks)
    if 日志:
        print(f"事件日志 {日志}（{len(watcher.events)} 拍）")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="gomoku", description="五子棋盯屏助手：采集 + 标定 + 解析 + 消抖 + RapFi 建议")
    ap.add_argument("--config", help="配置文件路径，默认 config/default.toml")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_parse = sub.add_parser("parse", help="离线解析一张截图")
    p_parse.add_argument("image")
    p_parse.add_argument("--stats", action="store_true", help="打印逐点亮度/暖度，用于调阈值")
    p_parse.add_argument("--json", help="把矩阵写到文件")
    p_parse.set_defaults(func=cmd_parse)

    p_watch = sub.add_parser("watch", help="实时采集循环，打印矩阵与逐拍差分")
    p_watch.add_argument("--once", action="store_true", help="只跑一拍")
    p_watch.add_argument("--full-screen", action="store_true", help="禁用 ROI 跟随，每拍全屏采集")
    p_watch.add_argument(
        "--log", nargs="?", const="", metavar="PATH",
        help="每拍写一行 JSONL：带路径写那儿，光写 --log 就自动落 logs/runs/ 下一局一个文件，"
             "不写这个选项则不落盘",
    )
    p_watch.set_defaults(func=cmd_watch)

    p_adv = sub.add_parser("advise", help="盯屏 + 消抖 + 调 RapFi 出建议 + 后台 LLM 讲解")
    p_adv.add_argument("--full-screen", action="store_true", help="禁用 ROI 跟随，每拍全屏采集")
    p_adv.add_argument("--no-explain", action="store_true", help="只出推荐，不发讲解请求")
    p_adv.add_argument("--me", help="声明你执哪一方：黑/白（也认 b/w/black/white）；不写则读 [play].my_colour，仍是 auto 就开局问一次")
    p_adv.add_argument(
        "--log", nargs="?", const="", metavar="PATH",
        help="逐拍 JSONL 日志（事后 review 唯一读得进来的东西）：不带路径就自动落 "
             f"{RUNS}/advise-<日期-时间>.jsonl，一局一个文件。**默认就写**。",
    )
    p_adv.add_argument("--no-log", action="store_true",
                       help="不落逐拍日志（省 0.9MB/局，代价是这一局事后 review 不了）")
    p_adv.add_argument("--lessons", default=str(DEFAULT_DOC),
                       help=f"启动时读回的经验文档（默认 {DEFAULT_DOC}）")
    p_adv.set_defaults(func=cmd_advise)

    p_rev = sub.add_parser("review", help="一局日志复盘：逐手对照引擎 + 写一条原因 + 追写经验文档")
    p_rev.add_argument("log", nargs="?",
                       help=f"要复盘的日志（JSONL 或从终端粘出来的整屏日志都认）；"
                            f"不写就读 {RUNS} 里最新一份")
    p_rev.add_argument("--me", help="你执哪一方：黑/白；不写则读 [play].my_colour，仍是 auto 就问一次")
    p_rev.add_argument("--note", help="不问引擎也不调模型，把这句话直接作为原因写进文档")
    p_rev.add_argument("--date", help="整屏日志只有时分秒，重放几天前的那份时用它补日期；"
                                      "JSONL 自带日期，不必给")
    p_rev.add_argument("--engine-ms", type=int, default=REVIEW_ENGINE_MS,
                       help=f"复盘每一问的引擎预算。离线问没有时效约束，比实盘那档放宽；"
                            f"调小省时间，调大更接近引擎真正的首选")
    p_rev.add_argument("--lessons", default=str(DEFAULT_DOC),
                       help=f"经验文档路径（默认 {DEFAULT_DOC}）")
    p_rev.set_defaults(func=cmd_review)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
