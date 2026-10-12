# 五子棋盯屏助手（gomoku-assistant）

在 macOS 上盯着一款五子棋 App 的画面：识别棋盘、跟踪落子，把「你的下一手」实时打在终端上，
异步给一段中文讲解；一局打完可 `review` 出逐手对照，并把胜负原因写进一份经验文档。

对手是 App 里的人机。这套工具不注入、不挂钩，只读屏幕 —— 截图 + 计算机视觉是唯一感知路线。

## 链路

采集 → 标定 → 解析 → 状态同步 → 引擎 → 棋型安全网 → 讲解 / 复盘

| 层 | 文件 | 干什么 |
| --- | --- | --- |
| 采集 | `src/gomoku/capture.py` | 调系统 `screencapture`；全屏定位一次后按棋盘 bbox 外扩跟随 ROI |
| 标定 | `src/gomoku/calibrate.py` | 检出 15×15 网格线；锁定期每拍先跑门卫校验，通过就沿用旧格点 |
| 解析 | `src/gomoku/vision.py` | 225 个交叉点逐点判空/黑/白（亮度＋暖度两条判据） |
| 状态 | `src/gomoku/state.py` | 消抖：默认新子当拍提交；单拍变化 >3 点判过渡帧整拍丢弃 |
| 引擎 | `src/gomoku/engine.py` | RapFi（Gomocup 协议）适配；每一问整盘重灌，回的是「轮到的一方」的一手 |
| 棋型 | `src/gomoku/pattern.py` | 活三/冲四/活四扫描；`safety_note` 安全网在推荐没应对方先手时报警 |
| 讲解 | `src/gomoku/explain.py` | 后台线程调 LLM 写 100–180 字中文讲解；不可用时静默降级，不影响建议 |
| 复盘 | `src/gomoku/lessons.py` | 从逐拍 JSONL 重建每手对照（偏离 / 该堵没堵），模型只写「原因」一句 |

## 环境与安装

要求：macOS（依赖系统自带的 `screencapture`）、Python ≥ 3.11，依赖 numpy / scipy / pillow
（见 `pyproject.toml`）。

```
python3 -m venv .venv
.venv/bin/pip install numpy scipy pillow     # 或 pip install -e .
```

运行本身不要求安装这个包：所有命令都走 `PYTHONPATH=src`（见下节）。

引擎不入库：从 [dhbloo/rapfi](https://github.com/dhbloo/rapfi) 的 Releases 取 macOS Apple
Silicon 构建，解压到 `vendor/rapfi/`，目录里必须有引擎二进制、`config.toml` 与全部权重文件。
默认二进制名写在 `config/default.toml` 的 `[engine].binary`。

讲解层的密钥只从环境变量 `DEEPSEEK_API_KEY` 或**仓库根目录的 `.env`** 读取，绝不写进配置
文件、日志或 prompt。`.env`、`vendor/`、`.venv/`、`logs/` 均已 gitignore。

## 跑

命令前缀统一为 `PYTHONPATH=src .venv/bin/python -m gomoku`。

```
gomoku parse 截图.png [--stats] [--json out.txt]      # 离线解析一张截图
gomoku watch  [--once] [--full-screen] [--log [路径]]  # 只采集打印，不问引擎
gomoku advise --me 黑 [--no-explain] [--no-log]        # 实时建议 + 讲解（主命令）
gomoku review [日志] [--engine-ms 6000]                # 复盘并写经验文档
```

- `advise` **默认就写逐拍日志**，落在 `logs/runs/advise-<日期-时间>.jsonl`，一局一个文件；
  `--no-log` 关闭（代价是这一局事后 review 不了）。`--me 黑/白` 声明执子方，不写则问一次。
- `review` 不带日志路径时读 `logs/runs/` 里最新一份读得出盘面的日志。
- 复盘默认每问给引擎 6000ms（实盘只有 1500ms，那档是为时效）；偏离计数会随预算档变化，
  所以输出里始终带着档位。

## 测试

```
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests
```

190 项全绿。含 golden file 回放（`tests/fixtures/`）与跑真引擎二进制时录下的协议用例；
`tools/probe_*.py` 是一次性的真机探针（需要真引擎与真日志），结论都写在代码 docstring 里。

`samples/` 存着三份 2026-09-28 的终端实盘日志（负 22 手 / 胜 14 手 / 胜 21 手），
`review` 与离线重放都拿它们当样本。

## 收尾时点的已知边界

- 建议行说的是「轮到的一方」的一手，出口会带执子方标签；用户落子在先的一拍会
  退成「复核」或「复盘」两种措辞。
- 引擎评分 `Eval` 的正负跟着**走子方**翻（`tools/probe_eval_side.py` 实测，黑白各一次
  同形局面均回 −999998），不跟颜色也不跟赢面。
- 经验文档只喂讲解层，**不参与选点**：文档里的「原因」是模型写的散文，且引擎首选
  本身随预算档漂移，两样都不适合作落点判据。
- 悬浮 overlay 已被否掉：终端一行字是最终形态。
- 四条结论目前只有单元测试 / 离线重放证据，还没有实盘坐实：复盘支对用户实际落点的报警
  文案、棋盘落后时的「堵不住」报警文案、「不走它会怎样」讲解槽位、经验真的进了 prompt。
  验证它们需要一局带偏离的真实日志，跑 `advise` → `review` 即可。
