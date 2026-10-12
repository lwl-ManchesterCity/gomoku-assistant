"""五子棋盯屏助手。

模块边界固定：采集 → 标定 → 解析 → 状态同步 → 决策 → overlay。
上下游之间只传递 15×15 矩阵（见 board.Board），换感知实现或换引擎都不动上游。
"""

__all__ = ["board", "calibrate", "capture", "config", "vision"]
