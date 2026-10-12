"""配置装载：config/default.toml → 不可变 dataclass。

阈值全是实测值，调整前先看 tests/test_replay.py 的 golden file 是否仍全绿。
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, fields
from pathlib import Path

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "config" / "default.toml"


@dataclass(frozen=True)
class BoardConfig:
    size: int
    sample_radius: int
    black_max_lum: int
    white_min_lum: int
    max_warmth: int


@dataclass(frozen=True)
class WoodConfig:
    min_r: int
    min_g: int
    min_b: int
    max_b: int
    min_rb_gap: int


@dataclass(frozen=True)
class GridConfig:
    dip_offset: int
    merge_gap: int
    max_spacing_cv: float


@dataclass(frozen=True)
class CaptureConfig:
    interval_s: float
    roi_expand: float
    roi_relocate_fails: int


@dataclass(frozen=True)
class StateConfig:
    confirm_ticks: int
    max_changes: int
    resync_ticks: int


@dataclass(frozen=True)
class EngineConfig:
    binary: str              # 相对仓库根或绝对路径；引擎要求工作目录里权重齐全，故只能是本地文件
    timeout_ms: int


@dataclass(frozen=True)
class PlayConfig:
    # 用户执哪一方只有用户本人知道，绝不能从石子奇偶反推（两局实盘都推错了）。
    # auto = 让 advise 开局问一次；其余取值交给 board.parse_colour 校验。
    my_colour: str


@dataclass(frozen=True)
class LlmConfig:
    base_url: str
    model: str               # 必须是配置项：exp 版模型会改名或下线
    api_key_env: str         # 只存环境变量名，密钥不进 TOML、不进日志
    timeout_ms: int
    max_tokens: int


@dataclass(frozen=True)
class Config:
    board: BoardConfig
    wood: WoodConfig
    grid: GridConfig
    capture: CaptureConfig
    state: StateConfig
    play: PlayConfig
    engine: EngineConfig
    llm: LlmConfig


def _build(cls, raw: dict):
    known = {f.name for f in fields(cls)}
    missing = known - set(raw)
    if missing:
        raise KeyError(f"配置 {cls.__name__} 缺少项: {sorted(missing)}")
    unknown = set(raw) - known
    if unknown:
        raise KeyError(f"配置 {cls.__name__} 存在未知项: {sorted(unknown)}")
    return cls(**raw)


def load(path: Path | None = None) -> Config:
    with open(path or DEFAULT_PATH, "rb") as fh:
        raw = tomllib.load(fh)
    return Config(
        board=_build(BoardConfig, raw["board"]),
        wood=_build(WoodConfig, raw["wood"]),
        grid=_build(GridConfig, raw["grid"]),
        capture=_build(CaptureConfig, raw["capture"]),
        state=_build(StateConfig, raw["state"]),
        play=_build(PlayConfig, raw["play"]),
        engine=_build(EngineConfig, raw["engine"]),
        llm=_build(LlmConfig, raw["llm"]),
    )
