"""采集：系统自带的 screencapture，零第三方依赖。

实测（2940×1912 内建屏）：全屏一帧 215ms，只抓棋盘面 ROI 一帧 72ms。标定之后一律走
ROI，帧率余量留给消抖和过渡帧判定。

注意 overlay 自激反馈：全屏采集会把画在棋盘区的标注读成棋子。将来上 overlay 时，
采集范围必须避开棋盘区或使用不被采集的窗口（sharingType=.none + 点击穿透），
这件事只需要在本文件里解决一次。
"""

from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

_PROBE_PT = 64


class CaptureError(Exception):
    """screencapture 调用失败（通常是屏幕录制权限没给）。"""


@dataclass(frozen=True)
class Frame:
    rgb: np.ndarray
    origin: tuple[int, int]   # 该帧 (0,0) 对应的全屏像素坐标

    @property
    def shape(self) -> tuple[int, ...]:
        return self.rgb.shape


def _decode(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


class Capture:
    def __init__(self, tmp_dir: str | Path | None = None):
        self._dir = Path(tmp_dir or tempfile.mkdtemp(prefix="gomoku_cap_"))
        self._frame = self._dir / "frame.png"
        self._scale: tuple[float, float] | None = None

    def _snap(self, rect_points: str) -> np.ndarray:
        cmd = ["screencapture", "-x", "-t", "png"]
        if rect_points:
            cmd += ["-R", rect_points]
        cmd.append(str(self._frame))
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or exc.stdout or "").strip()
            raise CaptureError(f"screencapture 失败 (rc={exc.returncode}) {detail}") from exc
        if not self._frame.exists():
            raise CaptureError("screencapture 没有产出文件")
        return _decode(self._frame)

    def pixel_scale(self) -> tuple[float, float]:
        """像素/点 的比例，Retina 上是 2.0；缩放屏上可能是非整数，所以实测而不是假定。"""
        if self._scale is None:
            probe = self._snap(f"0,0,{_PROBE_PT},{_PROBE_PT}")
            self._scale = (probe.shape[1] / _PROBE_PT, probe.shape[0] / _PROBE_PT)
        return self._scale

    def grab_full(self) -> Frame:
        return Frame(rgb=self._snap(""), origin=(0, 0))

    def grab_rect(self, bbox: tuple[int, int, int, int], expand: float = 1.0) -> Frame:
        """按全屏像素 bbox 抓一块，向外放大 expand 倍后取整到稳定的整数点边界。"""
        x0, y0, x1, y1 = bbox
        sx, sy = self.pixel_scale()
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        hw, hh = (x1 - x0) / 2.0 * expand, (y1 - y0) / 2.0 * expand
        px, py = max(int(np.floor((cx - hw) / sx)), 0), max(int(np.floor((cy - hh) / sy)), 0)
        pw, ph = int(np.ceil(2 * hw / sx)), int(np.ceil(2 * hh / sy))
        rgb = self._snap(f"{px},{py},{pw},{ph}")
        return Frame(rgb=rgb, origin=(int(round(px * sx)), int(round(py * sy))))

    def close(self) -> None:
        self._frame.unlink(missing_ok=True)
        self._dir.rmdir()
