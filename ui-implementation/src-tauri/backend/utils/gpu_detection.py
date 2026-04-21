"""v0.3.2 (0g): GPU VRAM detection.

Probes nvidia-smi, rocm-smi, and macOS unified memory in that order.
Returns `None` for both vram_gb and count when detection can't be trusted,
which callers must translate into "no warning" rather than a guess.

The model universe is infinite, so we never hard-code per-model VRAM limits
here — we just surface the user's available VRAM and let callers match it
against curated `vram_gb` fields or a parameter-count heuristic.
"""

from __future__ import annotations

import logging
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import Optional


logger = logging.getLogger(__name__)

_SUBPROCESS_FLAGS = 0
if sys.platform == "win32":
    _SUBPROCESS_FLAGS = subprocess.CREATE_NO_WINDOW


@dataclass(frozen=True)
class GpuInfo:
    vram_gb: Optional[float]  # None = unknown; UI must disable VRAM warnings
    count: int                # 0 = no GPU (or unknown); 1+ = known card count
    source: str               # "nvidia-smi" | "rocm-smi" | "macos" | "none"


def _run(cmd: list[str], timeout: float = 3.0) -> Optional[str]:
    """Run a short subprocess; return stdout or None if it fails or hangs."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            creationflags=_SUBPROCESS_FLAGS,
        )
        if result.returncode != 0:
            return None
        return result.stdout
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None


def _detect_nvidia() -> Optional[GpuInfo]:
    out = _run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"])
    if not out:
        return None
    lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
    mib_values: list[int] = []
    for ln in lines:
        try:
            mib_values.append(int(ln))
        except ValueError:
            continue
    if not mib_values:
        return None
    total_mib = max(mib_values)  # primary/largest GPU — we size against the best card
    return GpuInfo(vram_gb=total_mib / 1024.0, count=len(mib_values), source="nvidia-smi")


def _detect_amd() -> Optional[GpuInfo]:
    # rocm-smi's output format shifts across versions; we pull the raw MiB total.
    out = _run(["rocm-smi", "--showmeminfo", "vram", "--csv"])
    if not out:
        return None
    matches = re.findall(r"([0-9]+)\s*(?:MiB|MB)", out)
    if not matches:
        return None
    values_mib = [int(m) for m in matches]
    total_mib = max(values_mib)
    return GpuInfo(vram_gb=total_mib / 1024.0, count=len(values_mib), source="rocm-smi")


def _detect_macos() -> Optional[GpuInfo]:
    if sys.platform != "darwin":
        return None
    out = _run(["sysctl", "-n", "hw.memsize"])
    if not out:
        return None
    try:
        bytes_total = int(out.strip())
    except ValueError:
        return None
    # Apple Silicon shares RAM with the GPU — assume ~75% is usable by Metal.
    usable_gb = (bytes_total / (1024 ** 3)) * 0.75
    return GpuInfo(vram_gb=usable_gb, count=1, source="macos")


def detect_gpu() -> GpuInfo:
    """Probe NVIDIA → AMD → macOS unified → fallback."""
    for probe in (_detect_nvidia, _detect_amd, _detect_macos):
        info = probe()
        if info is not None:
            logger.info(
                "GPU detected via %s: %.1f GB × %d",
                info.source, info.vram_gb or 0.0, info.count,
            )
            return info
    logger.info("GPU VRAM detection: no supported tool found (warnings disabled)")
    return GpuInfo(vram_gb=None, count=0, source="none")


# ---------------------------------------------------------------------------
# Model-size heuristics
# ---------------------------------------------------------------------------

_PARAM_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*b\b", re.IGNORECASE)


def estimate_vram_from_tag(tag: str) -> Optional[float]:
    """Parse "qwen3:72b" → 72.0; estimate VRAM ≈ params × 0.75 (Q4_K_M rule of thumb).

    Returns None when the tag has no parsable parameter count — the caller
    must treat that as "don't warn", never a made-up number.
    """
    match = _PARAM_PATTERN.search(tag or "")
    if not match:
        return None
    try:
        params_b = float(match.group(1))
    except ValueError:
        return None
    return round(params_b * 0.75, 2)
