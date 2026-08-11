"""
cesm_hawc.outputs
==================
Standardized text-summary writing shared by the ``run`` CLI modes.
"""

from __future__ import annotations

import logging
import os

log = logging.getLogger(__name__)


def write_text_summary(lines: list[str], out_dir: str, filename: str = "summary.txt") -> str:
    """Join ``lines`` with newlines, print, and write to
    ``out_dir/filename``. Returns the written path."""
    text = "\n".join(lines)
    print(text)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    with open(path, "w") as f:
        f.write(text + "\n")
    return path


def format_burden_summary(burden: dict) -> list[str]:
    """Format a ``WACCMAtmosphere.sulfate_column_burden()`` result as
    aligned text lines."""
    lines = []
    for k, v in burden.items():
        lines.append(f"  {k:25s}: {v}" if isinstance(v, str) else f"  {k:25s}: {v:.4g}")
    return lines


def format_anomaly_summary(peak_ext_anom: float, peak_r_anom: float,
                            delta_burden: float) -> list[str]:
    """Format background/injection anomaly diagnostics as text lines."""
    return [
        f"Peak extinction anomaly (>15 km):  {peak_ext_anom:.3e} m⁻¹",
        f"Peak radius anomaly (>15 km):      {peak_r_anom:.1f} nm",
        f"Δ SO₄ burden:                      {delta_burden:.3f} mg m⁻²",
    ]
