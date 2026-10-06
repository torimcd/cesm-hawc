"""Text summaries written by the ``cesm-hawc run`` command."""

from __future__ import annotations

import logging
import os

log = logging.getLogger(__name__)


def write_text_summary(lines: list[str], out_dir: str, filename: str = "summary.txt") -> str:
    """Print a summary and write it to a text file.

    Parameters
    ----------
    lines : list of str
        Lines of the summary, joined with newlines.
    out_dir : str
        Directory to write into; created if it doesn't exist.
    filename : str, optional
        Name of the file. Default ``"summary.txt"``.

    Returns
    -------
    str
        Path of the written file.
    """
    text = "\n".join(lines)
    print(text)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    with open(path, "w") as f:
        f.write(text + "\n")
    return path


def format_burden_summary(burden: dict) -> list[str]:
    """Format a sulfate column burden as aligned text lines.

    Parameters
    ----------
    burden : dict
        Result of :meth:`cesm_hawc.waccm.WACCMAtmosphere.sulfate_column_burden`.

    Returns
    -------
    list of str
        One indented ``"key: value"`` line per entry, with numbers to four
        significant figures.
    """
    lines = []
    for k, v in burden.items():
        lines.append(f"  {k:25s}: {v}" if isinstance(v, str) else f"  {k:25s}: {v:.4g}")
    return lines
