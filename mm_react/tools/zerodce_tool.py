"""Zero-DCE expert tool wrapper."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any


ZERODCE_CONDA_ENV = os.environ.get("MM_REACT_ZERODCE_CONDA_ENV", "zerodce")
CONDA_EXECUTABLE = os.environ.get("MM_REACT_CONDA", "conda")
TOOL_TIMEOUT_SECONDS = int(os.environ.get("MM_REACT_TOOL_TIMEOUT_SECONDS", "900"))


def run_zerodce_tool(
    input_image: Path, tool_call: Any, output_image: Path
) -> str:
    """Run Zero-DCE through conda and return an execution observation."""

    output_image.parent.mkdir(parents=True, exist_ok=True)
    project_root = Path(__file__).resolve().parents[2]
    model_script = project_root / "experts" / "Zero-DCE" / "entrypoint.py"
    tool_args = dict(tool_call.args)

    cmd = [
        CONDA_EXECUTABLE,
        "run",
        "-n",
        ZERODCE_CONDA_ENV,
        "python",
        str(model_script),
        "--input",
        str(input_image),
        "--output",
        str(output_image),
        "--args-json",
        json.dumps(tool_args, ensure_ascii=False),
    ]

    try:
        subprocess.run(
            cmd,
            text=True,
            capture_output=True,
            check=True,
            timeout=TOOL_TIMEOUT_SECONDS,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(f"Could not find conda executable: {CONDA_EXECUTABLE}") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            "Zero-DCE tool timed out.\n"
            f"command={cmd}\n"
            f"timeout_seconds={TOOL_TIMEOUT_SECONDS}\n"
            f"stdout={exc.stdout}\n"
            f"stderr={exc.stderr}"
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "Zero-DCE tool failed.\n"
            f"command={cmd}\n"
            f"stdout={exc.stdout}\n"
            f"stderr={exc.stderr}"
        ) from exc

    return "图片已处理完成。"
