"""NAFNet expert tool wrapper.

The wrapper launches a small entrypoint in the NAFNet conda environment. The
entrypoint then calls either a user-provided command template or a local NAFNet
checkout.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any


NAFNET_CONDA_ENV = os.environ.get("MM_REACT_NAFNET_CONDA_ENV", "nafnet")
CONDA_EXECUTABLE = os.environ.get("MM_REACT_CONDA", "conda")
TOOL_TIMEOUT_SECONDS = int(os.environ.get("MM_REACT_TOOL_TIMEOUT_SECONDS", "900"))


def run_nafnet_tool(
    input_image: Path, tool_call: Any, output_image: Path
) -> str:
    """Run NAFNet through conda and return an execution observation."""

    output_image.parent.mkdir(parents=True, exist_ok=True)
    project_root = Path(__file__).resolve().parents[2]
    model_script = (
        project_root / "image_enhancement_experts" / "nafnet" / "entrypoint.py"
    )
    tool_args = dict(tool_call.args)

    cmd = [
        CONDA_EXECUTABLE,
        "run",
        "-n",
        NAFNET_CONDA_ENV,
        "python",
        str(model_script),
        "--input",
        str(input_image),
        "--output",
        str(output_image),
        "--task",
        _task_from_tool_call(tool_call.tool_name, tool_args),
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
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "NAFNet tool failed.\n"
            f"command={cmd}\n"
            f"stdout={exc.stdout}\n"
            f"stderr={exc.stderr}"
        ) from exc

    return "图片已处理完成。"


def _task_from_tool_call(tool_name: str, tool_args: dict[str, Any]) -> str:
    if tool_name in {"denoise", "deblur"}:
        return tool_name
    return str(tool_args.get("task", "denoise"))
