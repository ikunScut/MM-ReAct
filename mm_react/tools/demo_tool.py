"""Demo tool implementation used by the executor registry."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any


DEMO_CONDA_ENV = os.environ.get("MM_REACT_DEMO_CONDA_ENV", "base")
CONDA_EXECUTABLE = os.environ.get("MM_REACT_CONDA", "conda")
TOOL_TIMEOUT_SECONDS = int(os.environ.get("MM_REACT_TOOL_TIMEOUT_SECONDS", "300"))


def run_demo_tool(
    input_image: Path, tool_call: Any, output_image: Path
) -> dict[str, Any]:
    """Run the demo model through conda and return execution metadata."""

    output_image.parent.mkdir(parents=True, exist_ok=True)
    project_root = Path(__file__).resolve().parents[2]
    model_script = project_root / "image_enhancement_experts" / "demo" / "entrypoint.py"
    tool_args = dict(tool_call.args)

    cmd = [
        CONDA_EXECUTABLE,
        "run",
        "-n",
        DEMO_CONDA_ENV,
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
        result = subprocess.run(
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
            "demo tool failed.\n"
            f"command={cmd}\n"
            f"stdout={exc.stdout}\n"
            f"stderr={exc.stderr}"
        ) from exc

    return {
        "backend": "conda_run",
        "conda_env": DEMO_CONDA_ENV,
        "model_script": str(model_script),
        "input_image": str(input_image),
        "output_image": str(output_image),
        "model_metadata": _parse_model_metadata(result.stdout),
        "stderr": result.stderr,
        "args": tool_call.args,
    }


def _parse_model_metadata(stdout: str) -> dict[str, Any]:
    lines = [line for line in stdout.splitlines() if line.strip()]
    if not lines:
        return {}
    return json.loads(lines[-1])
