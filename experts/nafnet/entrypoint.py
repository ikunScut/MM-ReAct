"""NAFNet expert entrypoint.

Configuration options:

- MM_REACT_NAFNET_COMMAND_TEMPLATE:
  Shell-like command template with placeholders: {input}, {output}, {task},
  {args_json}. When set, this command is used directly.
- MM_REACT_NAFNET_ROOT:
  Path to a local NAFNet checkout. Used when no command template is set.
- MM_REACT_NAFNET_DENOISE_OPT / MM_REACT_NAFNET_DEBLUR_OPT:
  NAFNet option files, relative to MM_REACT_NAFNET_ROOT or absolute.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DEFAULT_OPT_FILES = {
    "denoise": "options/test/SIDD/NAFNet-width64.yml",
    "deblur": "options/test/GoPro/NAFNet-width64.yml",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NAFNet image restoration expert.")
    parser.add_argument("--input", required=True, help="Input image path.")
    parser.add_argument("--output", required=True, help="Output image path.")
    parser.add_argument(
        "--task",
        choices=("denoise", "deblur"),
        default="denoise",
        help="NAFNet restoration task.",
    )
    parser.add_argument("--args-json", default="{}", help="Tool args encoded as JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    tool_args = json.loads(args.args_json)

    if not input_path.is_file():
        raise FileNotFoundError(f"Input image does not exist: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    command_template = os.environ.get("MM_REACT_NAFNET_COMMAND_TEMPLATE")
    if command_template:
        command = _command_from_template(
            command_template,
            input_path=input_path,
            output_path=output_path,
            task=args.task,
            tool_args=tool_args,
        )
        completed = _run(command)
        artifact = "command_template"
    else:
        command, completed = _run_official_demo(
            input_path=input_path,
            output_path=output_path,
            task=args.task,
        )
        artifact = "official_demo"

    if not output_path.is_file():
        raise FileNotFoundError(f"NAFNet did not create output image: {output_path}")

    print(
        json.dumps(
            {
                "expert": "nafnet",
                "entrypoint": "image_enhancement_experts/nafnet/entrypoint.py",
                "artifact": artifact,
                "task": args.task,
                "command": command,
                "input_image": str(input_path),
                "output_image": str(output_path),
                "args": tool_args,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            },
            ensure_ascii=False,
        )
    )


def _command_from_template(
    template: str,
    input_path: Path,
    output_path: Path,
    task: str,
    tool_args: dict[str, Any],
) -> list[str]:
    rendered = template.format(
        input=str(input_path),
        output=str(output_path),
        task=task,
        args_json=json.dumps(tool_args, ensure_ascii=False),
    )
    return shlex.split(rendered)


def _run_official_demo(
    input_path: Path,
    output_path: Path,
    task: str,
) -> tuple[list[str], subprocess.CompletedProcess[str]]:
    nafnet_root_env = os.environ.get("MM_REACT_NAFNET_ROOT")
    if not nafnet_root_env:
        raise RuntimeError(
            "MM_REACT_NAFNET_ROOT is required unless "
            "MM_REACT_NAFNET_COMMAND_TEMPLATE is set."
        )

    nafnet_root = Path(nafnet_root_env)
    demo_script = nafnet_root / os.environ.get(
        "MM_REACT_NAFNET_DEMO",
        "basicsr/demo.py",
    )
    opt_file = _resolve_opt_file(nafnet_root, task)

    with tempfile.TemporaryDirectory(
        prefix="nafnet_",
        dir=str(output_path.parent),
    ) as temp_dir:
        temp_output_dir = Path(temp_dir)
        command = [
            sys.executable,
            str(demo_script),
            "-opt",
            str(opt_file),
            "--input_path",
            str(input_path),
            "--output_path",
            str(temp_output_dir),
        ]
        completed = _run(command, cwd=nafnet_root)
        restored_image = _find_restored_image(temp_output_dir, input_path.stem)
        shutil.copyfile(restored_image, output_path)

    return command, completed


def _resolve_opt_file(nafnet_root: Path, task: str) -> Path:
    env_name = f"MM_REACT_NAFNET_{task.upper()}_OPT"
    opt_value = os.environ.get(env_name, DEFAULT_OPT_FILES[task])
    opt_file = Path(opt_value)
    if not opt_file.is_absolute():
        opt_file = nafnet_root / opt_file
    if not opt_file.is_file():
        raise FileNotFoundError(
            f"NAFNet option file does not exist for task {task}: {opt_file}"
        )
    return opt_file


def _run(
    command: list[str],
    cwd: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    timeout = int(os.environ.get("MM_REACT_NAFNET_TIMEOUT_SECONDS", "900"))
    env = os.environ.copy()
    if cwd is not None:
        pythonpath = env.get("PYTHONPATH", "")
        paths = [str(cwd)]
        if pythonpath:
            paths.append(pythonpath)
        env["PYTHONPATH"] = os.pathsep.join(paths)

    return subprocess.run(
        command,
        cwd=str(cwd) if cwd is not None else None,
        env=env,
        text=True,
        capture_output=True,
        check=True,
        timeout=timeout,
    )


def _find_restored_image(output_dir: Path, input_stem: str) -> Path:
    candidates = [
        path
        for path in output_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    if not candidates:
        raise FileNotFoundError(f"No restored image was found under {output_dir}")

    same_stem = [path for path in candidates if path.stem == input_stem]
    if same_stem:
        return max(same_stem, key=lambda path: path.stat().st_mtime)
    return max(candidates, key=lambda path: path.stat().st_mtime)


if __name__ == "__main__":
    main()
