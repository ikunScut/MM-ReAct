"""Demo image-enhancement expert entrypoint.

This script represents a deep-learning model service/entrypoint that runs in
its own conda environment. The MM-ReAct tool wrapper launches it with:

    conda run -n ENV python image_enhancement_experts/demo/entrypoint.py
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Demo image enhancement expert.")
    parser.add_argument("--input", required=True, help="Input image path.")
    parser.add_argument("--output", required=True, help="Output image path.")
    parser.add_argument("--args-json", default="{}", help="Tool args encoded as JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    tool_args = json.loads(args.args_json)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if input_path.exists() and input_path.is_file():
        shutil.copyfile(input_path, output_path)
        artifact = "copied_input"
    else:
        output_path.write_text(
            "\n".join(
                [
                    "This artifact was produced by image_enhancement_experts/demo.",
                    f"input_image={input_path}",
                    f"args={tool_args}",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        artifact = "text_placeholder"

    print(
        json.dumps(
            {
                "expert": "demo",
                "entrypoint": "image_enhancement_experts/demo/entrypoint.py",
                "artifact": artifact,
                "input_image": str(input_path),
                "output_image": str(output_path),
                "args": tool_args,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
