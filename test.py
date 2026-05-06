"""Smoke test for the Zero-DCE MM-ReAct tool wrapper.

Run from the repository root:

    python test.py

Optional:

    python test.py --input 6.png --output 6_output.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mm_react.agent.planner import ToolCall
from mm_react.tools.zerodce_tool import run_zerodce_tool


DEFAULT_INPUT = Path("experts/Zero-DCE/Zero-DCE_code/data/test_data/LIME/1.bmp")
DEFAULT_OUTPUT = Path("outputs/zerodce_tool_test.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test Zero-DCE tool wrapper.")
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input image path. Default: {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output image path. Default: {DEFAULT_OUTPUT}",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tool_call = ToolCall(tool_name="low_light_enhance", args={})

    result = run_zerodce_tool(
        input_image=args.input,
        tool_call=tool_call,
        output_image=args.output,
    )

    print("Zero-DCE tool call succeeded.")
    print(f"input: {args.input}")
    print(f"output: {args.output}")
    print(f"output_exists: {args.output.is_file()}")
    print(
        json.dumps(
            {
                "expert": result["expert"],
                "conda_env": result["conda_env"],
                "device": result["model_metadata"].get("device"),
                "elapsed_seconds": result["model_metadata"].get("elapsed_seconds"),
                "entrypoint": result["model_metadata"].get("entrypoint"),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
