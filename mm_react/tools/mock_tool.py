"""Mock image-enhancement tool implementation for local demos."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def run_mock_image_tool(
    input_image: Path, tool_call: Any, output_image: Path
) -> dict[str, Any]:
    """Create a placeholder artifact and return mock execution metadata."""

    output_image.parent.mkdir(parents=True, exist_ok=True)
    output_image.write_text(
        "\n".join(
            [
                "This is a mock image artifact for the MM-ReAct demo.",
                f"input_image={input_image}",
                f"tool_name={tool_call.tool_name}",
                f"args={tool_call.args}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "backend": "mock",
        "note": "Replace this mock tool with a real image enhancement model.",
        "args": tool_call.args,
    }
