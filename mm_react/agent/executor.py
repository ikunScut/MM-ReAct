"""Executor demo for MM-ReAct image enhancement steps.

The ReAct loop gives the executor one ToolCall at a time. This demo uses mock
tools that only create output paths and metadata. Real image enhancement models
can be plugged in by passing a custom tool_registry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..tools import DEFAULT_TOOL_REGISTRY, ToolFn
from .memory import AgentMemory
from .planner import ToolCall


@dataclass(frozen=True)
class StepResult:
    """Result of one tool execution."""

    tool_name: str
    input_image: Path
    output_image: Path
    metadata: dict[str, Any] = field(default_factory=dict)


class ImageExecutor:
    """Executes one planned image enhancement step at a time."""

    def __init__(
        self,
        output_dir: str | Path = "outputs/images",
        tool_registry: dict[str, ToolFn] | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.tool_registry = tool_registry or dict(DEFAULT_TOOL_REGISTRY)

    def execute_step(
        self,
        tool_call: ToolCall,
        input_image: str | Path,
        step_index: int,
        memory: AgentMemory | None = None,
    ) -> StepResult:
        """Run exactly one tool call and return its observation."""

        self.output_dir.mkdir(parents=True, exist_ok=True)
        input_path = Path(input_image)
        tool = self.tool_registry.get(tool_call.tool_name)
        if tool is None:
            message = f"Tool is not registered: {tool_call.tool_name}"
            if memory is not None:
                memory.add_error(message, {"tool_name": tool_call.tool_name})
            raise KeyError(message)

        if memory is not None:
            memory.add_tool_start(step_index, tool_call, input_path)

        output_image = self._make_output_path(
            input_image=input_path,
            tool_name=tool_call.tool_name,
            step_index=step_index,
        )
        metadata = tool(input_path, tool_call, output_image)

        if memory is not None:
            memory.add_tool_result(
                step_index=step_index,
                tool_name=tool_call.tool_name,
                input_image=input_path,
                output_image=output_image,
                metadata=metadata,
            )

        return StepResult(
            tool_name=tool_call.tool_name,
            input_image=input_path,
            output_image=output_image,
            metadata=metadata,
        )

    def _make_output_path(
        self, input_image: Path, tool_name: str, step_index: int
    ) -> Path:
        suffix = input_image.suffix or ".png"
        stem = input_image.stem or "image"
        safe_tool_name = tool_name.replace(" ", "_")
        return self.output_dir / f"{stem}.s{step_index}_{safe_tool_name}{suffix}"
