"""Executor demo for MM-ReAct image enhancement plans.

The executor receives a Plan from the planner and runs each ToolCall in order.
This demo uses mock tools that only create output paths and metadata. Real image
enhancement models can be plugged in by passing a custom tool_registry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..tools import DEMO_TOOL_REGISTRY
from .memory import AgentMemory
from .planner import Plan, ToolCall


ToolFn = Callable[[Path, ToolCall, Path], dict[str, Any]]


@dataclass(frozen=True)
class StepResult:
    """Result of one tool execution."""

    tool_name: str
    input_image: Path
    output_image: Path
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExecutionResult:
    """Result of a complete plan execution."""

    input_image: Path
    final_image: Path
    steps: list[StepResult]


class ImageExecutor:
    """Executes planned image enhancement steps."""

    def __init__(
        self,
        output_dir: str | Path = "outputs/images",
        tool_registry: dict[str, ToolFn] | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.tool_registry = tool_registry or self._build_mock_registry()

    def execute(
        self, plan: Plan, memory: AgentMemory | None = None
    ) -> ExecutionResult:
        self.output_dir.mkdir(parents=True, exist_ok=True)

        current_image = plan.input_image
        results: list[StepResult] = []

        for index, tool_call in enumerate(plan.steps, start=1):
            step_result = self.execute_step(
                tool_call=tool_call,
                input_image=current_image,
                step_index=index,
                memory=memory,
            )
            results.append(step_result)
            current_image = step_result.output_image

        if memory is not None:
            memory.add_final_result(current_image)

        return ExecutionResult(
            input_image=plan.input_image,
            final_image=current_image,
            steps=results,
        )

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

    def _build_mock_registry(self) -> dict[str, ToolFn]:
        return {
            **DEMO_TOOL_REGISTRY,
            "deblur": self._mock_tool,
            "denoise": self._mock_tool,
            "low_light_enhance": self._mock_tool,
            "super_resolution": self._mock_tool,
            "color_enhance": self._mock_tool,
            "face_restore": self._mock_tool,
        }

    @staticmethod
    def _mock_tool(
        input_image: Path, tool_call: ToolCall, output_image: Path
    ) -> dict[str, Any]:
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
