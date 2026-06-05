"""Lightweight memory for an MM-ReAct run.

Memory stores the reasoning trace: user request, planned tool calls, execution
steps, observations, and final output. It is intentionally simple so it can be
printed, saved, or later replaced by a database-backed implementation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .planner import ReActDecision, ToolCall


@dataclass(frozen=True)
class MemoryEvent:
    """One event in the agent trace."""

    event_type: str
    message: str
    data: dict[str, Any] = field(default_factory=dict)


class AgentMemory:
    """Stores the short-term trace for one image enhancement request."""

    def __init__(self) -> None:
        self.events: list[MemoryEvent] = []
        self.sft_records: list[dict[str, Any]] = []

    def add_user_request(self, request: str, image_path: str | Path) -> None:
        self.add(
            event_type="user",
            message="Received user request.",
            data={"request": request, "image_path": str(image_path)},
        )

    def add_thought(self, turn: int, decision: ReActDecision) -> None:
        self.add(
            event_type="thought",
            message=f"ReAct turn {turn}: planner decided the next step.",
            data={
                "thought": decision.thought,
                "next_tool": (
                    decision.tool_call.tool_name if decision.tool_call else None
                ),
                "is_final": decision.is_final,
            },
        )

    def add_tool_start(
        self, step_index: int, tool_call: ToolCall, input_image: Path
    ) -> None:
        self.add(
            event_type="action",
            message=f"Running tool {step_index}: {tool_call.tool_name}.",
            data={
                "tool_name": tool_call.tool_name,
                "args": tool_call.args,
                "input_image": str(input_image),
                "reason": tool_call.reason,
            },
        )

    def add_tool_result(
        self,
        step_index: int,
        tool_name: str,
        input_image: Path,
        output_image: Path | None,
        observation: Any,
    ) -> None:
        data = {
            "tool_name": tool_name,
            "input_image": str(input_image),
            "observation": observation,
        }
        if output_image is not None:
            data["output_image"] = str(output_image)

        self.add(
            event_type="observation",
            message=f"Tool {step_index} finished: {tool_name}.",
            data=data,
        )

    def add_final_result(self, output_image: Path) -> None:
        self.add(
            event_type="final",
            message="Image enhancement pipeline finished.",
            data={"output_image": str(output_image)},
        )

    def add_final_answer(self, final_answer: str, output_image: Path) -> None:
        self.add(
            event_type="final",
            message="Agent returned the final answer.",
            data={"answer": final_answer, "output_image": str(output_image)},
        )

    def add_error(self, message: str, data: dict[str, Any] | None = None) -> None:
        self.add(event_type="error", message=message, data=data or {})

    def add(self, event_type: str, message: str, data: dict[str, Any]) -> None:
        self.events.append(
            MemoryEvent(event_type=event_type, message=message, data=data)
        )

    def add_sft_turn(
        self,
        current_image: str | Path,
        student_prompt: str,
        assistant_output: str,
    ) -> None:
        """Store one student-visible input/output pair for ReAct SFT."""

        current_image_text = str(current_image)
        record = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": current_image_text},
                        {"type": "text", "text": student_prompt},
                    ],
                },
                {"role": "assistant", "content": assistant_output},
            ],
        }
        self.sft_records.append(record)

    def to_sft_records(self) -> list[dict[str, Any]]:
        """Return JSON-serializable SFT records, one record per ReAct turn."""

        return self._to_jsonable(self.sft_records)

    def save_sft_jsonl(self, path: str | Path) -> Path:
        """Save SFT records as JSONL, one training example per line."""

        sft_path = Path(path)
        sft_path.parent.mkdir(parents=True, exist_ok=True)
        with sft_path.open("w", encoding="utf-8") as f:
            for record in self.to_sft_records():
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return sft_path

    def save_sft_json(self, path: str | Path) -> Path:
        """Save SFT records as one JSON array."""

        sft_path = Path(path)
        sft_path.parent.mkdir(parents=True, exist_ok=True)
        sft_path.write_text(
            json.dumps(self.to_sft_records(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return sft_path

    def to_trace(self) -> str:
        lines = ["MM-ReAct Trace", "==============", ""]
        for index, event in enumerate(self.events, start=1):
            lines.append(f"[{index}] {event.event_type.upper()}: {event.message}")
            for key, value in event.data.items():
                lines.append(f"    {key}: {value}")
            lines.append("")
        return "\n".join(lines).rstrip()

    def save_trace(self, path: str | Path) -> Path:
        trace_path = Path(path)
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_path.write_text(self.to_trace() + "\n", encoding="utf-8")
        return trace_path

    @staticmethod
    def _to_jsonable(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {
                str(key): AgentMemory._to_jsonable(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [AgentMemory._to_jsonable(item) for item in value]
        try:
            json.dumps(value, ensure_ascii=False)
        except TypeError:
            return str(value)
        return value
