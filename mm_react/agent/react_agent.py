"""ReAct loop controller for MM-ReAct.

This module wires planner, executor, and memory into the classic loop:

Thought -> Action -> Observation -> Thought -> ... -> Final Answer
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .executor import ImageExecutor, StepResult
from .memory import AgentMemory
from .planner import ImagePlanner, PlanningHistoryItem, ReActDecision


@dataclass(frozen=True)
class ReActRunResult:
    """Final result of one ReAct run."""

    final_answer: str
    final_image: Path
    steps: list[StepResult]
    memory: AgentMemory


class ReActAgent:
    """Runs image enhancement through a multi-turn ReAct loop."""

    def __init__(
        self,
        planner: ImagePlanner | None = None,
        executor: ImageExecutor | None = None,
        memory: AgentMemory | None = None,
        max_turns: int = 8,
    ) -> None:
        self.planner = planner or ImagePlanner()
        self.executor = executor or ImageExecutor()
        self.memory = memory or AgentMemory()
        self.max_turns = max_turns

    def run(self, user_request: str, input_image: str | Path) -> ReActRunResult:
        input_path = Path(input_image)
        current_image = input_path
        planning_history: list[PlanningHistoryItem] = []
        step_results: list[StepResult] = []

        self.memory.add_user_request(user_request, input_path)

        for turn in range(1, self.max_turns + 1):
            decision = self.planner.next_decision(
                user_request=user_request,
                input_image=input_path,
                current_image=current_image,
                planning_history=planning_history,
            )
            self.memory.add_thought(turn, decision)
            history_item = self._history_item(turn, decision)
            planning_history.append(history_item)

            if decision.is_final:
                final_answer = decision.final_answer or ""
                self.memory.add_final_answer(final_answer, current_image)
                return ReActRunResult(
                    final_answer=final_answer,
                    final_image=current_image,
                    steps=step_results,
                    memory=self.memory,
                )

            if decision.tool_call is None:
                raise RuntimeError(
                    "Planner returned neither a tool call nor final answer."
                )

            step_result = self.executor.execute_step(
                tool_call=decision.tool_call,
                input_image=current_image,
                step_index=len(step_results) + 1,
                memory=self.memory,
            )
            step_results.append(step_result)
            history_item.observation = {
                "tool_name": step_result.tool_name,
                "input_image": str(step_result.input_image),
                "observation": step_result.observation,
            }
            if step_result.output_image is not None:
                history_item.observation["output_image"] = str(
                    step_result.output_image
                )
                current_image = step_result.output_image

        final_answer = (
            "Final answer: stopped because the maximum number of ReAct turns "
            f"was reached. Latest image is saved at {current_image}."
        )
        self.memory.add_final_answer(final_answer, current_image)
        return ReActRunResult(
            final_answer=final_answer,
            final_image=current_image,
            steps=step_results,
            memory=self.memory,
        )

    @staticmethod
    def _history_item(turn: int, decision: ReActDecision) -> PlanningHistoryItem:
        return PlanningHistoryItem.from_decision(turn, decision)
