from __future__ import annotations

import os
import re
from pathlib import Path

from mm_react.agent.planner import ImagePlanner, PlannerBackend


VIZWIZ_PAPER_PROMPT_TEMPLATE = """{question}
When the provided information is insufficient, respond with 'Unanswerable'.
Answer the question using a single word or phrase."""


def make_prompt(question: str) -> str:
    return VIZWIZ_PAPER_PROMPT_TEMPLATE.format(question=question.strip())


def clean_answer(text: str) -> str:
    answer = str(text).strip()
    final_answer_match = re.search(
        r"<final_answer>\s*(.*?)\s*</final_answer>",
        answer,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if final_answer_match is not None:
        answer = final_answer_match.group(1).strip()

    answer = re.sub(r"^```(?:\w+)?\s*", "", answer).strip()
    answer = re.sub(r"\s*```$", "", answer).strip()
    answer = re.sub(
        r"^(?:final answer|answer|prediction)\s*:\s*",
        "",
        answer,
        flags=re.IGNORECASE,
    ).strip()
    return answer.strip("\"' ")


class PlannerVLMClient:
    """VizWiz VQA client backed by mm_react.agent.planner.ImagePlanner."""

    def __init__(
        self,
        backend: PlannerBackend = "transformers",
        planner: ImagePlanner | None = None,
    ) -> None:
        self.planner = planner or ImagePlanner(backend=backend)
        self.backend = self.planner.backend

    @classmethod
    def from_env(cls) -> "PlannerVLMClient":
        backend = os.environ.get("VIZWIZ_PLANNER_BACKEND", "transformers").strip()
        if backend not in ("openai", "transformers"):
            raise ValueError(
                "VIZWIZ_PLANNER_BACKEND must be either 'openai' or 'transformers'."
            )
        return cls(backend=backend)

    def answer(self, image_path: Path, question: str) -> str:
        prompt = make_prompt(question)
        return clean_answer(
            self.planner.generate_text(
                prompt=prompt,
                current_image=image_path,
            )
        )
