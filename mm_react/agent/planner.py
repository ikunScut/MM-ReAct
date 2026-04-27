"""Rule-based planner demo for image enhancement tools.

The planner turns a user request into an ordered list of tool calls. This is a
small demo version: it uses keywords instead of an LLM, but the returned data
structures are designed to survive an LLM-backed implementation later.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ToolCall:
    """A single tool invocation planned by the agent."""

    tool_name: str
    args: dict[str, Any] = field(default_factory=dict)
    reason: str = ""


@dataclass(frozen=True)
class Plan:
    """A complete plan for one user request."""

    user_request: str
    input_image: Path
    steps: list[ToolCall]
    final_goal: str

    def to_trace(self) -> str:
        lines = [
            f"User request: {self.user_request}",
            f"Input image: {self.input_image}",
            f"Final goal: {self.final_goal}",
            "Plan:",
        ]
        for index, step in enumerate(self.steps, start=1):
            lines.append(
                f"  {index}. {step.tool_name} args={step.args} reason={step.reason}"
            )
        return "\n".join(lines)

# ===== 给 [主循环] 的 Decision 接口 =====
# planner.plan 应返回此 dataclass 
# TODO：预训练模型正则提取得到
@dataclass(frozen=True) # frozen 只读对象
class ReActDecision:
    """One planner decision in the ReAct loop."""

    thought: str
    tool_call: ToolCall | None = None
    final_answer: str | None = None

    @property
    def is_final(self) -> bool:
        return self.final_answer is not None


class ImagePlanner:
    """Plans image enhancement tool chains from natural language requests."""

    def __init__(self, available_tools: set[str] | None = None) -> None:
        self.available_tools = available_tools or {
            "deblur",
            "denoise",
            "low_light_enhance",
            "super_resolution",
            "color_enhance",
            "face_restore",
        }

    def plan(self, user_request: str, input_image: str | Path) -> Plan:
        request = user_request.lower()
        image_path = Path(input_image)
        steps: list[ToolCall] = []

        if self._mentions(request, ["dark", "dim", "low light", "underexposed", "太暗", "暗光"]):
            steps.append(
                ToolCall(
                    tool_name="low_light_enhance",
                    args={"strength": 0.75},
                    reason="Improve brightness and recover details in dark regions.",
                )
            )

        if self._mentions(request, ["noise", "grain", "noisy", "噪声", "去噪", "颗粒"]):
            steps.append(
                ToolCall(
                    tool_name="denoise",
                    args={"level": "auto"},
                    reason="Remove noise before sharpening or super-resolution.",
                )
            )

        if self._mentions(request, ["blur", "blurry", "shake", "模糊", "去模糊"]):
            steps.append(
                ToolCall(
                    tool_name="deblur",
                    args={"mode": "motion_or_defocus"},
                    reason="Recover edge clarity and reduce blur.",
                )
            )

        if self._mentions(request, ["color", "contrast", "vivid", "颜色", "色彩", "对比度"]):
            steps.append(
                ToolCall(
                    tool_name="color_enhance",
                    args={"balance": "natural", "contrast": "medium"},
                    reason="Adjust color balance and contrast after restoration.",
                )
            )

        if self._mentions(request, ["face", "portrait", "人脸", "肖像"]):
            steps.append(
                ToolCall(
                    tool_name="face_restore",
                    args={"fidelity": "balanced"},
                    reason="Restore facial details with a portrait-specific model.",
                )
            )

        if self._mentions(request, ["upscale", "super resolution", "enlarge", "高清", "放大", "超分"]):
            steps.append(
                ToolCall(
                    tool_name="super_resolution",
                    args={"scale": 4},
                    reason="Increase output resolution as the final enhancement step.",
                )
            )

        if not steps:
            steps.append(
                ToolCall(
                    tool_name="color_enhance",
                    args={"balance": "natural", "contrast": "low"},
                    reason="Use a conservative general enhancement when no defect is specified.",
                )
            )

        steps = self._filter_available_tools(steps)
        return Plan(
            user_request=user_request,
            input_image=image_path,
            steps=steps,
            final_goal="Produce an enhanced image that matches the user's request.",
        )

    def next_decision(
        self,
        user_request: str,
        input_image: str | Path,
        current_image: str | Path,
        completed_steps: list[str],
        observations: list[dict[str, Any]],
    ) -> ReActDecision:
        """Choose the next ReAct step after observing previous tool outputs."""

        plan = self.plan(user_request=user_request, input_image=input_image)
        completed = set(completed_steps)
        last_observation = observations[-1] if observations else None

        for step in plan.steps:
            if step.tool_name not in completed:
                observation_note = ""
                if last_observation is not None:
                    observation_note = (
                        f" Last observation came from {last_observation['tool_name']} "
                        f"and produced {last_observation['output_image']}."
                    )
                return ReActDecision(
                    thought=(
                        f"The request still needs {step.tool_name}.{observation_note} "
                        f"I will run it next because: {step.reason}"
                    ),
                    tool_call=step,
                )

        return ReActDecision(
            thought=(
                "All requested enhancement operations have been completed. "
                "The latest image is ready to return as the final answer."
            ),
            final_answer=(
                "Final answer: the image enhancement pipeline is complete. "
                f"The final image is saved at {current_image}."
            ),
        )

    @staticmethod
    def _mentions(text: str, keywords: list[str]) -> bool:
        return any(keyword in text for keyword in keywords)

    def _filter_available_tools(self, steps: list[ToolCall]) -> list[ToolCall]:
        return [step for step in steps if step.tool_name in self.available_tools]


class TransformerImagePlanner(ImagePlanner):
    """Planner demo that gets a plan from a Hugging Face transformer model.

    By default this class uses a mock model response, so the demo does not
    download weights or call a real model. Set use_mock=False after installing
    transformers and preparing a local/remote model.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        available_tools: set[str] | None = None,
        use_mock: bool = True,
        device: str = "cpu",
        max_new_tokens: int = 512,
    ) -> None:
        super().__init__(available_tools=available_tools)
        self.model_name = model_name
        self.use_mock = use_mock
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.last_prompt = ""
        self.last_model_output = ""

    def plan(self, user_request: str, input_image: str | Path) -> Plan:
        prompt = self._build_prompt(user_request=user_request, input_image=input_image)
        model_output = self._generate_plan_text(
            prompt=prompt,
            user_request=user_request,
            input_image=input_image,
        )

        self.last_prompt = prompt
        self.last_model_output = model_output

        return self._parse_model_output(
            model_output=model_output,
            user_request=user_request,
            input_image=input_image,
        )

    def _build_prompt(self, user_request: str, input_image: str | Path) -> str:
        tools = "\n".join(f"- {tool}" for tool in sorted(self.available_tools))
        return f"""You are the planner in an MM-ReAct image enhancement agent.

Your job is to choose an ordered tool plan for the user request.

Available tools:
{tools}

Input image:
{input_image}

User request:
{user_request}

Return JSON only. Do not add markdown. The JSON schema is:
{{
  "final_goal": "short description",
  "steps": [
    {{
      "tool_name": "one available tool name",
      "args": {{}},
      "reason": "why this tool should run now"
    }}
  ]
}}
"""

    def _generate_plan_text(
        self,
        prompt: str,
        user_request: str,
        input_image: str | Path,
    ) -> str:
        if self.use_mock:
            return self._mock_model_output(
                user_request=user_request,
                input_image=input_image,
            )
        return self._generate_with_transformers(prompt)

    def _generate_with_transformers(self, prompt: str) -> str:
        """Real transformer call skeleton. Not used by the default demo."""

        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForCausalLM.from_pretrained(self.model_name)
        model.to(self.device)

        inputs = tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
        )

        prompt_length = inputs["input_ids"].shape[-1]
        generated_tokens = outputs[0][prompt_length:]
        return tokenizer.decode(generated_tokens, skip_special_tokens=True)

    def _mock_model_output(self, user_request: str, input_image: str | Path) -> str:
        rule_plan = super().plan(user_request=user_request, input_image=input_image)
        return json.dumps(
            {
                "final_goal": rule_plan.final_goal,
                "steps": [
                    {
                        "tool_name": step.tool_name,
                        "args": step.args,
                        "reason": step.reason,
                    }
                    for step in rule_plan.steps
                ],
            },
            ensure_ascii=False,
            indent=2,
        )

    def _parse_model_output(
        self,
        model_output: str,
        user_request: str,
        input_image: str | Path,
    ) -> Plan:
        try:
            payload = json.loads(self._extract_json_object(model_output))
            steps = [
                ToolCall(
                    tool_name=item["tool_name"],
                    args=item.get("args", {}),
                    reason=item.get("reason", ""),
                )
                for item in payload.get("steps", [])
                if item.get("tool_name") in self.available_tools
            ]
            if not steps:
                raise ValueError("Model output did not contain valid tool steps.")

            return Plan(
                user_request=user_request,
                input_image=Path(input_image),
                steps=steps,
                final_goal=payload.get(
                    "final_goal",
                    "Produce an enhanced image that matches the user's request.",
                ),
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            return super().plan(user_request=user_request, input_image=input_image)

    @staticmethod
    def _extract_json_object(text: str) -> str:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("No JSON object found in model output.")
        return text[start : end + 1]
