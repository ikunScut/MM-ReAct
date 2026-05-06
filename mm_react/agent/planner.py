"""Planner for one MM-ReAct decision.

核心流程只有一条：
1. planner 构造 prompt；
2. 调用预训练模型得到文本回复；
3. 用正则提取 <thought>、<tool>、<final_answer>；
4. 归一化为 ReActDecision，交给主循环执行。
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from ..tools import AVAILABLE_TOOL_NAMES

PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"


@dataclass(frozen=True)
class ToolCall:
    """One tool invocation selected by the planner."""

    tool_name: str
    args: dict[str, Any] = field(default_factory=dict)
    reason: str = ""


@dataclass(frozen=True)
class ReActDecision:
    """Decision returned to the ReAct main loop for exactly one turn."""

    thought: str
    tool_call: ToolCall | None = None
    final_answer: str | None = None

    @property
    def is_final(self) -> bool:
        return self.final_answer is not None


@dataclass
class PlanningHistoryItem:
    """One ReAct turn kept in planner context."""

    turn: int
    thought: str
    tool: ToolCall | None = None
    final_answer: str | None = None
    observation: dict[str, Any] | None = None

    @classmethod
    def from_decision(
        cls,
        turn: int,
        decision: ReActDecision,
    ) -> PlanningHistoryItem:
        return cls(
            turn=turn,
            thought=decision.thought,
            tool=decision.tool_call,
            final_answer=decision.final_answer,
        )


PlannerBackend = Literal["openai", "transformers"]


class ImagePlanner:
    """Call a pretrained model and parse its text response into a Decision."""

    DEFAULT_TOOLS = AVAILABLE_TOOL_NAMES

    def __init__(
        self,
        backend: PlannerBackend = "transformers",
        available_tools: set[str] | None = None,
    ) -> None:
        if backend not in ("openai", "transformers"):
            raise ValueError("backend must be either 'openai' or 'transformers'.")

        self.backend = backend
        self.available_tools = available_tools or set(self.DEFAULT_TOOLS)
        self.system_prompt = self._load_prompt("system_prompt")
        self.tools_prompt = self._load_prompt("tools_prompt")
        self.output_format_prompt = self._load_prompt("output_format_prompt")
        self.last_prompt = ""
        self.last_model_output = ""

    def next_decision(
        self,
        user_request: str,
        input_image: str | Path,
        current_image: str | Path,
        planning_history: list[PlanningHistoryItem] | None = None,
    ) -> ReActDecision:
        """Return the next Decision by model generation + regex extraction."""

        prompt = self._build_prompt(
            user_request=user_request,
            input_image=input_image,
            current_image=current_image,
            planning_history=planning_history or [],
        )
        model_output = self._generate_text(
            prompt=prompt,
            current_image=current_image,
        )

        self.last_prompt = prompt
        self.last_model_output = model_output

        return self._parse_decision(model_output)

    def _build_prompt(
        self,
        user_request: str,
        input_image: str | Path,
        current_image: str | Path,
        planning_history: list[PlanningHistoryItem] | None = None,
    ) -> str:
        planning_history_text = json.dumps(
            self._history_to_prompt_data(planning_history or []),
            ensure_ascii=False,
            indent=2,
        )
        runtime_context = f"""Runtime Context
===============

User request:
{user_request}

Original input image:
{input_image}

Current image:
{current_image}

Planning history:
{planning_history_text}
"""
        return "\n\n".join(
            (
                self.system_prompt,
                self.tools_prompt,
                runtime_context,
                self.output_format_prompt,
            )
        )

    def _generate_text(
        self,
        prompt: str,
        current_image: str | Path,
    ) -> str:
        if self.backend == "openai":
            return self._generate_with_openai_api(
                prompt=prompt,
                current_image=current_image,
            )
        return self._generate_with_transformers(
            prompt=prompt,
            current_image=current_image,
        )

    def _generate_with_openai_api(
        self,
        prompt: str,
        current_image: str | Path,
    ) -> str:
        from openai import OpenAI

        base_url = os.getenv("MM_REACT_OPENAI_BASE_URL") or os.getenv(
            "OPENAI_BASE_URL"
        )
        model_name = os.getenv(
            "MM_REACT_OPENAI_MODEL",
            "/model" if base_url else "gpt-4o-mini",
        )
        max_output_tokens = int(
            os.getenv(
                "MM_REACT_OPENAI_MAX_OUTPUT_TOKENS",
                os.getenv("MM_REACT_OPENAI_MAX_TOKENS", "512"),
            )
        )
        temperature = float(os.getenv("MM_REACT_OPENAI_TEMPERATURE", "0"))
        top_p = float(os.getenv("MM_REACT_OPENAI_TOP_P", "1"))
        presence_penalty = float(os.getenv("MM_REACT_OPENAI_PRESENCE_PENALTY", "0"))
        top_k = os.getenv("MM_REACT_OPENAI_TOP_K")
        image_detail = os.getenv("MM_REACT_OPENAI_IMAGE_DETAIL", "auto")
        api_type = os.getenv(
            "MM_REACT_OPENAI_API_TYPE",
            "chat_completions" if base_url else "responses",
        )
        send_image = self._env_flag("MM_REACT_OPENAI_SEND_IMAGE", default=True)

        client_kwargs = {}
        api_key = os.getenv("MM_REACT_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
        if base_url:
            client_kwargs["base_url"] = base_url
            client_kwargs["api_key"] = api_key or "EMPTY"
        elif api_key:
            client_kwargs["api_key"] = api_key

        client = OpenAI(**client_kwargs)

        if api_type in {"chat", "chat.completions", "chat_completions"}:
            if send_image:
                content: str | list[dict[str, Any]] = [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": self._image_to_data_url(current_image),
                            "detail": image_detail,
                        },
                    },
                    {"type": "text", "text": prompt},
                ]
            else:
                content = prompt

            request_kwargs = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": content},
                ],
                "max_tokens": max_output_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "presence_penalty": presence_penalty,
            }
            if top_k is not None:
                request_kwargs["extra_body"] = {"top_k": int(top_k)}

            response = client.chat.completions.create(**request_kwargs)
            text = self._normalize_openai_message_content(
                response.choices[0].message.content
            )
            if not text:
                raise ValueError("OpenAI-compatible API returned an empty response.")
            return text

        response = client.responses.create(
            model=model_name,
            instructions=self.system_prompt,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {"type": "input_text", "text": "Current image:"},
                        {
                            "type": "input_image",
                            "image_url": self._image_to_data_url(current_image),
                            "detail": image_detail,
                        },
                    ],
                },
            ],
            max_output_tokens=max_output_tokens,
            temperature=temperature,
        )

        content = response.output_text
        if not content:
            raise ValueError("OpenAI API returned an empty planner response.")
        return content

    @staticmethod
    def _env_flag(name: str, default: bool = False) -> bool:
        value = os.getenv(name)
        if value is None:
            return default
        return value.lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _normalize_openai_message_content(content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict) and isinstance(item.get("text"), str):
                    parts.append(item["text"])
                elif hasattr(item, "text") and isinstance(item.text, str):
                    parts.append(item.text)
            return "\n".join(parts).strip()
        return str(content)

    def _generate_with_transformers(
        self,
        prompt: str,
        current_image: str | Path,
    ) -> str:
        try:
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError:
            from transformers import AutoProcessor
            from transformers import (
                AutoModelForVision2Seq as AutoModelForImageTextToText,
            )

        model_name = os.getenv(
            "MM_REACT_TRANSFORMERS_MODEL",
            "Qwen/Qwen2.5-VL-3B-Instruct",
        )
        device = os.getenv("MM_REACT_TRANSFORMERS_DEVICE", "cpu")
        max_new_tokens = int(os.getenv("MM_REACT_TRANSFORMERS_MAX_NEW_TOKENS", "512"))
        trust_remote_code = (
            os.getenv("MM_REACT_TRANSFORMERS_TRUST_REMOTE_CODE", "false").lower()
            == "true"
        )

        processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
        )
        model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
        ).to(device)
        model.eval()

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        if hasattr(processor, "apply_chat_template"):
            text = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            text = prompt
        image = self._load_image(current_image)
        inputs = processor(
            text=[text],
            images=image,
            return_tensors="pt",
        ).to(device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

        prompt_length = inputs["input_ids"].shape[-1]
        generated_tokens = outputs[0][prompt_length:]
        return processor.decode(generated_tokens, skip_special_tokens=True)

    def _parse_decision(self, text: str) -> ReActDecision:
        thought = self._extract_tag(text, "thought")
        tool_text = self._extract_tag(text, "tool")
        final_answer = self._extract_tag(text, "final_answer")
        tool_item = json.loads(tool_text)

        if isinstance(tool_item, list):
            raise ValueError(
                "<tool> must contain one JSON object or null, not a JSON array."
            )
        if tool_item is not None and not isinstance(tool_item, dict):
            raise ValueError("<tool> must contain one JSON object or null.")

        if final_answer:
            if tool_item is not None:
                raise ValueError("A final decision cannot contain tool calls.")
            return ReActDecision(thought=thought, final_answer=final_answer)

        if tool_item is None:
            raise ValueError("A non-final decision must contain one tool call.")

        tool_call = self._parse_tool_call(tool_item, fallback_reason=thought)
        return ReActDecision(thought=thought, tool_call=tool_call)

    def _parse_tool_call(
        self,
        item: dict[str, Any],
        fallback_reason: str,
    ) -> ToolCall:
        if not isinstance(item, dict):
            raise ValueError("Each tool call must be a JSON object.")

        tool_name = item["tool_name"]
        if tool_name not in self.available_tools:
            raise ValueError(f"Unknown tool from model output: {tool_name}")

        args = item.get("args", {})
        if not isinstance(args, dict):
            raise ValueError("Tool args must be a JSON object.")

        return ToolCall(
            tool_name=tool_name,
            args=args,
            reason=item.get("reason", fallback_reason),
        )

    @staticmethod
    def _load_prompt(prompt_name: str) -> str:
        return (PROMPTS_DIR / prompt_name).read_text(encoding="utf-8").strip()

    @staticmethod
    def _extract_tag(text: str, tag_name: str) -> str:
        pattern = rf"<{tag_name}>\s*(.*?)\s*</{tag_name}>"
        match = re.search(pattern, text, flags=re.DOTALL | re.IGNORECASE)
        if match is None:
            raise ValueError(f"Missing <{tag_name}> section in model output.")
        return match.group(1).strip()

    @staticmethod
    def _history_to_prompt_data(
        planning_history: list[PlanningHistoryItem],
    ) -> list[dict[str, Any]]:
        return [asdict(item) for item in planning_history]

    @staticmethod
    def _image_to_data_url(image_path: str | Path) -> str:
        image_ref = str(image_path)
        if image_ref.startswith(("http://", "https://", "data:")):
            return image_ref

        path = Path(image_path)
        mime_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"

    @staticmethod
    def _load_image(image_path: str | Path) -> Any:
        from PIL import Image

        with Image.open(image_path) as image:
            return image.convert("RGB")
