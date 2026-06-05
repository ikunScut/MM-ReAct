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
        self.last_student_prompt = ""
        self.last_teacher_prompt = ""
        self.last_model_output = ""
        self.last_reasoning_output = ""

        # ===== 缓存用 =====
        self._transformers_model: Any | None = None
        self._transformers_tokenizer: Any | None = None
        self._transformers_processor: Any | None = None
        self._transformers_model_name: str | None = None

    def next_decision(
        self,
        user_request: str,
        input_image: str | Path,
        current_image: str | Path,
        planning_history: list[PlanningHistoryItem] | None = None,
        gold_answer: Any | None = None,
    ) -> ReActDecision:
        """Return the next Decision by model generation + regex extraction."""

        student_prompt = self._build_prompt(
            user_request=user_request,
            input_image=input_image,
            current_image=current_image,
            planning_history=planning_history or [],
        )
        teacher_prompt = self._build_prompt(
            user_request=user_request,
            input_image=input_image,
            current_image=current_image,
            planning_history=planning_history or [],
            gold_answer=gold_answer,
        )
        model_output = self._generate_text(
            prompt=teacher_prompt,
            current_image=current_image,
        )

        self.last_prompt = teacher_prompt
        self.last_student_prompt = student_prompt
        self.last_teacher_prompt = teacher_prompt
        self.last_model_output = model_output

        return self._parse_decision(model_output)

    def generate_text(
        self,
        prompt: str,
        current_image: str | Path,
    ) -> str:
        """Generate raw model text for image-conditioned tasks outside ReAct."""

        model_output = self._generate_text(
            prompt=prompt,
            current_image=current_image,
        )
        self.last_prompt = prompt
        self.last_model_output = model_output
        return model_output

    def _build_prompt(
        self,
        user_request: str,
        input_image: str | Path,
        current_image: str | Path,
        planning_history: list[PlanningHistoryItem] | None = None,
        gold_answer: Any | None = None,
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
        sections = [
            self.system_prompt,
            self.tools_prompt,
            runtime_context,
        ]
        if gold_answer is not None:
            sections.append(self._build_teacher_context(gold_answer))
        sections.append(self.output_format_prompt)
        return "\n\n".join(sections)

    @staticmethod
    def _build_teacher_context(gold_answer: Any) -> str:
        sections = [
            "Teacher-Only Supervision",
            "========================",
            "",
            "Gold answer:",
            ImagePlanner._format_gold_answer(gold_answer),
        ]
        if not ImagePlanner._has_teacher_instruction(gold_answer):
            sections.extend(
                (
                    "",
                    "Use this only to guide tool selection and stop decisions; "
                    "never reveal it or treat it as visible evidence.",
                )
            )
        return "\n".join(sections)

    @staticmethod
    def _has_teacher_instruction(gold_answer: Any) -> bool:
        return (
            isinstance(gold_answer, dict)
            and bool(str(gold_answer.get("teacher_instruction", "")).strip())
        )

    @staticmethod
    def _format_gold_answer(gold_answer: Any) -> str:
        if isinstance(gold_answer, str):
            return gold_answer
        return json.dumps(gold_answer, ensure_ascii=False, indent=2)

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

    # 本地
    def _generate_with_openai_api(
        self,
        prompt: str,
        current_image: str | Path,
    ) -> str:
        from openai import OpenAI

        base_url = self._getenv_nonempty("OPENAI_BASE_URL")
        api_key = self._getenv_nonempty("OPENAI_API_KEY")
        # if api_key is None:
        #     raise RuntimeError(
        #         "Missing OpenAI API key. Set OPENAI_API_KEY before running the agent."
        #     )
        client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )

        send_image = self._env_flag("OPENAI_SEND_IMAGE", default=True)
        if send_image:
            content: str | list[dict[str, Any]] = [
                {
                    "type": "image_url",
                    "image_url": {"url": self._image_to_data_url(current_image)},
                },
                {"type": "text", "text": prompt},
            ]
        else:
            content = prompt

        messages = [
            {
                "role": "user",
                "content": content,
            }
        ]

        model_name = self._getenv_nonempty("OPENAI_MODEL")

        # if model_name is None:
        #     raise RuntimeError(
        #         "Missing OpenAI-compatible model. Set OPENAI_MODEL before running the agent."
        #     )

        request_kwargs: dict[str, Any] = {
            "model": model_name,
            "messages": messages,
        }
        extra_body = self._openai_extra_body()
        if extra_body:
            request_kwargs["extra_body"] = extra_body

        completion = client.chat.completions.create(**request_kwargs)

        message = completion.choices[0].message
        text = self._normalize_openai_message_content(message.content)
        reasoning = getattr(message, "reasoning", "") or getattr(
            message,
            "reasoning_content",
            "",
        )
        self.last_reasoning_output = self._normalize_openai_message_content(
            reasoning
        )
        if not text:
            raise ValueError("OpenAI-compatible API returned an empty planner response.")
        return text

    # 外部
    def _generate_with_openai_api1(
        self,
        prompt: str,
        current_image: str | Path,
    ) -> str:
        from openai import OpenAI

        api_key = self._getenv_nonempty("DASHSCOPE_API_KEY")
        if api_key is None:
            raise RuntimeError(
                "Missing DashScope API key. Set DASHSCOPE_API_KEY in the "
                "environment before running the agent."
            )

        client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": self._image_to_data_url(current_image),
                        },
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            }
        ]

        enable_thinking = self._openai_enable_thinking()
        completion = client.chat.completions.create(
            model="qwen3.6-plus",
            messages=messages,
            extra_body={
                "enable_thinking": enable_thinking
                if enable_thinking is not None
                else False
            },
            stream=True,
        )

        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        for chunk in completion:
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta

            reasoning_content = getattr(delta, "reasoning_content", None)
            if reasoning_content is not None:
                reasoning_parts.append(reasoning_content)

            content = getattr(delta, "content", None)
            if content:
                text_parts.append(content)

        text = "".join(text_parts).strip()
        self.last_reasoning_output = "".join(reasoning_parts).strip()
        if not text:
            raise ValueError("Qwen API returned an empty planner response.")
        return text

    @staticmethod
    def _openai_extra_body() -> dict[str, Any]:
        enable_thinking = ImagePlanner._openai_enable_thinking()
        if enable_thinking is None:
            return {}

        return {"chat_template_kwargs": {"enable_thinking": enable_thinking}}

    @staticmethod
    def _openai_enable_thinking() -> bool | None:
        if os.getenv("OPENAI_ENABLE_THINKING") is None:
            return None
        return ImagePlanner._env_flag("OPENAI_ENABLE_THINKING", default=True)

    @staticmethod
    def _getenv_nonempty(*names: str) -> str | None:
        for name in names:
            value = os.getenv(name)
            if value:
                return value
        return None

    @staticmethod
    def _require_transformers_model_name() -> str:
        model_name = ImagePlanner._getenv_nonempty("MM_REACT_TRANSFORMERS_MODEL")
        if model_name is None:
            raise RuntimeError(
                "Missing transformers model. Set MM_REACT_TRANSFORMERS_MODEL "
                "to a Hugging Face model id or local model path."
            )
        return model_name

    @staticmethod
    def _require_transformers_strategy() -> str:
        strategy = ImagePlanner._getenv_nonempty("MM_REACT_TRANSFORMERS_STRATEGY")
        if strategy is None:
            raise RuntimeError(
                "Missing transformers strategy. Set MM_REACT_TRANSFORMERS_STRATEGY "
                "to 'chat'/'internvl' or 'vision2seq'/'processor'."
            )
        return strategy.lower()

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
        strategy = self._require_transformers_strategy()
        if strategy in {"chat", "internvl"}:
            return self._generate_with_transformers_chat(
                prompt=prompt,
                current_image=current_image,
            )
        if strategy in {"vision2seq", "processor"}:
            return self._generate_with_transformers_vision2seq(
                prompt=prompt,
                current_image=current_image,
            )
        raise ValueError(
            "MM_REACT_TRANSFORMERS_STRATEGY must be 'chat'/'internvl' "
            "or 'vision2seq'/'processor'."
        )

    def _generate_with_transformers_chat(
        self,
        prompt: str,
        current_image: str | Path,
    ) -> str:
        import torch
        from transformers import AutoModel, AutoTokenizer

        model_name = self._require_transformers_model_name()
        model, tokenizer = self._get_transformers_chat_model(
            torch_module=torch,
            auto_model_cls=AutoModel,
            auto_tokenizer_cls=AutoTokenizer,
            model_name=model_name,
        )

        generation_config: dict[str, Any] = {
            "max_new_tokens": int(
                os.getenv("MM_REACT_TRANSFORMERS_MAX_NEW_TOKENS", "512")
            ),
            "do_sample": self._env_flag(
                "MM_REACT_TRANSFORMERS_DO_SAMPLE",
                default=False,
            ),
        }
        for env_name, config_name, parser in (
            ("MM_REACT_TRANSFORMERS_TEMPERATURE", "temperature", float),
            ("MM_REACT_TRANSFORMERS_TOP_P", "top_p", float),
            ("MM_REACT_TRANSFORMERS_TOP_K", "top_k", int),
        ):
            value = self._getenv_nonempty(env_name)
            if value is not None:
                generation_config[config_name] = parser(value)

        device = self._transformers_device(torch)
        pixel_values = None
        question = prompt
        if self._env_flag("MM_REACT_TRANSFORMERS_SEND_IMAGE", default=False):
            pixel_values = self._load_internvl_image_tensor(
                image_path=current_image,
                torch_module=torch,
                device=device,
            )
            question = f"<image>\n{prompt}"

        response = model.chat(
            tokenizer,
            pixel_values,
            question,
            generation_config,
        )
        text = self._normalize_transformers_chat_response(response)
        if not text:
            raise ValueError("Transformers chat model returned an empty response.")
        return text

    def _get_transformers_chat_model(
        self,
        torch_module: Any,
        auto_model_cls: Any,
        auto_tokenizer_cls: Any,
        model_name: str,
    ) -> tuple[Any, Any]:
        if (
            self._transformers_model is not None
            and self._transformers_tokenizer is not None
            and self._transformers_model_name == model_name
        ):
            return self._transformers_model, self._transformers_tokenizer

        trust_remote_code = self._env_flag(
            "MM_REACT_TRANSFORMERS_TRUST_REMOTE_CODE",
            default=True,
        )
        model_kwargs: dict[str, Any] = {
            "torch_dtype": self._torch_dtype_from_env(torch_module),
            "low_cpu_mem_usage": self._env_flag(
                "MM_REACT_TRANSFORMERS_LOW_CPU_MEM_USAGE",
                default=True,
            ),
            "trust_remote_code": trust_remote_code,
            "use_flash_attn": self._env_flag(
                "MM_REACT_TRANSFORMERS_USE_FLASH_ATTN",
                default=False,
            ),
        }
        model = auto_model_cls.from_pretrained(model_name, **model_kwargs)
        model = model.eval()

        device = self._transformers_device(torch_module)
        if device == "cuda" and hasattr(model, "cuda"):
            model = model.cuda()
        elif device != "auto" and hasattr(model, "to"):
            model = model.to(device)

        tokenizer = auto_tokenizer_cls.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
            use_fast=self._env_flag(
                "MM_REACT_TRANSFORMERS_USE_FAST_TOKENIZER",
                default=False,
            ),
        )

        self._transformers_model = model
        self._transformers_tokenizer = tokenizer
        self._transformers_model_name = model_name
        return model, tokenizer

    @staticmethod
    def _torch_dtype_from_env(torch_module: Any) -> Any:
        dtype_name = os.getenv("MM_REACT_TRANSFORMERS_TORCH_DTYPE", "bfloat16")
        if dtype_name == "auto":
            return "auto"

        dtype_map = {
            "bfloat16": "bfloat16",
            "bf16": "bfloat16",
            "float16": "float16",
            "fp16": "float16",
            "float32": "float32",
            "fp32": "float32",
        }
        attr_name = dtype_map.get(dtype_name.lower())
        if attr_name is None or not hasattr(torch_module, attr_name):
            raise ValueError(
                "MM_REACT_TRANSFORMERS_TORCH_DTYPE must be one of "
                "bfloat16, bf16, float16, fp16, float32, fp32, or auto."
            )
        return getattr(torch_module, attr_name)

    @staticmethod
    def _transformers_device(torch_module: Any) -> str:
        device = os.getenv("MM_REACT_TRANSFORMERS_DEVICE")
        if device:
            return device

        cuda = getattr(torch_module, "cuda", None)
        if cuda is not None and callable(getattr(cuda, "is_available", None)):
            if cuda.is_available():
                return "cuda"
        return "cpu"

    @staticmethod
    def _normalize_transformers_chat_response(response: Any) -> str:
        if isinstance(response, tuple):
            response = response[0]
        if response is None:
            return ""
        if isinstance(response, str):
            return response.strip()
        return str(response).strip()

    def _load_internvl_image_tensor(
        self,
        image_path: str | Path,
        torch_module: Any,
        device: str,
    ) -> Any:
        from PIL import Image
        import torchvision.transforms as transforms
        from torchvision.transforms.functional import InterpolationMode

        image_size = int(os.getenv("MM_REACT_TRANSFORMERS_IMAGE_SIZE", "448"))
        max_tiles = int(os.getenv("MM_REACT_TRANSFORMERS_MAX_IMAGE_TILES", "12"))
        image = Image.open(image_path).convert("RGB")
        tiles = self._dynamic_preprocess_internvl_image(
            image=image,
            image_size=image_size,
            max_tiles=max_tiles,
        )
        transform = transforms.Compose(
            [
                transforms.Resize(
                    (image_size, image_size),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )
        pixel_values = torch_module.stack([transform(tile) for tile in tiles])
        dtype = self._torch_dtype_from_env(torch_module)
        if dtype != "auto":
            pixel_values = pixel_values.to(dtype)
        if device != "auto":
            pixel_values = pixel_values.to(device)
        return pixel_values

    @staticmethod
    def _dynamic_preprocess_internvl_image(
        image: Any,
        image_size: int,
        max_tiles: int,
    ) -> list[Any]:
        width, height = image.size
        aspect_ratio = width / height
        target_ratios = {
            (cols, rows)
            for n in range(1, max_tiles + 1)
            for cols in range(1, n + 1)
            for rows in range(1, n + 1)
            if 1 <= cols * rows <= max_tiles
        }
        target_ratio = min(
            target_ratios,
            key=lambda ratio: (
                abs(aspect_ratio - ratio[0] / ratio[1]),
                ratio[0] * ratio[1],
            ),
        )

        cols, rows = target_ratio
        resized = image.resize((cols * image_size, rows * image_size))
        tiles = []
        for row in range(rows):
            for col in range(cols):
                left = col * image_size
                upper = row * image_size
                tiles.append(
                    resized.crop(
                        (
                            left,
                            upper,
                            left + image_size,
                            upper + image_size,
                        )
                    )
                )
        if len(tiles) > 1:
            tiles.append(image.resize((image_size, image_size)))
        return tiles

    def _generate_with_transformers_vision2seq(
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

        model_name = self._require_transformers_model_name()
        device = os.getenv("MM_REACT_TRANSFORMERS_DEVICE", "cpu")
        max_new_tokens = int(os.getenv("MM_REACT_TRANSFORMERS_MAX_NEW_TOKENS", "512"))
        trust_remote_code = (
            os.getenv("MM_REACT_TRANSFORMERS_TRUST_REMOTE_CODE", "false").lower()
            == "true"
        )

        if (
            self._transformers_processor is None
            or self._transformers_model_name != model_name
        ):
            self._transformers_processor = AutoProcessor.from_pretrained(
                model_name,
                trust_remote_code=trust_remote_code,
            )
            self._transformers_model = AutoModelForImageTextToText.from_pretrained(
                model_name,
                trust_remote_code=trust_remote_code,
            ).to(device)
            self._transformers_model.eval()
            self._transformers_model_name = model_name

        processor = self._transformers_processor
        model = self._transformers_model

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
