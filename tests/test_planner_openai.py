from __future__ import annotations

import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from mm_react.agent.planner import ImagePlanner


class PlannerOpenAITest(unittest.TestCase):
    def test_openai_compatible_multimodal_planner_runs_through(self) -> None:
        calls: dict[str, object] = {}

        class FakeCompletions:
            def create(self, **kwargs: object) -> object:
                calls["create_kwargs"] = kwargs
                message = types.SimpleNamespace(
                    content=(
                        "<thought>图像偏暗，先进行低光增强。</thought>\n"
                        "<tool>\n"
                        '{"tool_name": "low_light_enhance", "args": {}, '
                        '"reason": "提升整体亮度"}\n'
                        "</tool>\n"
                        "<final_answer></final_answer>"
                    )
                )
                choice = types.SimpleNamespace(message=message)
                return types.SimpleNamespace(choices=[choice])

        class FakeChat:
            def __init__(self) -> None:
                self.completions = FakeCompletions()

        class FakeOpenAI:
            def __init__(self, **kwargs: object) -> None:
                calls["client_kwargs"] = kwargs
                self.chat = FakeChat()

        fake_openai = types.ModuleType("openai")
        fake_openai.OpenAI = FakeOpenAI

        env = {
            "OPENAI_BASE_URL": "http://127.0.0.1:8000/v1",
            "MM_REACT_OPENAI_API_KEY": "EMPTY",
            "MM_REACT_OPENAI_MODEL": "/model",
            "MM_REACT_OPENAI_API_TYPE": "chat_completions",
            "MM_REACT_OPENAI_MAX_OUTPUT_TOKENS": "256",
            "MM_REACT_OPENAI_TEMPERATURE": "1.0",
            "MM_REACT_OPENAI_TOP_P": "0.95",
            "MM_REACT_OPENAI_PRESENCE_PENALTY": "1.5",
            "MM_REACT_OPENAI_TOP_K": "20",
            "MM_REACT_OPENAI_IMAGE_DETAIL": "auto",
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            image_path = Path(tmp_dir) / "current.jpg"
            image_path.write_bytes(b"fake-jpeg-bytes")

            with (
                patch.dict(os.environ, env, clear=False),
                patch.dict(sys.modules, {"openai": fake_openai}),
            ):
                os.environ.pop("MM_REACT_OPENAI_BASE_URL", None)
                os.environ.pop("MM_REACT_OPENAI_SEND_IMAGE", None)
                planner = ImagePlanner(backend="openai")
                decision = planner.next_decision(
                    user_request="请增强这张暗光照片",
                    input_image=image_path,
                    current_image=image_path,
                    planning_history=[],
                )

        self.assertFalse(decision.is_final)
        self.assertIsNotNone(decision.tool_call)
        self.assertEqual(decision.tool_call.tool_name, "low_light_enhance")
        self.assertIn("低光增强", decision.thought)
        self.assertIn("请增强这张暗光照片", planner.last_prompt)

        self.assertEqual(
            calls["client_kwargs"],
            {"base_url": "http://127.0.0.1:8000/v1", "api_key": "EMPTY"},
        )

        create_kwargs = calls["create_kwargs"]
        assert isinstance(create_kwargs, dict)
        self.assertEqual(create_kwargs["model"], "/model")
        self.assertEqual(create_kwargs["max_tokens"], 256)
        self.assertEqual(create_kwargs["temperature"], 1.0)
        self.assertEqual(create_kwargs["top_p"], 0.95)
        self.assertEqual(create_kwargs["presence_penalty"], 1.5)
        self.assertEqual(create_kwargs["extra_body"], {"top_k": 20})

        messages = create_kwargs["messages"]
        assert isinstance(messages, list)
        user_content = messages[1]["content"]
        self.assertIsInstance(user_content, list)
        self.assertEqual(user_content[0]["type"], "image_url")
        self.assertEqual(user_content[1]["type"], "text")
        self.assertTrue(
            user_content[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
        )

    def test_parser_rejects_multiple_tools_after_one_thought(self) -> None:
        planner = ImagePlanner(backend="openai")

        with self.assertRaisesRegex(ValueError, "not a JSON array"):
            planner._parse_decision(
                "<thought>需要多个增强步骤，但本轮不能批量调用。</thought>\n"
                "<tool>\n"
                "[\n"
                '  {"tool_name": "low_light_enhance", "args": {}},\n'
                '  {"tool_name": "denoise", "args": {}}\n'
                "]\n"
                "</tool>\n"
                "<final_answer></final_answer>"
            )

    def test_parser_accepts_final_answer_with_null_tool(self) -> None:
        planner = ImagePlanner(backend="openai")

        decision = planner._parse_decision(
            "<thought>所有增强步骤已经完成。</thought>\n"
            "<tool>null</tool>\n"
            "<final_answer>处理完成。</final_answer>"
        )

        self.assertTrue(decision.is_final)
        self.assertIsNone(decision.tool_call)
        self.assertEqual(decision.final_answer, "处理完成。")


if __name__ == "__main__":
    unittest.main()
