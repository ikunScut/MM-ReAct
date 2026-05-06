from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from mm_react.agent.executor import ImageExecutor
from mm_react.agent.memory import AgentMemory
from mm_react.agent.planner import ToolCall


class ExecutorObservationTest(unittest.TestCase):
    def test_image_tool_returns_observation_and_output_image(self) -> None:
        def image_tool(
            input_image: Path, tool_call: ToolCall, output_image: Path
        ) -> str:
            output_image.write_bytes(input_image.read_bytes())
            return "image processed"

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            input_image = root / "input.jpg"
            input_image.write_bytes(b"fake-image")
            memory = AgentMemory()
            executor = ImageExecutor(
                output_dir=root / "outputs",
                tool_registry={"enhance": image_tool},
            )

            result = executor.execute_step(
                tool_call=ToolCall(tool_name="enhance"),
                input_image=input_image,
                step_index=1,
                memory=memory,
            )

        self.assertEqual(result.observation, "image processed")
        self.assertIsNotNone(result.output_image)
        assert result.output_image is not None
        self.assertEqual(result.output_image.name, "input.s1_enhance.jpg")

        event_data = memory.events[-1].data
        self.assertEqual(event_data["observation"], "image processed")
        self.assertIn("output_image", event_data)
        self.assertNotIn("metadata", event_data)

    def test_non_image_tool_returns_json_observation_without_output_image(self) -> None:
        observation = {
            "objects": [
                {"label": "person", "bbox": [10, 20, 30, 40], "score": 0.98}
            ]
        }

        def detection_tool(
            input_image: Path, tool_call: ToolCall, output_image: Path
        ) -> dict[str, Any]:
            return observation

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            input_image = root / "input.jpg"
            input_image.write_bytes(b"fake-image")
            output_dir = root / "outputs"
            output_dir.mkdir()
            stale_output = output_dir / "input.s1_detect.jpg"
            stale_output.write_bytes(b"stale")

            memory = AgentMemory()
            executor = ImageExecutor(
                output_dir=output_dir,
                tool_registry={"detect": detection_tool},
            )

            result = executor.execute_step(
                tool_call=ToolCall(tool_name="detect"),
                input_image=input_image,
                step_index=1,
                memory=memory,
            )

        self.assertEqual(result.observation, observation)
        self.assertIsNone(result.output_image)

        event_data = memory.events[-1].data
        self.assertEqual(event_data["observation"], observation)
        self.assertNotIn("output_image", event_data)
        self.assertNotIn("metadata", event_data)


if __name__ == "__main__":
    unittest.main()
