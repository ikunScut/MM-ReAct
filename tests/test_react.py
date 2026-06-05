from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mm_react.env import load_local_env
from mm_react.agent.executor import ImageExecutor
from mm_react.agent.planner import ImagePlanner
from mm_react.agent.react_agent import ReActAgent, ReActRunResult


# 在这里改图片路径、问题和运行配置。
IMAGE_PATH = Path("image.png")
QUESTION = "这张图片有什么？"

# 可选: "openai" 或 "transformers"
PLANNER_BACKEND = "openai"
MAX_TURNS = 4
OUTPUT_DIR = Path("outputs/react_test")


def print_steps(result: ReActRunResult) -> None:
    print("\n=== Tool steps ===")
    print(f"step_count: {len(result.steps)}")
    for index, step in enumerate(result.steps, start=1):
        print(f"\nStep {index}")
        print(f"tool_name: {step.tool_name}")
        print(f"input_image: {step.input_image}")
        print(f"output_image: {step.output_image}")
        print(f"observation: {step.observation}")


def print_simple_checks(result: ReActRunResult) -> None:
    print("\n=== 简单检查 ===")
    print(f"React 运行完成: {bool(result.final_answer)}")
    print(f"最终图片存在: {result.final_image.exists()}")
    print(f"工具执行次数: {len(result.steps)}")
    print(f"Memory 事件数: {len(result.memory.events)}")
    print(
        "有工具调用或直接最终回答: "
        f"{len(result.steps) > 0 or bool(result.final_answer)}"
    )


def main() -> None:
    if not QUESTION.strip():
        print("请先填写 QUESTION。")
        return

    image_path = IMAGE_PATH.expanduser()
    if not image_path.exists():
        print(f"图片路径不存在: {image_path}")
        return

    load_local_env()
    planner = ImagePlanner(backend=PLANNER_BACKEND)
    executor = ImageExecutor(output_dir=OUTPUT_DIR)
    agent = ReActAgent(
        planner=planner,
        executor=executor,
        max_turns=MAX_TURNS,
    )

    print("=== 输入 ===")
    print(f"Image: {image_path}")
    print(f"Question: {QUESTION}")
    print(f"Planner backend: {PLANNER_BACKEND}")
    print(f"Max turns: {MAX_TURNS}")
    print(f"Output dir: {OUTPUT_DIR}")

    try:
        result = agent.run(QUESTION, image_path)
    except Exception as exc:
        print("\n=== ReAct 运行失败 ===")
        print(repr(exc))
        if planner.last_prompt:
            print("\n=== Last planner prompt ===")
            print(planner.last_prompt)
        if planner.last_model_output:
            print("\n=== Last planner raw model output ===")
            print(planner.last_model_output)
        if planner.last_reasoning_output:
            print("\n=== Last planner reasoning output ===")
            print(planner.last_reasoning_output)
        print("\n=== Memory trace ===")
        print(agent.memory.to_trace())
        raise

    print("\n=== Final result ===")
    print(f"final_answer: {result.final_answer}")
    print(f"final_image: {result.final_image}")

    print_steps(result)

    print("\n=== Last planner prompt ===")
    print(planner.last_prompt)

    print("\n=== Last planner raw model output ===")
    print(planner.last_model_output)

    if planner.last_reasoning_output:
        print("\n=== Last planner reasoning output ===")
        print(planner.last_reasoning_output)

    print("\n=== Memory trace ===")
    print(result.memory.to_trace())

    print_simple_checks(result)


if __name__ == "__main__":
    main()
