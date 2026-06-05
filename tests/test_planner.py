from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mm_react.env import load_local_env
from mm_react.agent.planner import ImagePlanner


# 在这里改图片路径和问题。
IMAGE_PATH = Path("image.png")
QUESTION = "这张图里面有什么？"

# 可选: "openai" 或 "transformers"
PLANNER_BACKEND = "openai"


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

    try:
        decision = planner.next_decision(
            user_request=QUESTION,
            input_image=image_path,
            current_image=image_path,
        )
    except Exception as exc:
        print("\n=== Planner 运行失败 ===")
        print(repr(exc))
        if planner.last_prompt:
            print("\n=== Planner prompt ===")
            print(planner.last_prompt)
        if planner.last_model_output:
            print("\n=== Planner raw model output ===")
            print(planner.last_model_output)
        raise

    print("\n=== 输入 ===")
    print(f"Image: {image_path}")
    print(f"Question: {QUESTION}")

    print("\n=== Planner prompt ===")
    print(planner.last_prompt)

    print("\n=== Planner raw model output ===")
    print(planner.last_model_output)

    if planner.last_reasoning_output:
        print("\n=== Planner reasoning output ===")
        print(planner.last_reasoning_output)

    print("\n=== Parsed planner decision ===")
    print(f"thought: {decision.thought}")

    if decision.tool_call is not None:
        print(f"tool_name: {decision.tool_call.tool_name}")
        print(f"args: {decision.tool_call.args}")
        print(f"reason: {decision.tool_call.reason}")

    if decision.final_answer is not None:
        print(f"final_answer: {decision.final_answer}")

    print("\n=== 简单检查 ===")
    print(f"模型原始输出非空: {bool(planner.last_model_output.strip())}")
    print(f"thought 非空: {bool(decision.thought.strip())}")
    print(f"有工具调用或最终答案: {decision.is_final or decision.tool_call is not None}")


if __name__ == "__main__":
    main()
