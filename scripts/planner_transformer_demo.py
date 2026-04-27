from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from mm_react.agent import TransformerImagePlanner


def main() -> None:
    planner = TransformerImagePlanner(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        use_mock=True,
    )

    plan = planner.plan(
        user_request="这张照片太暗、有噪声，还有点模糊，请增强成高清版本",
        input_image="examples/inputs/demo.jpg",
    )

    print("=== Prompt sent to transformer ===")
    print(planner.last_prompt)
    print()
    print("=== Mock transformer output ===")
    print(planner.last_model_output)
    print()
    print("=== Parsed Plan ===")
    print(plan.to_trace())


if __name__ == "__main__":
    main()
