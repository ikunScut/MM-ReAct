from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from mm_react.agent import ImagePlanner


def main() -> None:
    planner = ImagePlanner(backend="transformers")

    decision = planner.next_decision(
        user_request="这张照片太暗、有噪声，还有点模糊，请增强成高清版本",
        input_image="examples/inputs/demo.jpg",
        current_image="examples/inputs/demo.jpg",
        planning_history=[],
    )

    print("=== Prompt sent to transformer ===")
    print(planner.last_prompt)
    print()
    print("=== Transformer output ===")
    print(planner.last_model_output)
    print()
    print("=== Parsed Decision ===")
    print(decision)


if __name__ == "__main__":
    main()
