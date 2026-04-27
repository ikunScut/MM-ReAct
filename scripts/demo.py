from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from mm_react.agent import ReActAgent


def main() -> None:
    user_request = "这张照片太暗、有噪声，还有点模糊，请增强成高清版本"
    input_image = "examples/inputs/demo.jpg"

    agent = ReActAgent(max_turns=8)
    result = agent.run(user_request=user_request, input_image=input_image)
    trace_path = result.memory.save_trace("outputs/traces/demo_trace.txt")

    print(result.memory.to_trace())
    print()
    print(result.final_answer)
    print(f"Final image: {result.final_image}")
    print(f"Trace saved to: {trace_path}")


if __name__ == "__main__":
    main()
