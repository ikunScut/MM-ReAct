from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evals.vizwiz.data import VizWizSample, append_jsonl, iter_vizwiz_samples
from evals.vizwiz.vlm_client import clean_answer, make_prompt
from mm_react.agent.executor import ImageExecutor, StepResult
from mm_react.agent.memory import AgentMemory, MemoryEvent
from mm_react.agent.planner import ImagePlanner, PlannerBackend
from mm_react.agent.react_agent import ReActAgent
from mm_react.env import load_local_env


MAX_TURNS_MARKER = "maximum number of ReAct turns was reached"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run VizWiz VQA through the MM-ReAct tool-using agent."
    )
    parser.add_argument(
        "--vizwiz-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "VizWiz",
    )
    parser.add_argument("--split", default="val")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "vizwiz_agent" / "predictions.jsonl",
    )
    parser.add_argument(
        "--image-output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "vizwiz_agent" / "images",
        help="Directory where agent tool outputs are saved.",
    )
    parser.add_argument("--skip-missing", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--backend", choices=["openai", "transformers"], default=None)
    parser.add_argument(
        "--max-turns",
        type=int,
        default=4,
        help="Maximum ReAct planner/tool turns per VizWiz sample.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel agent runs. Use 1 for transformers.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Seconds to sleep between submitted samples.",
    )
    parser.add_argument(
        "--include-prompt",
        action="store_true",
        help="Store the exact agent user_request prompt in each prediction row.",
    )
    return parser.parse_args()


def run_one(
    sample: VizWizSample,
    backend: PlannerBackend,
    max_turns: int,
    image_output_dir: Path,
    include_prompt: bool,
    planner: ImagePlanner | None = None,
) -> dict[str, Any]:
    user_request = make_prompt(sample.question)
    memory = AgentMemory()
    executor = ImageExecutor(output_dir=image_output_dir)
    agent = ReActAgent(
        planner=planner or ImagePlanner(backend=backend),
        executor=executor,
        memory=memory,
        max_turns=max_turns,
    )

    try:
        result = agent.run(
            user_request=user_request,
            input_image=sample.image_path,
            gold_answer=None,
        )
        raw_final_answer = str(result.final_answer)
        prediction = clean_answer(raw_final_answer)
        error = None
        if MAX_TURNS_MARKER in raw_final_answer:
            prediction = ""
            error = "max_turns_reached"

        item = {
            "index": sample.index,
            "sample_id": sample.sample_id,
            "image": sample.image,
            "question": sample.question,
            "prediction": prediction,
            "final_answer": raw_final_answer,
            "final_image": str(result.final_image),
            "gt_answers": sample.answers,
            "answerable": sample.answerable,
            "answer_type": sample.answer_type,
            "num_steps": len(result.steps),
            "tools": [step.tool_name for step in result.steps],
            "steps": [_step_to_json(step) for step in result.steps],
            "trace": [_event_to_json(event) for event in result.memory.events],
            "error": error,
        }
        if include_prompt:
            item["user_request"] = user_request
        return item
    except Exception as exc:
        item = {
            "index": sample.index,
            "sample_id": sample.sample_id,
            "image": sample.image,
            "question": sample.question,
            "prediction": "",
            "gt_answers": sample.answers,
            "answerable": sample.answerable,
            "answer_type": sample.answer_type,
            "num_steps": 0,
            "tools": [],
            "steps": [],
            "trace": [_event_to_json(event) for event in memory.events],
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }
        if include_prompt:
            item["user_request"] = user_request
        return item


def _step_to_json(step: StepResult) -> dict[str, Any]:
    data: dict[str, Any] = {
        "tool_name": step.tool_name,
        "input_image": str(step.input_image),
        "observation": _to_jsonable(step.observation),
    }
    if step.output_image is not None:
        data["output_image"] = str(step.output_image)
    return data


def _event_to_json(event: MemoryEvent) -> dict[str, Any]:
    return {
        "event_type": event.event_type,
        "message": event.message,
        "data": _to_jsonable(event.data),
    }


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    try:
        json.dumps(value, ensure_ascii=False)
    except TypeError:
        return str(value)
    return value


def main() -> None:
    args = parse_args()
    load_local_env()

    backend = args.backend or os.environ.get("VIZWIZ_AGENT_BACKEND")
    backend = backend or os.environ.get("VIZWIZ_PLANNER_BACKEND", "transformers")
    if backend not in ("openai", "transformers"):
        raise ValueError("Backend must be either 'openai' or 'transformers'.")

    workers = max(args.workers, 1)
    if backend == "transformers" and workers != 1:
        raise ValueError("Use --workers 1 with the transformers backend.")
    if args.max_turns <= 0:
        raise ValueError("--max-turns must be positive.")

    if args.output.exists():
        if not args.overwrite:
            raise FileExistsError(
                "Output already exists, pass --overwrite to replace it: "
                f"{args.output}"
            )
        args.output.unlink()

    samples = list(
        iter_vizwiz_samples(
            vizwiz_root=args.vizwiz_root,
            split=args.split,
            start_index=args.start_index,
            limit=args.limit,
            skip_missing=args.skip_missing,
        )
    )

    shared_planner = ImagePlanner(backend=backend) if backend == "transformers" else None
    count = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = []
        for sample in samples:
            futures.append(
                pool.submit(
                    run_one,
                    sample,
                    backend,
                    args.max_turns,
                    args.image_output_dir,
                    args.include_prompt,
                    shared_planner,
                )
            )
            if args.sleep > 0:
                time.sleep(args.sleep)

        for future in as_completed(futures):
            item = future.result()
            append_jsonl(args.output, item)
            count += 1
            print(
                json.dumps(
                    {
                        "count": count,
                        "sample_id": item["sample_id"],
                        "prediction": item.get("prediction", ""),
                        "num_steps": item.get("num_steps", 0),
                        "tools": item.get("tools", []),
                        "error": item["error"],
                    },
                    ensure_ascii=False,
                )
            )

    print(f"Saved {count} agent predictions to {args.output}")


if __name__ == "__main__":
    main()
