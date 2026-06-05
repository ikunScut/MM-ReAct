"""Generate VizWiz ReAct SFT data with an OpenAI teacher planner.

The script runs the full MM-ReAct agent on VizWiz samples. Each teacher
planner turn is recorded by AgentMemory as one student-visible SFT example.

Example:
    python sft/generation/generate_vizwiz_sft.py \
        --vizwiz-root data/VizWiz \
        --split train \
        --output outputs/sft_data/vizwiz_train_react_sft.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import traceback
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evals.vizwiz.data import VizWizSample, append_jsonl, iter_vizwiz_samples
from mm_react.agent.executor import ImageExecutor
from mm_react.agent.memory import AgentMemory
from mm_react.agent.planner import ImagePlanner
from mm_react.agent.react_agent import ReActAgent
from mm_react.env import load_local_env


DEFAULT_VIZWIZ_ROOT = PROJECT_ROOT / "data" / "VizWiz"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "sft_data" / "vizwiz_train_react_sft.jsonl"
DEFAULT_IMAGE_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "sft_generation" / "images"
DEFAULT_ERROR_LOG = PROJECT_ROOT / "outputs" / "sft_generation" / "errors.jsonl"


@dataclass(frozen=True)
class GenerationConfig:
    vizwiz_root: Path
    split: str
    output: Path
    image_output_dir: Path
    error_log: Path
    start_index: int
    limit: int
    num_workers: int
    max_turns: int
    skip_missing: bool
    stop_on_error: bool
    save_traces: bool
    trace_dir: Path


def parse_args() -> GenerationConfig:
    parser = argparse.ArgumentParser(
        description="Generate VizWiz ReAct SFT JSONL with the OpenAI teacher planner."
    )
    parser.add_argument("--vizwiz-root", type=Path, default=DEFAULT_VIZWIZ_ROOT)
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--image-output-dir", type=Path, default=DEFAULT_IMAGE_OUTPUT_DIR)
    parser.add_argument("--error-log", type=Path, default=DEFAULT_ERROR_LOG)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Number of dataset samples to process. Use 0 for all remaining samples.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of samples to process concurrently. Use 1 for serial execution.",
    )
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument("--skip-missing", action="store_true")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Log sample errors and continue instead of failing fast.",
    )
    parser.add_argument("--save-traces", action="store_true")
    parser.add_argument(
        "--trace-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "sft_generation" / "traces",
    )
    args = parser.parse_args()
    return GenerationConfig(
        vizwiz_root=args.vizwiz_root.expanduser().resolve(),
        split=args.split,
        output=args.output.expanduser().resolve(),
        image_output_dir=args.image_output_dir.expanduser().resolve(),
        error_log=args.error_log.expanduser().resolve(),
        start_index=args.start_index,
        limit=args.limit,
        num_workers=max(1, args.num_workers),
        max_turns=args.max_turns,
        skip_missing=args.skip_missing,
        stop_on_error=not args.continue_on_error,
        save_traces=args.save_traces,
        trace_dir=args.trace_dir.expanduser().resolve(),
    )


def build_user_request(sample: VizWizSample) -> str:
    return (
        f"{sample.question}\n"
        "When the provided information is insufficient, respond with 'Unanswerable'.\n"
        "Answer the question using a single word or phrase."
    )


def build_gold_answer(sample: VizWizSample) -> dict[str, Any]:
    answers = [
        str(item.get("answer", "")).strip()
        for item in sample.answers
        if str(item.get("answer", "")).strip()
    ]
    answer_counts = Counter(answers)
    majority_answer = answer_counts.most_common(1)[0][0] if answer_counts else ""
    return {
        "majority_answer": majority_answer,
        "answers": sample.answers,
        "answerable": sample.answerable,
        "answer_type": sample.answer_type,
        "teacher_instruction": (
            "Use gold answers only as teacher-only supervision. "
            "Do not expose or derive visible outputs from them; visible outputs "
            "must be grounded in image evidence and tool observations."
        ),
    }


def run_one_sample(
    sample: VizWizSample,
    planner: ImagePlanner,
    config: GenerationConfig,
) -> tuple[list[dict[str, Any]], str]:
    memory = AgentMemory()
    executor = ImageExecutor(output_dir=config.image_output_dir)
    agent = ReActAgent(
        planner=planner,
        executor=executor,
        memory=memory,
        max_turns=config.max_turns,
    )
    result = agent.run(
        user_request=build_user_request(sample),
        input_image=sample.image_path,
        gold_answer=build_gold_answer(sample),
    )
    records = result.memory.to_sft_records()

    if config.save_traces:
        result.memory.save_trace(config.trace_dir / f"{sample.sample_id}.txt")

    return records, result.final_answer


_WORKER_STATE = threading.local()


def run_one_sample_in_worker(
    sample: VizWizSample,
    config: GenerationConfig,
) -> tuple[list[dict[str, Any]], str]:
    planner = getattr(_WORKER_STATE, "planner", None)
    if planner is None:
        planner = ImagePlanner(backend="openai")
        _WORKER_STATE.planner = planner
    return run_one_sample(sample, planner, config)


def initialize_output(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


def append_sft_records(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def log_error(path: Path, sample: VizWizSample, exc: BaseException) -> None:
    append_jsonl(
        path,
        {
            "sample_index": sample.index,
            "sample_id": sample.sample_id,
            "image": sample.image,
            "question": sample.question,
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        },
    )


def main() -> None:
    config = parse_args()
    load_local_env()

    planner = ImagePlanner(backend="openai")

    processed_samples = 0
    written_records = 0
    skipped_samples = 0
    failed_samples = 0

    print("VizWiz ReAct SFT generation")
    print("===========================")
    print(f"vizwiz_root: {config.vizwiz_root}")
    print(f"split: {config.split}")
    print(f"output: {config.output}")
    print(f"image_output_dir: {config.image_output_dir}")
    print(f"start_index: {config.start_index}")
    print(f"limit: {config.limit}")
    print(f"num_workers: {config.num_workers}")
    print(f"max_turns: {config.max_turns}")

    samples = list(iter_vizwiz_samples(
        vizwiz_root=config.vizwiz_root,
        split=config.split,
        start_index=config.start_index,
        limit=config.limit,
        skip_missing=config.skip_missing,
    ))
    initialize_output(config.output)

    if config.num_workers == 1:
        for sample in samples:
            try:
                records, final_answer = run_one_sample(sample, planner, config)
            except Exception as exc:
                failed_samples += 1
                log_error(config.error_log, sample, exc)
                print(f"[error] {sample.index} {sample.sample_id}: {exc!r}")
                if config.stop_on_error:
                    raise
                continue

            append_sft_records(config.output, records)
            processed_samples += 1
            written_records += len(records)
            print(
                f"[ok] {sample.index} {sample.sample_id}: "
                f"{len(records)} records, final={final_answer!r}"
            )
    else:
        with ThreadPoolExecutor(max_workers=config.num_workers) as executor:
            futures = {
                executor.submit(run_one_sample_in_worker, sample, config): sample
                for sample in samples
            }
            for future in as_completed(futures):
                sample = futures[future]
                try:
                    records, final_answer = future.result()
                except Exception as exc:
                    failed_samples += 1
                    log_error(config.error_log, sample, exc)
                    print(f"[error] {sample.index} {sample.sample_id}: {exc!r}")
                    if config.stop_on_error:
                        for pending in futures:
                            pending.cancel()
                        raise
                    continue

                append_sft_records(config.output, records)
                processed_samples += 1
                written_records += len(records)
                print(
                    f"[ok] {sample.index} {sample.sample_id}: "
                    f"{len(records)} records, final={final_answer!r}"
                )

    print("")
    print("Done")
    print(f"processed_samples: {processed_samples}")
    print(f"skipped_samples: {skipped_samples}")
    print(f"failed_samples: {failed_samples}")
    print(f"written_records: {written_records}")
    print(f"output: {config.output}")


if __name__ == "__main__":
    main()
