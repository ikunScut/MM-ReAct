from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mm_react.env import load_local_env
from evals.vizwiz.data import VizWizSample, append_jsonl, iter_vizwiz_samples
from evals.vizwiz.vlm_client import PlannerVLMClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run VizWiz through ImagePlanner's model backend."
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
        default=PROJECT_ROOT / "outputs" / "vizwiz_planner" / "predictions.jsonl",
    )
    parser.add_argument("--skip-missing", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--backend", choices=["openai", "transformers"], default=None)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel planner calls.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Seconds to sleep between model calls.",
    )
    return parser.parse_args()


def run_one(sample: VizWizSample, client: PlannerVLMClient) -> dict:
    try:
        prediction = client.answer(sample.image_path, sample.question)
        return {
            "index": sample.index,
            "sample_id": sample.sample_id,
            "image": sample.image,
            "question": sample.question,
            "prediction": prediction,
            "gt_answers": sample.answers,
            "answerable": sample.answerable,
            "answer_type": sample.answer_type,
            "error": None,
        }
    except Exception as exc:
        return {
            "index": sample.index,
            "sample_id": sample.sample_id,
            "image": sample.image,
            "question": sample.question,
            "prediction": "",
            "gt_answers": sample.answers,
            "answerable": sample.answerable,
            "answer_type": sample.answer_type,
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }


def main() -> None:
    args = parse_args()
    load_local_env()
    backend = args.backend or os.environ.get("VIZWIZ_PLANNER_BACKEND", "transformers")
    if backend not in ("openai", "transformers"):
        raise ValueError("Backend must be either 'openai' or 'transformers'.")

    workers = max(args.workers, 1)
    if backend == "transformers" and workers != 1:
        raise ValueError("Use --workers 1 with the transformers backend.")

    client = PlannerVLMClient(backend=backend)

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

    count = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = []
        for sample in samples:
            futures.append(pool.submit(run_one, sample, client))
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
                        "error": item["error"],
                    },
                    ensure_ascii=False,
                )
            )

    print(f"Saved {count} predictions to {args.output}")


if __name__ == "__main__":
    main()
