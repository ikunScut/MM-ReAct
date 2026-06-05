from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evals.vizwiz.data import load_annotations, load_jsonl
from evals.vizwiz.metrics import evaluate_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate VizWiz prediction JSONL.")
    parser.add_argument("predictions", type=Path)
    parser.add_argument(
        "--annotations",
        type=Path,
        default=None,
        help="Optional VizWiz annotation JSON. Useful when predictions lack gt_answers.",
    )
    parser.add_argument(
        "--caption-metrics",
        action="store_true",
        help="Also run the VizWiz API caption metrics. Requires Java/COCO eval dependencies.",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    items = load_jsonl(args.predictions)
    annotations = load_annotations(args.annotations) if args.annotations else None
    result = evaluate_predictions(
        items,
        annotation_rows=annotations,
        include_caption_metrics=args.caption_metrics,
    )
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
