from __future__ import annotations

import contextlib
import io
import sys
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
VIZWIZ_API_ROOT = PROJECT_ROOT / "data" / "VizWiz" / "API"
VIZWIZ_HELPER_TOOLS = VIZWIZ_API_ROOT / "PythonHelperTools" / "vqaTools"
VIZWIZ_EVAL_TOOLS = VIZWIZ_API_ROOT / "PythonEvaluationTools"


def _ensure_vizwiz_api_on_path() -> None:
    for path in (VIZWIZ_HELPER_TOOLS, VIZWIZ_EVAL_TOOLS):
        if not path.exists():
            raise FileNotFoundError(f"VizWiz API path not found: {path}")
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def _load_vizwiz_api() -> tuple[type[Any], type[Any]]:
    _ensure_vizwiz_api_on_path()
    from vqa import VQA  # type: ignore
    from vqaEvaluation.vqaEval import VQAEval  # type: ignore

    return VQA, VQAEval


def _make_vqa(rows: list[dict[str, Any]]) -> Any:
    VQA, _ = _load_vizwiz_api()
    vqa = VQA()
    vqa.dataset = rows
    vqa.imgToQA = {row["image"]: row for row in rows}
    return vqa


def _prediction_image(item: dict[str, Any], index: int) -> str:
    image = item.get("image") or item.get("image_id") or item.get("sample_id")
    if image:
        return str(image)
    return f"prediction_{index:08d}"


def _prediction_answerable(item: dict[str, Any]) -> float:
    value = (
        item.get("predicted_answerable")
        if "predicted_answerable" in item
        else item.get("answerable_prediction")
    )
    if value is None:
        return 1.0
    if isinstance(value, bool):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 1.0


def _annotation_by_image(
    annotation_rows: Iterable[dict[str, Any]] | None,
) -> dict[str, dict[str, Any]]:
    if annotation_rows is None:
        return {}
    return {str(row["image"]): row for row in annotation_rows if "image" in row}


def _coerce_eval_rows(
    items: list[dict[str, Any]],
    annotation_rows: Iterable[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    annotations = _annotation_by_image(annotation_rows)
    gt_rows: list[dict[str, Any]] = []
    result_rows: list[dict[str, Any]] = []
    error_count = 0

    for index, item in enumerate(items):
        if item.get("error"):
            error_count += 1
            continue

        image = _prediction_image(item, index)
        annotation = annotations.get(image, {})
        answers = item.get("gt_answers") or item.get("answers") or annotation.get("answers")
        if not answers:
            continue

        question = item.get("question") or annotation.get("question") or ""
        answer_type = item.get("answer_type") or annotation.get("answer_type") or "unknown"
        answerable = item.get("answerable", annotation.get("answerable"))
        if answerable is None:
            answerable = 1

        gt_rows.append(
            {
                "image": image,
                "question": str(question),
                "answers": list(answers),
                "answer_type": str(answer_type),
                "answerable": int(answerable),
            }
        )
        result_rows.append(
            {
                "image": image,
                "answer": str(item.get("prediction", "")),
                "answerable": _prediction_answerable(item),
            }
        )

    return gt_rows, result_rows, error_count


@lru_cache(maxsize=1)
def _official_normalizer() -> Any:
    _, VQAEval = _load_vizwiz_api()
    return VQAEval(_make_vqa([]), _make_vqa([]))


def normalize_answer(answer: str) -> str:
    """Normalize with the VizWiz official evaluator's answer processing."""
    evaluator = _official_normalizer()
    text = str(answer).replace("\n", " ").replace("\t", " ").strip()
    text = evaluator.processPunctuation(text)
    return evaluator.processDigitArticle(text)


def _evaluate_accuracy_only(vqa_eval: Any, imgs: list[str]) -> None:
    acc_qa: list[float] = []
    acc_ans_type: dict[str, list[float]] = {}

    for img in imgs:
        gt = vqa_eval.vqa.imgToQA[img]
        res = vqa_eval.vqaRes.imgToQA[img]

        res_ans = str(res["answer"]).replace("\n", " ").replace("\t", " ").strip()
        res_ans = vqa_eval.processPunctuation(res_ans)
        res_ans = vqa_eval.processDigitArticle(res_ans)

        gt_acc = []
        for i, _ in enumerate(gt["answers"]):
            other_gt_answers = [
                item for j, item in enumerate(gt["answers"]) if i != j
            ]
            matching_answers = [
                item for item in other_gt_answers if item["answer"] == res_ans
            ]
            gt_acc.append(min(1, float(len(matching_answers)) / 3))

        avg_gt_acc = float(sum(gt_acc)) / len(gt_acc)
        answer_type = gt["answer_type"]
        acc_ans_type.setdefault(answer_type, []).append(avg_gt_acc)
        acc_qa.append(avg_gt_acc)
        vqa_eval.setEvalQA(img, avg_gt_acc)
        vqa_eval.setEvalAnsType(img, answer_type, avg_gt_acc)

    vqa_eval.setAccuracy(acc_qa, acc_ans_type)


def vqa_accuracy(prediction: str, answers: list[dict[str, Any]]) -> float:
    rows = [
        {
            "image": "single",
            "prediction": prediction,
            "gt_answers": answers,
            "answer_type": "unknown",
            "answerable": 1,
        }
    ]
    result = evaluate_predictions(rows)
    return float(result["accuracy"])


def evaluate_predictions(
    items: list[dict[str, Any]],
    annotation_rows: Iterable[dict[str, Any]] | None = None,
    include_caption_metrics: bool = False,
) -> dict[str, Any]:
    gt_rows, result_rows, error_count = _coerce_eval_rows(items, annotation_rows)
    total = len(gt_rows)
    if total == 0:
        return {
            "total_rows": len(items),
            "count": 0,
            "error_count": error_count,
            "accuracy": 0.0,
            "accuracy_percent": 0.0,
            "official_accuracy": {"overall": 0.0, "perAnswerType": {}},
            "by_answer_type": {},
            "by_answerable": {},
            "top_predictions": [],
            "eval_qa": {},
        }

    _, VQAEval = _load_vizwiz_api()
    vqa_eval = VQAEval(_make_vqa(gt_rows), _make_vqa(result_rows), n=2)
    imgs = [row["image"] for row in gt_rows]

    with contextlib.redirect_stdout(io.StringIO()):
        if include_caption_metrics:
            vqa_eval.evaluate(imgs=imgs)
        else:
            _evaluate_accuracy_only(vqa_eval, imgs)

    by_answerable: dict[str, list[float]] = defaultdict(list)
    prediction_counter: Counter[str] = Counter()
    for row, result_row in zip(gt_rows, result_rows):
        score = float(vqa_eval.evalQA[row["image"]]) / 100.0
        by_answerable[str(row["answerable"])].append(score)
        prediction_counter[normalize_answer(result_row["answer"])] += 1

    def average(scores: list[float]) -> float:
        return sum(scores) / len(scores) if scores else 0.0

    by_answer_type = {
        key: {
            "count": sum(1 for row in gt_rows if row["answer_type"] == key),
            "accuracy": value / 100.0,
            "accuracy_percent": value,
        }
        for key, value in sorted(vqa_eval.accuracy["perAnswerType"].items())
    }

    result: dict[str, Any] = {
        "total_rows": len(items),
        "count": total,
        "error_count": error_count,
        "accuracy": float(vqa_eval.accuracy["overall"]) / 100.0,
        "accuracy_percent": float(vqa_eval.accuracy["overall"]),
        "official_accuracy": vqa_eval.accuracy,
        "by_answer_type": by_answer_type,
        "by_answerable": {
            key: {"count": len(scores), "accuracy": average(scores)}
            for key, scores in sorted(by_answerable.items())
        },
        "top_predictions": prediction_counter.most_common(20),
        "eval_qa": vqa_eval.evalQA,
    }
    if include_caption_metrics:
        result["caption_metrics"] = vqa_eval.caption_metric

    return result
