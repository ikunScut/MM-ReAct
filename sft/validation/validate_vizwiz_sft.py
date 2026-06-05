"""Validate VizWiz ReAct SFT JSONL records.

Example:
    python sft/validation/validate_vizwiz_sft.py \
        --data outputs/sft_data/vizwiz_train_react_sft.jsonl \
        --vizwiz-root data/VizWiz \
        --split train
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evals.vizwiz.data import load_annotations, resolve_vizwiz_paths
from mm_react.tools import AVAILABLE_TOOL_NAMES


DEFAULT_DATA = PROJECT_ROOT / "outputs" / "sft_data" / "vizwiz_train_react_sft.jsonl"

REACT_OUTPUT_RE = re.compile(
    r"\A\s*<thought>\s*(?P<thought>.*?)\s*</thought>\s*"
    r"<tool>\s*(?P<tool>.*?)\s*</tool>\s*"
    r"<final_answer>\s*(?P<final_answer>.*?)\s*</final_answer>\s*\Z",
    re.DOTALL,
)
ORIGINAL_IMAGE_RE = re.compile(r"Original input image:\n(?P<path>.+?)(?:\n\n|\Z)")
CURRENT_IMAGE_RE = re.compile(r"Current image:\n(?P<path>.+?)(?:\n\n|\Z)")
MAX_TURN_FAILURE_TEXT = "stopped because the maximum number of ReAct turns was reached"
USER_LEAKAGE_MARKERS = [
    "Teacher-Only Supervision",
    "\nGold answer:\n",
    '"answers"',
    '"majority_answer"',
    '"teacher_instruction"',
    '"answerable"',
    '"answer_type"',
]
ASSISTANT_LEAKAGE_MARKERS = [
    "teacher-only",
    "gold answer",
    "gold answers",
    "dataset annotation",
    "dataset annotations",
    "majority answer",
    "majority answers",
    "majority_answer",
]


@dataclass(frozen=True)
class ValidationIssue:
    line_number: int
    severity: str
    code: str
    message: str


@dataclass(frozen=True)
class ParsedAssistantOutput:
    thought: str
    tool_value: Any
    final_answer: str
    is_final: bool
    tool_name: str | None


@dataclass
class ValidationStats:
    records: int = 0
    final_records: int = 0
    intermediate_records: int = 0
    image_records: int = 0
    max_assistant_chars: int = 0
    max_user_text_chars: int = 0
    tool_counts: Counter[str] = field(default_factory=Counter)
    final_answer_counts: Counter[str] = field(default_factory=Counter)
    records_by_original_image: Counter[str] = field(default_factory=Counter)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate VizWiz ReAct SFT JSONL records."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--vizwiz-root", type=Path, default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--no-check-images",
        action="store_true",
        help="Do not require image paths referenced by SFT records to exist.",
    )
    parser.add_argument(
        "--show-issues",
        type=int,
        default=50,
        help="Maximum number of issues to print. Use 0 to hide issue details.",
    )
    parser.add_argument(
        "--fail-on-warning",
        action="store_true",
        help="Exit non-zero when warnings are present.",
    )
    return parser.parse_args()


def iter_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                yield line_number, {
                    "__json_error__": f"{exc.msg} at column {exc.colno}"
                }
                continue
            if not isinstance(record, dict):
                yield line_number, {"__type_error__": type(record).__name__}
                continue
            yield line_number, record


def load_vizwiz_answers(
    vizwiz_root: Path | None,
    split: str,
) -> dict[str, list[str]]:
    if vizwiz_root is None:
        return {}

    annotations_path, _ = resolve_vizwiz_paths(vizwiz_root, split)
    answers_by_image: dict[str, list[str]] = {}
    for row in load_annotations(annotations_path):
        image = str(row.get("image", ""))
        answers = [
            normalize_answer(str(answer.get("answer", "")))
            for answer in row.get("answers", []) or []
            if normalize_answer(str(answer.get("answer", "")))
        ]
        if image:
            answers_by_image[Path(image).stem] = answers
    return answers_by_image


def validate_record(
    record: dict[str, Any],
    line_number: int,
    check_images: bool,
    stats: ValidationStats,
) -> tuple[list[ValidationIssue], ParsedAssistantOutput | None, str | None]:
    issues: list[ValidationIssue] = []
    parsed_output: ParsedAssistantOutput | None = None
    original_image: str | None = None

    if "__json_error__" in record:
        return [
            issue(line_number, "error", "invalid_json", str(record["__json_error__"]))
        ], None, None
    if "__type_error__" in record:
        return [
            issue(
                line_number,
                "error",
                "invalid_record_type",
                f"Expected JSON object, got {record['__type_error__']}.",
            )
        ], None, None

    messages = record.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        return [
            issue(
                line_number,
                "error",
                "invalid_messages",
                "Expected messages list with at least user and assistant messages.",
            )
        ], None, None

    user_message = messages[0]
    assistant_message = messages[-1]
    if not isinstance(user_message, dict) or user_message.get("role") != "user":
        issues.append(
            issue(line_number, "error", "invalid_user", "First message must be user.")
        )
    if (
        not isinstance(assistant_message, dict)
        or assistant_message.get("role") != "assistant"
    ):
        issues.append(
            issue(
                line_number,
                "error",
                "invalid_assistant",
                "Last message must be assistant.",
            )
        )

    user_text, image_path = extract_user_text_and_image(user_message)
    assistant_text = (
        assistant_message.get("content")
        if isinstance(assistant_message, dict)
        else None
    )
    if not isinstance(assistant_text, str) or not assistant_text.strip():
        issues.append(
            issue(
                line_number,
                "error",
                "invalid_assistant_content",
                "Assistant content must be a non-empty string.",
            )
        )
        assistant_text = ""

    stats.records += 1
    stats.max_user_text_chars = max(stats.max_user_text_chars, len(user_text))
    stats.max_assistant_chars = max(stats.max_assistant_chars, len(assistant_text))

    if image_path is None:
        issues.append(
            issue(line_number, "error", "missing_image", "User content has no image.")
        )
    else:
        stats.image_records += 1
        if check_images and not Path(image_path).exists():
            issues.append(
                issue(
                    line_number,
                    "error",
                    "image_not_found",
                    f"Image path does not exist: {image_path}",
                )
            )

    original_image = extract_runtime_path(user_text, ORIGINAL_IMAGE_RE)
    if original_image:
        stats.records_by_original_image[original_image] += 1
    current_image = extract_runtime_path(user_text, CURRENT_IMAGE_RE)
    if current_image and image_path and current_image != image_path:
        issues.append(
            issue(
                line_number,
                "warning",
                "current_image_mismatch",
                "User image content does not match prompt Current image path.",
            )
        )

    for marker in find_teacher_leakage_markers(user_text, assistant_text):
        issues.append(
            issue(
                line_number,
                "error",
                "teacher_leakage",
                f"Student-visible record contains teacher-only marker: {marker}",
            )
        )

    if "```" in assistant_text:
        issues.append(
            issue(line_number, "error", "markdown_output", "Assistant uses code fences.")
        )

    parsed_output, output_issues = parse_assistant_output(assistant_text, line_number)
    issues.extend(output_issues)
    if parsed_output is not None:
        if parsed_output.is_final:
            stats.final_records += 1
            normalized_final = normalize_answer(parsed_output.final_answer)
            if normalized_final:
                stats.final_answer_counts[normalized_final] += 1
        else:
            stats.intermediate_records += 1
        if parsed_output.tool_name:
            stats.tool_counts[parsed_output.tool_name] += 1

    return issues, parsed_output, original_image


def extract_user_text_and_image(user_message: Any) -> tuple[str, str | None]:
    if not isinstance(user_message, dict):
        return "", None

    content = user_message.get("content")
    if not isinstance(content, list):
        return "", None

    texts: list[str] = []
    image_path: str | None = None
    for item in content:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "text" and isinstance(item.get("text"), str):
            texts.append(item["text"])
        elif item.get("type") == "image" and isinstance(item.get("image"), str):
            if image_path is None:
                image_path = item["image"]
    return "\n".join(texts), image_path


def extract_runtime_path(text: str, pattern: re.Pattern[str]) -> str | None:
    match = pattern.search(text)
    if not match:
        return None
    return match.group("path").strip()


def find_teacher_leakage_markers(
    user_text: str,
    assistant_text: str,
) -> list[str]:
    """Return teacher-only markers while allowing normal anti-leakage policy text."""

    markers: list[str] = []
    lowered_user_text = user_text.lower()
    lowered_assistant_text = assistant_text.lower()

    for marker in USER_LEAKAGE_MARKERS:
        if marker.lower() in lowered_user_text:
            markers.append(marker)
    for marker in ASSISTANT_LEAKAGE_MARKERS:
        if marker.lower() in lowered_assistant_text:
            markers.append(marker)
    return markers


def parse_assistant_output(
    text: str,
    line_number: int,
) -> tuple[ParsedAssistantOutput | None, list[ValidationIssue]]:
    issues: list[ValidationIssue] = []
    match = REACT_OUTPUT_RE.fullmatch(text.strip())
    if not match:
        return None, [
            issue(
                line_number,
                "error",
                "invalid_react_format",
                "Assistant output must contain exactly thought, tool, final_answer tags.",
            )
        ]

    thought = match.group("thought").strip()
    tool_text = match.group("tool").strip()
    final_answer = match.group("final_answer").strip()

    if not thought:
        issues.append(
            issue(line_number, "error", "empty_thought", "Thought must be non-empty.")
        )
    if "\n" in thought:
        issues.append(
            issue(line_number, "warning", "multiline_thought", "Thought is multiline.")
        )
    if len(thought) > 180:
        issues.append(
            issue(line_number, "warning", "long_thought", "Thought is unusually long.")
        )

    try:
        tool_value = json.loads(tool_text)
    except json.JSONDecodeError as exc:
        return None, issues + [
            issue(
                line_number,
                "error",
                "invalid_tool_json",
                f"Tool tag is not valid JSON: {exc.msg}.",
            )
        ]

    tool_name: str | None = None
    if tool_value is None:
        is_final = True
        if not final_answer:
            issues.append(
                issue(
                    line_number,
                    "error",
                    "empty_final_answer",
                    "Final turn must provide final_answer.",
                )
            )
    elif isinstance(tool_value, dict):
        is_final = False
        tool_name_value = tool_value.get("tool_name")
        args_value = tool_value.get("args")
        if not isinstance(tool_name_value, str) or not tool_name_value:
            issues.append(
                issue(
                    line_number,
                    "error",
                    "missing_tool_name",
                    "Tool object must contain tool_name.",
                )
            )
        else:
            tool_name = tool_name_value
            if tool_name not in AVAILABLE_TOOL_NAMES:
                issues.append(
                    issue(
                        line_number,
                        "error",
                        "unknown_tool_name",
                        f"Unknown tool_name: {tool_name}",
                    )
                )
        if not isinstance(args_value, dict):
            issues.append(
                issue(
                    line_number,
                    "error",
                    "invalid_tool_args",
                    "Tool object must contain args as an object.",
                )
            )
        if final_answer:
            issues.append(
                issue(
                    line_number,
                    "error",
                    "intermediate_has_final",
                    "Intermediate tool turn must leave final_answer empty.",
                )
            )
    else:
        return None, issues + [
            issue(
                line_number,
                "error",
                "invalid_tool_value",
                "Tool tag must be either null or a JSON object.",
            )
        ]

    if MAX_TURN_FAILURE_TEXT.lower() in final_answer.lower():
        issues.append(
            issue(
                line_number,
                "error",
                "max_turn_failure",
                "Final answer is the max-turn failure fallback.",
            )
        )

    return (
        ParsedAssistantOutput(
            thought=thought,
            tool_value=tool_value,
            final_answer=final_answer,
            is_final=is_final,
            tool_name=tool_name,
        ),
        issues,
    )


def score_final_answer(final_answer: str, gold_answers: list[str]) -> float:
    if not gold_answers:
        return 0.0
    normalized = normalize_answer(final_answer)
    matches = sum(1 for answer in gold_answers if answer == normalized)
    return min(1.0, matches / 3.0)


def normalize_answer(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"^(a|an|the) ", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()


def issue(
    line_number: int,
    severity: str,
    code: str,
    message: str,
) -> ValidationIssue:
    return ValidationIssue(
        line_number=line_number,
        severity=severity,
        code=code,
        message=message,
    )


def main() -> None:
    args = parse_args()
    data_path = args.data.expanduser().resolve()
    vizwiz_root = (
        args.vizwiz_root.expanduser().resolve() if args.vizwiz_root is not None else None
    )
    check_images = not args.no_check_images

    answers_by_image = load_vizwiz_answers(vizwiz_root, args.split)
    stats = ValidationStats()
    issues: list[ValidationIssue] = []
    issue_counts: Counter[tuple[str, str]] = Counter()
    final_scores: list[float] = []
    final_records_with_gold = 0
    final_records_without_gold = 0

    if not data_path.exists():
        raise FileNotFoundError(f"SFT data file does not exist: {data_path}")

    for line_number, record in iter_jsonl(data_path):
        record_issues, parsed_output, original_image = validate_record(
            record=record,
            line_number=line_number,
            check_images=check_images,
            stats=stats,
        )
        issues.extend(record_issues)
        for record_issue in record_issues:
            issue_counts[(record_issue.severity, record_issue.code)] += 1

        if parsed_output is None or not parsed_output.is_final or not original_image:
            continue
        if not answers_by_image:
            continue

        original_stem = Path(original_image).stem
        gold_answers = answers_by_image.get(original_stem)
        if gold_answers is None:
            final_records_without_gold += 1
            continue
        final_records_with_gold += 1
        final_scores.append(score_final_answer(parsed_output.final_answer, gold_answers))

    error_count = sum(1 for item in issues if item.severity == "error")
    warning_count = sum(1 for item in issues if item.severity == "warning")

    print("VizWiz ReAct SFT validation")
    print("===========================")
    print(f"data: {data_path}")
    print(f"records: {stats.records}")
    print(f"image_records: {stats.image_records}")
    print(f"intermediate_records: {stats.intermediate_records}")
    print(f"final_records: {stats.final_records}")
    print(f"errors: {error_count}")
    print(f"warnings: {warning_count}")
    print(f"unique_original_images: {len(stats.records_by_original_image)}")
    print(f"max_user_text_chars: {stats.max_user_text_chars}")
    print(f"max_assistant_chars: {stats.max_assistant_chars}")

    if stats.records_by_original_image:
        turns = list(stats.records_by_original_image.values())
        print(f"turns_per_original_image_min: {min(turns)}")
        print(f"turns_per_original_image_max: {max(turns)}")
        print(f"turns_per_original_image_avg: {sum(turns) / len(turns):.2f}")

    if stats.tool_counts:
        print("")
        print("Tool counts")
        for tool_name, count in stats.tool_counts.most_common():
            print(f"{tool_name}: {count}")

    if stats.final_answer_counts:
        print("")
        print("Top final answers")
        for answer, count in stats.final_answer_counts.most_common(20):
            print(f"{answer or '<empty>'}: {count}")

    if final_scores:
        accuracy = sum(final_scores) / len(final_scores)
        print("")
        print("VizWiz answer check")
        print(f"final_records_with_gold: {final_records_with_gold}")
        print(f"final_records_without_gold: {final_records_without_gold}")
        print(f"approx_vqa_accuracy: {accuracy:.4f}")

    if issue_counts:
        print("")
        print("Issue counts")
        for (severity, code), count in sorted(issue_counts.items()):
            print(f"{severity}.{code}: {count}")

    if args.show_issues > 0 and issues:
        print("")
        print("Issues")
        for item in issues[: args.show_issues]:
            print(
                f"{item.severity.upper()} line {item.line_number} "
                f"{item.code}: {item.message}"
            )
        if len(issues) > args.show_issues:
            print(f"... {len(issues) - args.show_issues} more issues not shown")

    if error_count or (warning_count and args.fail_on_warning):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
