from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class VizWizSample:
    index: int
    image: str
    image_path: Path
    question: str
    answers: list[dict[str, Any]]
    answerable: int | None
    answer_type: str | None

    @property
    def sample_id(self) -> str:
        return Path(self.image).stem


def load_annotations(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}, got {type(data)!r}.")
    return data


def _first_existing_path(paths: list[Path], description: str) -> Path:
    for path in paths:
        if path.exists():
            return path

    tried = "\n".join(f"  - {path}" for path in paths)
    raise FileNotFoundError(f"VizWiz {description} not found. Tried:\n{tried}")


def resolve_vizwiz_paths(vizwiz_root: Path, split: str) -> tuple[Path, Path]:
    annotations_path = _first_existing_path(
        [
            vizwiz_root / "annotations" / f"{split}.json",
            vizwiz_root / "Annotations" / f"{split}.json",
        ],
        f"annotations for split {split!r}",
    )
    images_dir = _first_existing_path(
        [
            vizwiz_root / "images" / split,
            vizwiz_root / "Images" / split,
            vizwiz_root / split,
        ],
        f"image directory for split {split!r}",
    )
    return annotations_path, images_dir


def iter_vizwiz_samples(
    vizwiz_root: Path,
    split: str,
    start_index: int = 0,
    limit: int = 0,
    skip_missing: bool = False,
) -> Iterable[VizWizSample]:
    annotations_path, images_dir = resolve_vizwiz_paths(vizwiz_root, split)
    annotations = load_annotations(annotations_path)

    end_index = len(annotations) if limit <= 0 else min(len(annotations), start_index + limit)
    for index in range(start_index, end_index):
        row = annotations[index]
        image = str(row["image"])
        image_path = images_dir / image
        if not image_path.exists():
            if skip_missing:
                continue
            raise FileNotFoundError(f"VizWiz image not found: {image_path}")

        yield VizWizSample(
            index=index,
            image=image,
            image_path=image_path,
            question=str(row.get("question", "")).strip(),
            answers=list(row.get("answers") or []),
            answerable=row.get("answerable"),
            answer_type=row.get("answer_type"),
        )


def append_jsonl(path: Path, item: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
    return items
