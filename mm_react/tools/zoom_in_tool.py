from __future__ import annotations

from pathlib import Path
from typing import Any


DEFAULT_PADDING_RATIO = 0.2
DEFAULT_MIN_OUTPUT_SIZE = 768


def run_zoom_in_tool(
    input_image: Path,
    tool_call: Any,
    output_image: Path,
) -> str:
    """Crop and enlarge a region of an image from a pixel bounding box."""

    tool_args = getattr(tool_call, "args", {}) or {}
    box = _box_from_tool_args(tool_args)
    padding_ratio = max(
        0.0, float(tool_args.get("padding_ratio", DEFAULT_PADDING_RATIO))
    )
    min_output_size = max(
        1, int(tool_args.get("min_output_size", DEFAULT_MIN_OUTPUT_SIZE))
    )
    image_module = _load_image_module()

    output_image.parent.mkdir(parents=True, exist_ok=True)

    with image_module.open(input_image.expanduser()) as image:
        image = image.convert("RGB")
        image_width, image_height = image.size
        crop_box = _expanded_clamped_box(
            box=box,
            padding_ratio=padding_ratio,
            image_width=image_width,
            image_height=image_height,
        )
        crop = image.crop(tuple(crop_box))
        resized, _ = _resize_to_min_long_side(crop, min_output_size, image_module)
        resized.save(output_image.expanduser())

    return "Image crop and zoom-in enlargement completed."


def _load_image_module() -> Any:
    try:
        from PIL import Image
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "zoom_in_image requires Pillow. Install it with `pip install Pillow` "
            "in the MM-ReAct runtime environment."
        ) from exc
    return Image


def _box_from_tool_args(tool_args: dict[str, Any]) -> list[int]:
    raw_box = (
        tool_args.get("box")
        or tool_args.get("bbox")
        or tool_args.get("crop_box")
        or tool_args.get("region")
    )
    if raw_box is None:
        raise ValueError(
            "zoom_in_image requires a box argument: [x_min, y_min, x_max, y_max]."
        )
    if not isinstance(raw_box, (list, tuple)) or len(raw_box) != 4:
        raise ValueError(
            "zoom_in_image box must be a list of four numbers: "
            "[x_min, y_min, x_max, y_max]."
        )

    try:
        x_min, y_min, x_max, y_max = [int(round(float(value))) for value in raw_box]
    except (TypeError, ValueError) as exc:
        raise ValueError("zoom_in_image box values must be numeric.") from exc

    if x_max <= x_min or y_max <= y_min:
        raise ValueError(
            "zoom_in_image box must satisfy x_max > x_min and y_max > y_min."
        )
    return [x_min, y_min, x_max, y_max]


def _expanded_clamped_box(
    box: list[int],
    padding_ratio: float,
    image_width: int,
    image_height: int,
) -> list[int]:
    x_min, y_min, x_max, y_max = box
    box_width = x_max - x_min
    box_height = y_max - y_min
    pad_x = int(round(box_width * padding_ratio))
    pad_y = int(round(box_height * padding_ratio))

    crop_box = [
        max(0, x_min - pad_x),
        max(0, y_min - pad_y),
        min(image_width, x_max + pad_x),
        min(image_height, y_max + pad_y),
    ]
    if crop_box[2] <= crop_box[0] or crop_box[3] <= crop_box[1]:
        raise ValueError(
            "zoom_in_image box is outside the image after clamping to bounds."
        )
    return crop_box


def _resize_to_min_long_side(
    image: Any,
    min_output_size: int,
    image_module: Any,
) -> tuple[Any, float]:
    long_side = max(image.size)
    if long_side >= min_output_size:
        return image, 1.0

    scale = min_output_size / long_side
    new_size = (
        max(1, int(round(image.width * scale))),
        max(1, int(round(image.height * scale))),
    )
    resampling = getattr(image_module, "Resampling", image_module).LANCZOS
    return image.resize(new_size, resampling), scale
