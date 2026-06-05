from __future__ import annotations

from pathlib import Path
from typing import Any


DEFAULT_DIRECTION = "clockwise"
DEFAULT_EXPAND = True


def run_rotate_tool(
    input_image: Path,
    tool_call: Any,
    output_image: Path,
) -> str:
    """Rotate the current image by a requested angle."""

    tool_args = getattr(tool_call, "args", {}) or {}
    angle = _angle_from_tool_args(tool_args)
    direction = _direction_from_tool_args(tool_args)
    expand = _bool_from_tool_args(tool_args.get("expand", DEFAULT_EXPAND))

    image_module, image_ops_module, image_color_module = _load_pillow_modules()
    output_image.parent.mkdir(parents=True, exist_ok=True)

    with image_module.open(input_image.expanduser()) as image:
        image = image_ops_module.exif_transpose(image)
        image = _normalize_mode_for_rotation(image)
        fill_color = _fill_color_from_tool_args(
            tool_args=tool_args,
            image=image,
            image_color_module=image_color_module,
        )
        pil_angle = -angle if direction == "clockwise" else angle
        resampling = getattr(image_module, "Resampling", image_module).BICUBIC
        rotated = image.rotate(
            pil_angle,
            resample=resampling,
            expand=expand,
            fillcolor=fill_color,
        )
        rotated = _normalize_for_output_suffix(rotated, output_image)
        rotated.save(output_image.expanduser())

    return (
        f"Image rotation by {angle:g} degrees {direction} completed. "
        "This observation only confirms the pixel transform; it does not verify "
        "that the image is now upright or correctly oriented."
    )


def _load_pillow_modules() -> tuple[Any, Any, Any]:
    try:
        from PIL import Image, ImageColor, ImageOps
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "rotate_image requires Pillow. Install it with `pip install Pillow` "
            "in the MM-ReAct runtime environment."
        ) from exc
    return Image, ImageOps, ImageColor


def _angle_from_tool_args(tool_args: dict[str, Any]) -> float:
    raw_angle = (
        tool_args.get("angle")
        if "angle" in tool_args
        else tool_args.get("degrees", tool_args.get("rotation_angle"))
    )
    if raw_angle is None:
        raise ValueError("rotate_image requires an angle argument in degrees.")
    try:
        return float(raw_angle)
    except (TypeError, ValueError) as exc:
        raise ValueError("rotate_image angle must be numeric.") from exc


def _direction_from_tool_args(tool_args: dict[str, Any]) -> str:
    raw_direction = str(tool_args.get("direction", DEFAULT_DIRECTION)).strip().lower()
    clockwise_aliases = {"clockwise", "cw", "right", "rightward", "顺时针", "右转"}
    counterclockwise_aliases = {
        "counterclockwise",
        "counter-clockwise",
        "anticlockwise",
        "ccw",
        "left",
        "leftward",
        "逆时针",
        "左转",
    }
    if raw_direction in clockwise_aliases:
        return "clockwise"
    if raw_direction in counterclockwise_aliases:
        return "counterclockwise"
    raise ValueError(
        "rotate_image direction must be 'clockwise' or 'counterclockwise'."
    )


def _bool_from_tool_args(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y", "on"}:
            return True
        if normalized in {"false", "0", "no", "n", "off"}:
            return False
    return bool(value)


def _fill_color_from_tool_args(
    tool_args: dict[str, Any],
    image: Any,
    image_color_module: Any,
) -> Any:
    raw_fill_color = tool_args.get("fill_color")
    if raw_fill_color is None:
        if image.mode in {"RGBA", "LA"} or "transparency" in image.info:
            return (0, 0, 0, 0)
        return "white"

    if isinstance(raw_fill_color, str):
        try:
            return image_color_module.getcolor(raw_fill_color, image.mode)
        except ValueError as exc:
            raise ValueError(
                f"rotate_image fill_color is not a valid color: {raw_fill_color!r}."
            ) from exc

    if isinstance(raw_fill_color, (list, tuple)):
        try:
            return tuple(int(value) for value in raw_fill_color)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "rotate_image fill_color list values must be integers."
            ) from exc

    raise ValueError("rotate_image fill_color must be a color string or RGB/RGBA list.")


def _normalize_mode_for_rotation(image: Any) -> Any:
    if image.mode in {"P", "LA"} and (
        image.mode == "LA" or "transparency" in image.info
    ):
        return image.convert("RGBA")
    return image


def _normalize_for_output_suffix(image: Any, output_image: Path) -> Any:
    if output_image.suffix.lower() in {".jpg", ".jpeg"} and image.mode in {
        "RGBA",
        "LA",
        "P",
    }:
        return image.convert("RGB")
    return image
