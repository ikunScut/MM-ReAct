"""Central tool registry for planner prompts and executor dispatch."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from .enhance_light import run_enhancement_tool
from .grounding_dino_tool import run_grounding_dino_tool
from .nafnet_tool import run_nafnet_tool
from .rotate_tool import run_rotate_tool
from .super_resolution_tool import run_super_resolution_tool
from .zoom_in_tool import run_zoom_in_tool

ToolObservation = str | dict[str, Any] | list[Any]
ToolFn = Callable[[Path, Any, Path], ToolObservation]

TOOL_REGISTRY: dict[str, ToolFn] = {
    "object_detection_image": run_grounding_dino_tool,
    "low_light_enhance": run_enhancement_tool,
    "nafnet_image_restoration": run_nafnet_tool,
    "rotate_image": run_rotate_tool,
    "super_resolution_image": run_super_resolution_tool,
    "zoom_in_image": run_zoom_in_tool,
}
AVAILABLE_TOOL_NAMES = set(TOOL_REGISTRY)

__all__ = [
    "AVAILABLE_TOOL_NAMES",
    "TOOL_REGISTRY",
    "ToolFn",
    "ToolObservation",
    "run_enhancement_tool",
    "run_grounding_dino_tool",
    "run_nafnet_tool",
    "run_rotate_tool",
    "run_super_resolution_tool",
    "run_zoom_in_tool",
]
