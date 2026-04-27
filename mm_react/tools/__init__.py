"""Central tool registry for planner prompts and executor dispatch."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from .demo_tool import run_demo_tool
from .mock_tool import run_mock_image_tool

ToolFn = Callable[[Path, Any, Path], dict[str, Any]]

DEFAULT_TOOL_NAMES = (
    "deblur",
    "denoise",
    "low_light_enhance",
    "super_resolution",
    "color_enhance",
    "face_restore",
)

MOCK_TOOL_REGISTRY: dict[str, ToolFn] = {
    tool_name: run_mock_image_tool for tool_name in DEFAULT_TOOL_NAMES
}
DEMO_TOOL_REGISTRY: dict[str, ToolFn] = {
    tool_name: run_demo_tool for tool_name in DEFAULT_TOOL_NAMES
}

REAL_TOOL_REGISTRY: dict[str, ToolFn] = {}

DEFAULT_TOOL_REGISTRY = MOCK_TOOL_REGISTRY
TOOL_REGISTRY = DEFAULT_TOOL_REGISTRY
AVAILABLE_TOOL_NAMES = set(DEFAULT_TOOL_REGISTRY)

__all__ = [
    "AVAILABLE_TOOL_NAMES",
    "DEFAULT_TOOL_NAMES",
    "DEFAULT_TOOL_REGISTRY",
    "DEMO_TOOL_REGISTRY",
    "MOCK_TOOL_REGISTRY",
    "REAL_TOOL_REGISTRY",
    "TOOL_REGISTRY",
    "ToolFn",
    "run_demo_tool",
    "run_mock_image_tool",
]
