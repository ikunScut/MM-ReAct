"""Central tool registry for planner prompts and executor dispatch."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from .demo_tool import run_demo_tool
from .mock_tool import run_mock_image_tool
from .zerodce_tool import run_zerodce_tool

ToolObservation = str | dict[str, Any] | list[Any]
ToolFn = Callable[[Path, Any, Path], ToolObservation]

ZERODCE_TOOL_NAMES = ("low_light_enhance",)
DEFAULT_TOOL_NAMES = ZERODCE_TOOL_NAMES

MOCK_TOOL_REGISTRY: dict[str, ToolFn] = {
    tool_name: run_mock_image_tool for tool_name in DEFAULT_TOOL_NAMES
}
DEMO_TOOL_REGISTRY: dict[str, ToolFn] = {
    tool_name: run_demo_tool for tool_name in DEFAULT_TOOL_NAMES
}
ZERODCE_TOOL_REGISTRY: dict[str, ToolFn] = {
    "low_light_enhance": run_zerodce_tool,
}

REAL_TOOL_REGISTRY = ZERODCE_TOOL_REGISTRY

# 工具注册 AVAILABLE_TOOL_NAMES 供外界得到能用的工具名字
DEFAULT_TOOL_REGISTRY = REAL_TOOL_REGISTRY
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
    "ToolObservation",
    "ZERODCE_TOOL_REGISTRY",
    "ZERODCE_TOOL_NAMES",
    "run_demo_tool",
    "run_mock_image_tool",
    "run_zerodce_tool",
]
