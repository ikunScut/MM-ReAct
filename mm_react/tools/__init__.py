"""Tool implementations that can be registered with the image executor."""

from .demo_tool import run_demo_tool

DEMO_TOOL_REGISTRY = {
    "demo": run_demo_tool,
}

__all__ = ["DEMO_TOOL_REGISTRY", "run_demo_tool"]
