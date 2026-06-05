from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from mm_react.env import load_local_env


DEFAULT_API_URL = "http://127.0.0.1:8001/nafnet/process"
DEFAULT_TIMEOUT_SECONDS = 900
SUPPORTED_MODES = {"denoise", "deblur"}


def run_nafnet_tool(input_image: Path, tool_call: Any, output_image: Path) -> str:
    """Run NAFNet denoise/deblur through the local FastAPI service."""

    load_local_env()
    output_image.parent.mkdir(parents=True, exist_ok=True)
    tool_args = getattr(tool_call, "args", {}) or {}
    mode = _mode_from_tool_call(tool_call, tool_args)
    api_url = os.environ.get("MM_REACT_NAFNET_API_URL", DEFAULT_API_URL)
    timeout_seconds = int(
        os.environ.get("MM_REACT_TOOL_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS)
    )
    payload = json.dumps(
        {
            "input_path": str(input_image.expanduser().resolve()),
            "output_path": str(output_image.expanduser().resolve()),
            "mode": mode,
        },
        ensure_ascii=False,
    ).encode("utf-8")
    request = urllib.request.Request(
        str(api_url),
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            "NAFNet API 请求失败。请检查输入参数和服务日志。"
            f"HTTP {exc.code}: {detail}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            "NAFNet API 调用失败。请确认服务已启动，并监听 "
            f"{api_url}。"
        ) from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError("NAFNet API 返回的内容不是合法 JSON。") from exc

    if not isinstance(result, dict):
        raise RuntimeError(f"NAFNet API 返回格式无效: {result!r}")

    if result.get("status") != "success":
        message = result.get("message") or result.get("detail") or result
        raise RuntimeError(f"NAFNet API 处理失败: {message}")

    if not output_image.exists():
        api_output = result.get("output", str(output_image))
        raise RuntimeError(f"NAFNet API 未生成输出图片: {api_output}")

    if mode == "denoise":
        return "Image denoising completed."
    return "Image deblurring completed."


def _mode_from_tool_call(tool_call: Any, tool_args: dict[str, Any]) -> str:
    tool_name = getattr(tool_call, "tool_name", "")
    if tool_name in SUPPORTED_MODES:
        return tool_name

    mode = str(tool_args.get("mode", "denoise")).strip().lower()
    if mode not in SUPPORTED_MODES:
        raise ValueError(
            f"NAFNet mode must be one of {sorted(SUPPORTED_MODES)}, got {mode!r}."
        )
    return mode
