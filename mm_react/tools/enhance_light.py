from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from mm_react.env import load_local_env


DEFAULT_API_URL = "http://127.0.0.1:8005/api/enhance"
DEFAULT_TIMEOUT_SECONDS = 900


def run_enhancement_tool(
    input_image: Path,
    tool_call: Any,
    output_image: Path,
) -> str:
    """Run low-light enhancement through the local FastAPI service."""

    _ = tool_call
    load_local_env()
    output_image.parent.mkdir(parents=True, exist_ok=True)
    api_url = os.environ.get("MM_REACT_LOW_LIGHT_API_URL", DEFAULT_API_URL)
    timeout_seconds = int(
        os.environ.get("MM_REACT_TOOL_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS)
    )
    payload = {
        "input_path": str(input_image.expanduser().resolve()),
        "output_path": str(output_image.expanduser().resolve()),
    }
    req = urllib.request.Request(
        str(api_url),
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout_seconds) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            "低光照增强 API 请求失败。请检查输入参数和服务日志。"
            f"HTTP {exc.code}: {detail}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            "低光照增强 API 调用失败。请确认服务已启动，并监听 "
            f"{api_url}。"
        ) from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError("低光照增强 API 返回的内容不是合法 JSON。") from exc

    if not isinstance(result, dict):
        raise RuntimeError(f"低光照增强 API 返回格式无效: {result!r}")

    if result.get("status") not in (None, "success"):
        message = result.get("message") or result.get("detail") or result
        raise RuntimeError(f"低光照增强 API 处理失败: {message}")

    if not output_image.exists():
        api_output = result.get("output") or result.get("output_path") or output_image
        raise RuntimeError(f"低光照增强 API 未生成输出图片: {api_output}")

    return "Low-light image brightening completed."
