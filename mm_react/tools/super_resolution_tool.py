from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from mm_react.env import load_local_env


DEFAULT_API_URL = "http://127.0.0.1:8002/enhance"
DEFAULT_TIMEOUT_SECONDS = 900

_SUPER_RESOLUTION_LOCK = threading.Lock()


def run_super_resolution_tool(
    input_image: Path, tool_call: Any, output_image: Path
) -> str:
    """Run super-resolution through the InvSR FastAPI service."""

    load_local_env()
    output_image.parent.mkdir(parents=True, exist_ok=True)
    tool_args = getattr(tool_call, "args", {}) or {}
    api_url = (
        os.environ.get("MM_REACT_SUPER_RESOLUTION_API_URL")
        or os.environ.get("SUPER_RESOLUTION_API_URL")
        or DEFAULT_API_URL
    )
    timeout_seconds = int(
        os.environ.get(
            "MM_REACT_TOOL_TIMEOUT_SECONDS",
            os.environ.get("TOOL_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS),
        )
    )
    payload = json.dumps(
        {
            "in_path": str(input_image.expanduser().resolve()),
            "out_path": str(output_image.expanduser().resolve()),
            "bs": int(tool_args.get("bs", 1)),
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
        lock_start = time.monotonic()
        with _SUPER_RESOLUTION_LOCK:
            wait_seconds = time.monotonic() - lock_start
            if wait_seconds >= 1:
                print(
                    "[super_resolution] waited "
                    f"{wait_seconds:.1f}s for serialized API access",
                    flush=True,
                )
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            "超分辨率 API 请求失败。请检查输入参数和服务日志。"
            f"HTTP {exc.code}: {detail}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            "超分辨率 API 调用失败。请确认服务已启动，并监听 "
            f"{api_url}。"
        ) from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError("超分辨率 API 返回的内容不是合法 JSON。") from exc

    if not isinstance(result, dict):
        raise RuntimeError(f"超分辨率 API 返回格式无效: {result!r}")

    if result.get("status") != "success":
        message = result.get("message") or result
        raise RuntimeError(f"超分辨率 API 处理失败: {message}")

    if not output_image.exists():
        api_output = result.get("output", str(output_image))
        raise RuntimeError(f"超分辨率 API 未生成输出图片: {api_output}")

    return "Image super-resolution upscaling completed."
