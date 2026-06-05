from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from mm_react.env import load_local_env


DEFAULT_API_URL = "http://127.0.0.1:8004/detect"
DEFAULT_TIMEOUT_SECONDS = 900


def run_grounding_dino_tool(
    input_image: Path, tool_call: Any, output_image: Path
) -> dict[str, Any]:
    """Run open-vocabulary detection through a GroundingDINO API service."""

    load_local_env()
    _ = output_image
    tool_args = getattr(tool_call, "args", {}) or {}
    text_prompt = _text_prompt_from_tool_args(tool_args)
    api_url = os.environ.get("MM_REACT_GROUNDING_DINO_API_URL", DEFAULT_API_URL)
    timeout_seconds = int(
        os.environ.get("MM_REACT_TOOL_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS)
    )
    payload = json.dumps(
        {
            "input_path": str(input_image.expanduser().resolve()),
            "text_prompt": text_prompt,
            "output_path": None,
            "box_threshold": float(tool_args.get("box_threshold", 0.3)),
            "text_threshold": float(tool_args.get("text_threshold", 0.25)),
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
            "GroundingDINO 检测 API 请求失败。请检查输入参数和服务日志。"
            f"HTTP {exc.code}: {detail}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            "GroundingDINO 检测 API 调用失败。请确认服务已启动，并监听 "
            f"{api_url}。"
        ) from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError("GroundingDINO 检测 API 返回的内容不是合法 JSON。") from exc

    if not isinstance(result, dict):
        raise RuntimeError(f"GroundingDINO 检测 API 返回格式无效: {result!r}")

    if result.get("status") != "success":
        message = result.get("message") or result.get("detail") or result
        raise RuntimeError(f"GroundingDINO 检测 API 处理失败: {message}")

    return {
        "image_size": result.get("image_size"),
        "detections_count": result.get("detections_count", 0),
        "predictions": result.get("predictions", []),
    }


def _text_prompt_from_tool_args(tool_args: dict[str, Any]) -> str:
    for key in ("text_prompt", "prompt", "classes", "class_names"):
        raw_value = tool_args.get(key, "")
        if isinstance(raw_value, (list, tuple)):
            value = " . ".join(
                str(item).strip() for item in raw_value if str(item).strip()
            )
        else:
            value = str(raw_value).strip()
        if value:
            return value
    return "object"
