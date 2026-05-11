"""
VibeCut - MCP 工具层
封装所有与 VM（虚拟机/沙箱）交互的操作，以及 AI 生成素材的调用。
每个函数都可以作为 LangGraph 节点内的原子操作调用。
"""

import asyncio
import json
import logging
import os
import re
import subprocess
import uuid
from pathlib import Path
from typing import Any, Optional

import aiofiles
import aiohttp

logger = logging.getLogger("vibecut.tools")

# ---------------------------------------------------------------------------
# 配置（从环境变量读取，实际部署时替换为 Secret Manager）
# ---------------------------------------------------------------------------

VM_HOST = os.getenv("VIBECUT_VM_HOST", "localhost")
VM_PORT = int(os.getenv("VIBECUT_VM_PORT", "8765"))
VM_API_BASE = f"http://{VM_HOST}:{VM_PORT}"

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

DASHSCOPE_VIDEO_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/video-generation/video-synthesis"
DASHSCOPE_TASK_URL = "https://dashscope.aliyuncs.com/api/v1/tasks/{task_id}"

VM_WORKSPACE = os.getenv("VIBECUT_VM_WORKSPACE", "/tmp/vibecut")
VM_PYTHON = os.getenv("VIBECUT_VM_PYTHON", "python3")
EDITOR_SCRIPT_VM_PATH = os.getenv(
    "VIBECUT_EDITOR_SCRIPT_PATH",
    f"{VM_WORKSPACE}/editor_script.py",
)

# ---------------------------------------------------------------------------
# 素材生成工具
# ---------------------------------------------------------------------------

def _prompt_to_slug(prompt: str, max_len: int = 45) -> str:
    """把提示词转换成适合文件名的短字符串（字母数字+下划线）。"""
    slug = re.sub(r"[^\w\s]", "", prompt.lower())
    slug = re.sub(r"\s+", "_", slug.strip())
    slug = slug[:max_len].strip("_")
    return slug or "clip"


async def generate_clip(
    prompt: str,
    duration: int,
    output_dir: str,
    scene_id: Any,
) -> str:
    """
    调用阿里百炼 happyhorse-1.0-t2v 生成单个场景的原始视频片段。

    Args:
        prompt:     用于生成视频的提示词（支持中文）
        duration:   目标时长（秒），百炼目前固定支持 5s
        output_dir: 本地保存目录
        scene_id:   场景编号，用于文件命名

    Returns:
        生成的视频文件本地路径
    """
    logger.info("[tools.generate_clip] scene=%s | duration=%ds | prompt=%r", scene_id, duration, prompt)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    slug = _prompt_to_slug(prompt)
    candidate = Path(output_dir) / f"{slug}.mp4"
    # 若同名文件已存在（同一次运行重复生成），加短后缀避免覆盖
    if candidate.exists():
        candidate = Path(output_dir) / f"{slug}_{uuid.uuid4().hex[:6]}.mp4"
    output_path = str(candidate)

    if not DASHSCOPE_API_KEY:
        logger.warning("[tools.generate_clip] DASHSCOPE_API_KEY 未配置，使用 MOCK 素材")
        _create_mock_clip(output_path, duration)
        return output_path

    headers = {
        "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
        "Content-Type": "application/json",
        "X-DashScope-Async": "enable",
    }
    payload = {
        "model": "happyhorse-1.0-t2v",
        "input": {
            "prompt": prompt,
        },
        "parameters": {
            "resolution": "720P",
            "ratio": "16:9",
            "duration": 5,
        },
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(
            DASHSCOPE_VIDEO_URL,
            json=payload,
            headers=headers,
        ) as resp:
            if resp.status != 200:
                body = await resp.text()
                raise RuntimeError(f"百炼 API 错误 {resp.status}: {body}")
            data = await resp.json()

        task_id = data["output"]["task_id"]
        logger.info("[tools.generate_clip] 百炼任务已提交，task_id=%s", task_id)

        video_url = await _poll_dashscope_task(session, task_id, headers)
        await _download_file(session, video_url, output_path)

    logger.info("[tools.generate_clip] 素材已保存：%s", output_path)
    return output_path


async def _poll_dashscope_task(
    session: aiohttp.ClientSession,
    task_id: str,
    headers: dict,
    poll_interval: int = 5,
    timeout: int = 600,
) -> str:
    """轮询百炼异步任务直到完成，返回视频 URL。"""
    elapsed = 0
    poll_headers = {k: v for k, v in headers.items() if k != "X-DashScope-Async"}
    while elapsed < timeout:
        await asyncio.sleep(poll_interval)
        elapsed += poll_interval
        async with session.get(
            DASHSCOPE_TASK_URL.format(task_id=task_id),
            headers=poll_headers,
        ) as resp:
            data = await resp.json()
            status = data.get("output", {}).get("task_status")
            logger.debug("[tools._poll_dashscope_task] task=%s status=%s", task_id, status)
            if status == "SUCCEEDED":
                return data["output"]["video_url"]
            elif status in ("FAILED", "CANCELLED"):
                msg = data.get("output", {}).get("message", "未知错误")
                raise RuntimeError(f"百炼任务 {task_id} 以状态 {status} 结束：{msg}")
    raise TimeoutError(f"百炼任务 {task_id} 超时（{timeout}s）")


async def _download_file(
    session: aiohttp.ClientSession, url: str, local_path: str
) -> None:
    async with session.get(url) as resp:
        resp.raise_for_status()
        async with aiofiles.open(local_path, "wb") as f:
            async for chunk in resp.content.iter_chunked(1024 * 64):
                await f.write(chunk)


def _create_mock_clip(output_path: str, duration: int) -> None:
    """在没有 API Key 时用 ffmpeg 生成彩色测试视频作为占位素材。"""
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", f"color=c=blue:s=1280x720:d={duration}",
        "-c:v", "libx264", "-t", str(duration),
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.warning("[tools._create_mock_clip] ffmpeg 不可用，创建空文件代替")
        Path(output_path).touch()


# ---------------------------------------------------------------------------
# VM 操作工具
# ---------------------------------------------------------------------------

async def upload_asset(local_paths: list[str]) -> None:
    """
    将本地素材文件批量上传到 VM 工作目录。
    实际实现可替换为 SCP / 对象存储 / VM HTTP API。
    """
    logger.info("[tools.upload_asset] 上传 %d 个文件到 VM", len(local_paths))

    async with aiohttp.ClientSession() as session:
        for local_path in local_paths:
            path = Path(local_path)
            if not path.exists():
                raise FileNotFoundError(f"素材文件不存在：{local_path}")

            async with aiofiles.open(local_path, "rb") as f:
                data = await f.read()

            form = aiohttp.FormData()
            form.add_field("file", data, filename=path.name, content_type="application/octet-stream")
            form.add_field("dest_dir", VM_WORKSPACE)

            async with session.post(f"{VM_API_BASE}/upload", data=form) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    raise RuntimeError(f"上传失败 [{path.name}]: {body}")
                logger.debug("[tools.upload_asset] 已上传：%s", path.name)


async def run_editor_in_vm(edit_script: dict[str, Any]) -> str:
    """
    将剪辑指令序列化为 JSON，通过 VM API 触发 editor_script.py 执行。

    Args:
        edit_script: 传递给 editor_script.py 的完整剪辑配置字典

    Returns:
        VM 内生成的最终视频路径
    """
    logger.info("[tools.run_editor_in_vm] 触发 VM 剪辑任务")

    script_json = json.dumps(edit_script, ensure_ascii=False)

    async with aiohttp.ClientSession() as session:
        payload = {
            "command": VM_PYTHON,
            "args": [
                EDITOR_SCRIPT_VM_PATH,
                "--config-json", script_json,
            ],
            "working_dir": VM_WORKSPACE,
            "timeout": 600,
        }

        async with session.post(f"{VM_API_BASE}/exec", json=payload) as resp:
            if resp.status != 200:
                body = await resp.text()
                raise RuntimeError(f"VM exec 失败：{body}")
            result = await resp.json()

        if result.get("exit_code", 1) != 0:
            stderr = result.get("stderr", "")
            raise RuntimeError(f"editor_script.py 执行错误：\n{stderr}")

        vm_output_path = result.get("output_path") or f"{VM_WORKSPACE}/final.mp4"
        logger.info("[tools.run_editor_in_vm] 剪辑完成，VM 输出路径：%s", vm_output_path)
        return vm_output_path


async def download_result(vm_path: str, local_path: str) -> str:
    """
    从 VM 下载最终视频到本地。

    Args:
        vm_path:    VM 内的视频路径
        local_path: 期望保存的本地路径

    Returns:
        实际保存的本地绝对路径
    """
    logger.info("[tools.download_result] 从 VM 下载：%s → %s", vm_path, local_path)
    Path(local_path).parent.mkdir(parents=True, exist_ok=True)

    async with aiohttp.ClientSession() as session:
        params = {"path": vm_path}
        async with session.get(f"{VM_API_BASE}/download", params=params) as resp:
            if resp.status != 200:
                body = await resp.text()
                raise RuntimeError(f"下载失败：{body}")
            await _download_file(session, str(resp.url), local_path)

    abs_path = str(Path(local_path).resolve())
    logger.info("[tools.download_result] 下载完成：%s", abs_path)
    return abs_path


# ---------------------------------------------------------------------------
# MCP Tool Schema（供 LangGraph ToolNode / function calling 注册用）
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS = [
    {
        "name": "generate_clip",
        "description": "调用 AI 视频生成服务，根据提示词生成指定时长的视频片段",
        "parameters": {
            "type": "object",
            "properties": {
                "prompt":     {"type": "string",  "description": "英文视频生成提示词"},
                "duration":   {"type": "integer", "description": "时长（秒）"},
                "output_dir": {"type": "string",  "description": "本地保存目录"},
                "scene_id":   {"type": "string",  "description": "场景 ID"},
            },
            "required": ["prompt", "duration", "output_dir", "scene_id"],
        },
        "fn": generate_clip,
    },
    {
        "name": "run_editor_in_vm",
        "description": "在远程 VM 内执行 MoviePy 剪辑脚本，完成视频合成",
        "parameters": {
            "type": "object",
            "properties": {
                "edit_script": {"type": "object", "description": "剪辑配置字典"},
            },
            "required": ["edit_script"],
        },
        "fn": run_editor_in_vm,
    },
]
