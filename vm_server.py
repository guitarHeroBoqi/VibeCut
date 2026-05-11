"""
VibeCut - 本地 Mock VM 服务器

模拟真实 VM 的 HTTP API，让整个 Agent 管线可以在本机完整运行，无需真实虚拟机。

暴露三个接口：
  POST /upload   - 接收素材文件，存入工作目录
  POST /exec     - 在本机执行指定命令（editor_script.py），捕获输出
  GET  /download - 将工作目录内的文件返回给调用方

启动方式：
  python vm_server.py              # 默认监听 localhost:8765
  python vm_server.py --port 9000  # 自定义端口
"""

import argparse
import asyncio
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

from aiohttp import web

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] vm_server: %(message)s",
)
logger = logging.getLogger("vm_server")

WORKSPACE = Path(os.getenv("VIBECUT_VM_WORKSPACE", "/tmp/vibecut"))


# ---------------------------------------------------------------------------
# 路由处理
# ---------------------------------------------------------------------------

async def handle_upload(request: web.Request) -> web.Response:
    """
    POST /upload
    表单字段：
      file     - 文件二进制内容（multipart）
      dest_dir - 目标目录（字符串），不传则默认写到 WORKSPACE
    """
    reader = await request.multipart()

    dest_dir = WORKSPACE
    saved_files: list[str] = []

    async for part in reader:
        if part.name == "dest_dir":
            dest_dir_str = await part.read(decode=True)
            dest_dir = Path(dest_dir_str.decode())
        elif part.name == "file":
            dest_dir.mkdir(parents=True, exist_ok=True)
            filename = part.filename or "upload"
            save_path = dest_dir / filename
            with open(save_path, "wb") as f:
                while True:
                    chunk = await part.read_chunk(65536)
                    if not chunk:
                        break
                    f.write(chunk)
            saved_files.append(str(save_path))
            logger.info("[upload] 已保存：%s", save_path)

    return web.json_response({"saved": saved_files, "status": "ok"})


async def handle_exec(request: web.Request) -> web.Response:
    """
    POST /exec
    请求体（JSON）：
      command     - 可执行文件，例如 "python3"
      args        - 参数列表，例如 ["/tmp/vibecut/editor_script.py", "--config-json", "..."]
      working_dir - 工作目录（字符串）
      timeout     - 超时秒数（整数）

    返回（JSON）：
      exit_code   - 进程退出码
      stdout      - 标准输出
      stderr      - 标准错误
      output_path - 从 stdout 解析出的视频路径（若存在）
    """
    body = await request.json()

    command: str = body.get("command", "python3")
    args: list[str] = body.get("args", [])
    working_dir: str = body.get("working_dir", str(WORKSPACE))
    timeout: int = int(body.get("timeout", 600))

    cmd = [command] + args
    logger.info("[exec] 执行命令：%s", " ".join(cmd))

    Path(working_dir).mkdir(parents=True, exist_ok=True)

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=working_dir,
        )
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.communicate()
            logger.error("[exec] 命令超时（%ds）", timeout)
            return web.json_response(
                {"exit_code": -1, "stdout": "", "stderr": f"超时（{timeout}s）", "output_path": None},
                status=200,
            )
    except FileNotFoundError as e:
        logger.error("[exec] 命令不存在：%s", e)
        return web.json_response(
            {"exit_code": -1, "stdout": "", "stderr": str(e), "output_path": None},
            status=200,
        )

    stdout = stdout_bytes.decode(errors="replace")
    stderr = stderr_bytes.decode(errors="replace")
    exit_code = proc.returncode

    logger.info("[exec] 命令完成，exit_code=%d", exit_code)
    if stderr:
        logger.debug("[exec] stderr: %s", stderr[:500])

    # editor_script.py 会在最后一行打印 {"output_path": "...", "exit_code": 0}
    output_path = None
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                result_json = json.loads(line)
                output_path = result_json.get("output_path")
                break
            except json.JSONDecodeError:
                continue

    return web.json_response({
        "exit_code": exit_code,
        "stdout": stdout,
        "stderr": stderr,
        "output_path": output_path,
    })


async def handle_download(request: web.Request) -> web.Response:
    """
    GET /download?path=<VM 内的文件路径>
    将文件内容以二进制流返回。
    """
    file_path_str = request.rel_url.query.get("path", "")
    if not file_path_str:
        return web.Response(status=400, text="缺少 path 参数")

    file_path = Path(file_path_str)
    if not file_path.exists():
        logger.error("[download] 文件不存在：%s", file_path)
        return web.Response(status=404, text=f"文件不存在：{file_path}")

    logger.info("[download] 返回文件：%s（%.1f MB）", file_path, file_path.stat().st_size / 1024 / 1024)
    return web.FileResponse(file_path)


# ---------------------------------------------------------------------------
# 应用组装
# ---------------------------------------------------------------------------

def build_app() -> web.Application:
    app = web.Application(client_max_size=2 * 1024 ** 3)  # 最大 2 GB 上传
    app.router.add_post("/upload", handle_upload)
    app.router.add_post("/exec", handle_exec)
    app.router.add_get("/download", handle_download)
    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VibeCut 本地 Mock VM 服务器")
    parser.add_argument("--host", default="localhost", help="监听地址（默认：localhost）")
    parser.add_argument("--port", type=int, default=8765, help="监听端口（默认：8765）")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    logger.info("VM Mock 服务器启动：http://%s:%d", args.host, args.port)
    logger.info("工作目录：%s", WORKSPACE)
    app = build_app()
    web.run_app(app, host=args.host, port=args.port, access_log=logger)


if __name__ == "__main__":
    main()
