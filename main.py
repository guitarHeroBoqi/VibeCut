"""
VibeCut - AI 全自动视频制片 Agent
入口文件：负责解析用户指令，启动 LangGraph 状态机，并输出最终产物路径。
"""

import argparse
import asyncio
import logging
from datetime import datetime
from pathlib import Path

from graph import build_graph, VideoState

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("vibecut")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VibeCut: AI 全自动视频制片 Agent"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="视频主题或创作提示词，例如：'制作一段关于太空探索的60秒宣传片'",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/final.mp4",
        help="最终输出视频路径（默认：output/final.mp4）",
    )
    parser.add_argument(
        "--assets-dir",
        type=str,
        default="assets",
        help="素材目录（默认：assets/）",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="qwen3.6-plus",
        help="驱动 Agent 的 LLM 模型（默认：qwen3.6-plus）",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="节点失败时最大重试次数（默认：3）",
    )
    return parser.parse_args()


async def run(args: argparse.Namespace) -> None:
    # 每次运行创建独立的带时间戳子目录，避免不同运行的素材混在一起
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_assets_dir = str(Path(args.assets_dir) / run_id)
    Path(run_assets_dir).mkdir(parents=True, exist_ok=True)
    logger.info("本次运行素材目录：%s", run_assets_dir)

    # 输出文件名加上时间戳，避免覆盖历史成片
    _out = Path(args.output)
    output_path = str(_out.parent / f"{_out.stem}_{run_id}{_out.suffix}")
    _out.parent.mkdir(parents=True, exist_ok=True)
    logger.info("本次运行输出路径：%s", output_path)

    initial_state: VideoState = {
        "prompt": args.prompt,
        "output_path": output_path,
        "assets_dir": run_assets_dir,
        "model": args.model,
        "max_retries": args.max_retries,
        # 以下字段由各节点在运行时填充
        "plan": None,
        "raw_clips": [],
        "edit_script": None,
        "final_video_path": None,
        "error": None,
        "retry_count": 0,
    }

    logger.info("启动 VibeCut Agent | prompt=%r", args.prompt)
    graph = build_graph()
    final_state = await graph.ainvoke(initial_state)

    if final_state.get("error"):
        logger.error("Agent 执行失败：%s", final_state["error"])
        raise SystemExit(1)

    logger.info("视频制作完成！输出路径：%s", final_state["final_video_path"])


def main() -> None:
    args = parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
