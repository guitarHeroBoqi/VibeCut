"""
VibeCut - LangGraph 状态机定义
节点流转：planner → generator → editor → (成功) END
                                        ↑          |
                                        └──(重试)──┘
"""

import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Any, Optional
from typing_extensions import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

from tools import (
    generate_clip,
    run_editor_in_vm,
    upload_asset,
    download_result,
    VM_WORKSPACE,
)

logger = logging.getLogger("vibecut.graph")


def _extract_json(text: str) -> str:
    """清洗 LLM 返回内容：去除 <think>...</think> 思考链和 markdown 代码块包裹。"""
    # 去掉 qwen3 等模型的思考链
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # 提取 ```json ... ``` 或 ``` ... ``` 内的内容
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if match:
        text = match.group(1)
    return text.strip()

# ---------------------------------------------------------------------------
# 状态定义
# ---------------------------------------------------------------------------

class VideoState(TypedDict):
    prompt: str                          # 用户输入的创作提示词
    output_path: str                     # 期望的最终输出路径
    assets_dir: str                      # 素材根目录
    model: str                           # LLM 模型名称
    max_retries: int                     # 最大重试次数
    plan: Optional[dict[str, Any]]       # planner 生成的拍摄/剪辑计划
    raw_clips: list[str]                 # generator 生成的原始素材路径列表
    edit_script: Optional[dict[str, Any]]# 传给 editor_script.py 的剪辑指令
    final_video_path: Optional[str]      # 最终视频绝对路径
    error: Optional[str]                 # 最新一次错误信息
    retry_count: int                     # 当前重试次数


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

async def _ask_user(prompt_str: str) -> str:
    """在 async 上下文中安全地等待终端输入，不阻塞事件循环。"""
    return await asyncio.to_thread(input, prompt_str)


_AUDIO_EXTS = {".mp3", ".wav", ".aac", ".m4a", ".ogg"}


def _list_bgm_files(bgm_dir: str) -> list[Path]:
    """返回 bgm_dir 下所有音频文件，按文件名排序。"""
    base = Path(bgm_dir)
    if not base.exists():
        return []
    return sorted(p for p in base.iterdir() if p.is_file() and p.suffix.lower() in _AUDIO_EXTS)


async def recommend_bgm(
    bgm_dir: str,
    plan: dict[str, Any],
    model: str,
) -> Optional[str]:
    """
    用大模型根据视频风格和 BGM 文件名推荐最合适的一首。
    返回推荐文件的绝对路径字符串，或 None（无法确定时）。
    """
    files = _list_bgm_files(bgm_dir)
    if not files:
        return None

    file_list = "\n".join(p.name for p in files)
    bgm_style = plan.get("bgm_style", "")
    title = plan.get("title", "")

    llm = ChatOpenAI(model=model, temperature=0)
    messages = [
        SystemMessage(content=(
            "你是音乐选曲助手。根据视频主题和风格，从 BGM 文件列表中推荐最合适的一首。\n"
            "文件名能反映音乐风格和内容。只有在高度吻合时才推荐，不确定时返回 null。\n"
            "只返回 JSON，格式：{\"match\": \"文件名\"} 或 {\"match\": null}，不要其他内容。"
        )),
        HumanMessage(content=(
            f"视频标题：{title}\n"
            f"期望音乐风格：{bgm_style}\n\n"
            f"可用 BGM 文件：\n{file_list}"
        )),
    ]

    try:
        response = await llm.ainvoke(messages)
        result = json.loads(_extract_json(response.content))
        matched_name = result.get("match")
        if matched_name:
            for p in files:
                if p.name == matched_name:
                    return str(p)
    except Exception as e:
        logger.warning("[recommend_bgm] 推荐失败，跳过：%s", e)

    return None


async def _manual_bgm_select(bgm_dir: str) -> Optional[str]:
    """
    展示 BGM 目录下所有文件的编号列表，等待用户选择。
    返回选中文件的绝对路径字符串，或 None（目录为空时）。
    """
    files = _list_bgm_files(bgm_dir)
    if not files:
        print("   BGM 目录为空，将跳过背景音乐。")
        return None

    print("\n   可用 BGM 列表：")
    for i, p in enumerate(files, 1):
        print(f"   {i}. {p.name}")

    while True:
        raw = (await _ask_user(f"   请输入编号 [1-{len(files)}]：")).strip()
        if raw.isdigit() and 1 <= int(raw) <= len(files):
            chosen = files[int(raw) - 1]
            print(f"   ✓ 已选择：{chosen.name}")
            return str(chosen)
        print(f"   请输入 1 到 {len(files)} 之间的数字。")


async def find_reusable_clip(
    scene_desc: str,
    assets_base_dir: str,
    model: str,
) -> Optional[str]:
    """
    遍历素材库所有 .mp4 文件名，用大模型判断是否有合适的片段可复用。
    只有在文件名与场景描述高度吻合时才推荐复用。
    返回匹配的文件路径字符串，或 None。
    """
    base = Path(assets_base_dir)
    if not base.exists():
        return None

    all_clips = sorted(base.rglob("*.mp4"))
    if not all_clips:
        return None

    file_list = "\n".join(str(p) for p in all_clips)
    llm = ChatOpenAI(model=model, temperature=0)
    messages = [
        SystemMessage(content=(
            "你是视频素材管理助手。根据场景描述，从素材文件列表中找出最合适复用的视频片段。\n"
            "文件名反映了视频内容（由英文提示词生成）。只有在高度吻合时才推荐复用，不确定时返回 null。\n"
            "只返回 JSON，格式：{\"match\": \"文件路径\"} 或 {\"match\": null}，不要其他内容。"
        )),
        HumanMessage(content=f"场景描述：{scene_desc}\n\n可用素材文件：\n{file_list}"),
    ]

    try:
        response = await llm.ainvoke(messages)
        result = json.loads(_extract_json(response.content))
        matched = result.get("match")
        if matched and Path(matched).exists():
            return matched
    except Exception as e:
        logger.warning("[find_reusable_clip] 检索失败，跳过复用：%s", e)

    return None


# ---------------------------------------------------------------------------
# 节点实现
# ---------------------------------------------------------------------------

async def planner_node(state: VideoState) -> dict[str, Any]:
    """
    Planner 节点：接收用户 prompt，使用 LLM 生成结构化的视频制作计划。
    输出：plan（场景列表、字幕、背景音乐风格、时长分配等）
    """
    logger.info("[planner] 开始生成制作计划 | prompt=%r", state["prompt"])

    llm = ChatOpenAI(model=state["model"], temperature=0.7)

    system_prompt = """你是一位专业的视频导演兼剪辑师 AI。
根据用户的视频主题，输出一份 JSON 格式的制作计划，包含以下字段：
{
  "title": "视频标题",
  "duration_seconds": 60,
  "scenes": [
    {
      "id": 1,
      "description": "场景描述",
      "duration": 10,
      "visual_prompt": "用于 AI 生成图/视频的提示词（英文）",
      "caption": "字幕文本",
      "transition": "淡入淡出 | 硬切 | 滑动"
    }
  ],
  "bgm_style": "epic cinematic",
  "color_grade": "warm | cool | neutral"
}
只返回 JSON，不要有其他内容。"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=state["prompt"]),
    ]

    response = await llm.ainvoke(messages)
    cleaned = _extract_json(response.content)

    try:
        plan = json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.warning("[planner] LLM 返回内容无法解析为 JSON，原始内容：%r，错误：%s", cleaned, e)
        plan = {"raw": response.content}

    logger.info("[planner] 计划生成完毕，共 %d 个场景", len(plan.get("scenes", [])))
    return {"plan": plan, "error": None}


async def _confirm_plan(plan: dict[str, Any], model: str) -> dict[str, Any]:
    """
    向用户展示制作计划摘要，允许用户确认或提出修改意见（最多重规划 3 次）。
    返回最终确认的 plan 字典。
    """
    llm = ChatOpenAI(model=model, temperature=0.7)

    for attempt in range(3):
        scenes = plan.get("scenes", [])
        print(f"\n{'═' * 60}")
        print(f"📋 制作计划确认（第 {attempt + 1} 次）")
        print(f"{'═' * 60}")
        print(f"  标题：{plan.get('title', '未命名')}")
        print(f"  总时长：{plan.get('duration_seconds', '?')}s  |  "
              f"音乐风格：{plan.get('bgm_style', '?')}  |  "
              f"调色：{plan.get('color_grade', '?')}")
        print(f"\n  场景列表（共 {len(scenes)} 个）：")
        for scene in scenes:
            print(f"  {'─' * 56}")
            print(f"  场景 {scene.get('id', '?')} [{scene.get('duration', '?')}s]  {scene.get('description', '')}")
            print(f"    提示词：{scene.get('visual_prompt', '')}")
            if scene.get("caption"):
                print(f"    字幕：{scene.get('caption')}")
        print(f"{'═' * 60}")

        feedback = (await _ask_user(
            "  直接回车确认开始生成，或输入修改意见（例如：减少场景 / 改为白天氛围）：\n  > "
        )).strip()

        if not feedback:
            print("  ✓ 计划已确认，开始生成素材。\n")
            return plan

        print("\n  🔄 正在根据您的意见重新规划...")
        try:
            response = await llm.ainvoke([
                SystemMessage(content=(
                    "你是专业视频导演 AI。根据用户反馈修改现有制作计划，返回完整修改后的 JSON，"
                    "格式与原计划完全相同，只返回 JSON。"
                )),
                HumanMessage(content=(
                    f"原计划：\n{json.dumps(plan, ensure_ascii=False)}\n\n用户反馈：{feedback}"
                )),
            ])
            plan = json.loads(_extract_json(response.content))
            logger.info("[planner] 用户修改后重新规划完毕，共 %d 个场景", len(plan.get("scenes", [])))
        except Exception as e:
            logger.warning("[planner] 重新规划失败，沿用原计划：%s", e)
            print(f"  ⚠️  重新规划失败（{e}），沿用当前计划。")
            return plan

    print("  ⚠️  已达最大修改次数（3次），使用最新计划继续。\n")
    return plan


async def generator_node(state: VideoState) -> dict[str, Any]:
    """
    Generator 节点：
      0. 展示制作计划让用户确认或修改
      1. 对每个场景用大模型检索素材库，询问是否复用
      2. 无复用时可重新规划场景描述，再次搜索（循环直到找到或决定生成）
      3. 确认提示词后调用 AI 生成视频
    输出：raw_clips（本地素材路径列表）
    """
    # ── Step 0: 展示计划，让用户确认或修改 ───────────────────────────
    plan = await _confirm_plan(state["plan"], state["model"])
    scenes = plan.get("scenes", [])
    logger.info("[generator] 开始处理 %d 个场景", len(scenes))

    # 素材库根目录（绝对路径，避免子进程中相对路径失效）
    assets_base_dir = str(Path(state["assets_dir"]).parent.resolve())

    clips: list[str] = []

    for scene in scenes:
        scene_id = scene.get("id", "?")
        visual_prompt = scene.get("visual_prompt", scene.get("description", ""))
        scene_desc = scene.get("description", visual_prompt)
        duration = scene.get("duration", 5)

        print(f"\n{'─' * 60}")
        print(f"场景 {scene_id}：{scene_desc}")
        print(f"{'─' * 60}")

        # ── Step 1+2: 检索素材库循环（可重新规划场景描述后再搜）────────
        clip_reused = False
        while True:
            print("🔍 正在检索素材库...")
            reusable = await find_reusable_clip(scene_desc, assets_base_dir, state["model"])

            if reusable:
                print(f"\n💡 发现可复用素材：{Path(reusable).name}")
                choice = (await _ask_user("   是否复用该片段？[Y/n] ")).strip().lower()
                if choice in ("", "y", "yes"):
                    clips.append(reusable)
                    clip_reused = True
                    print("   ✓ 已标记复用，跳过生成。")
                    break
                print("   跳过复用，继续操作。")
            else:
                print("   未找到可复用素材。")

            print(f"\n  请选择操作：")
            print(f"  1. 用当前提示词生成新片段（消耗 API 额度）")
            print(f"  2. 重新描述该场景，再次搜索素材库（省钱）")
            action = (await _ask_user("  请输入 1 或 2 [默认 1]：")).strip()

            if action == "2":
                new_desc = (await _ask_user(
                    "  请输入新的场景描述（中文即可，系统会自动生成英文提示词）：\n  > "
                )).strip()
                if not new_desc:
                    print("  输入为空，沿用原提示词生成。")
                    break
                scene_desc = new_desc
                print("  🔄 正在生成对应的英文提示词...")
                try:
                    _llm = ChatOpenAI(model=state["model"], temperature=0.3)
                    resp = await _llm.ainvoke([
                        SystemMessage(content=(
                            "你是视频提示词专家。将中文场景描述转换为适合 AI 视频生成的英文提示词，"
                            "风格写实、电影感强，只返回英文提示词，不要其他内容。"
                        )),
                        HumanMessage(content=scene_desc),
                    ])
                    visual_prompt = _extract_json(resp.content) or resp.content.strip()
                    print(f"  ✓ 新提示词：{visual_prompt}")
                except Exception as e:
                    logger.warning("[generator] 提示词翻译失败，使用原描述：%s", e)
                    visual_prompt = scene_desc
                continue  # 再次搜索素材库
            else:
                break  # 选择生成新片段

        if clip_reused:
            continue  # 该场景已通过复用处理，进入下一个场景

        # ── Step 3: 用户确认或修改提示词后生成视频 ──────────────────────
        print(f"\n📝 将使用以下提示词生成视频：")
        print(f"   {visual_prompt}")
        user_input = (await _ask_user("   直接回车确认，或输入修改后的提示词：")).strip()
        if user_input:
            visual_prompt = user_input
            print(f"   ✓ 已更新提示词：{visual_prompt}")

        print(f"\n⏳ 正在生成场景 {scene_id} 的视频片段...")
        try:
            clip_path = await generate_clip(
                prompt=visual_prompt,
                duration=duration,
                output_dir=state["assets_dir"],
                scene_id=scene_id,
            )
            clips.append(clip_path)
            print(f"   ✓ 已保存：{clip_path}")
            logger.info("[generator] 场景 %s 素材已生成：%s", scene_id, clip_path)
        except Exception as e:
            logger.error("[generator] 场景 %s 生成失败：%s", scene_id, e)
            return {"raw_clips": clips, "error": f"场景 {scene_id} 素材生成失败：{e}"}

    print(f"\n{'─' * 60}")
    print(f"✅ 全部 {len(scenes)} 个场景处理完毕，共 {len(clips)} 个片段。")
    print(f"{'─' * 60}\n")
    return {"plan": plan, "raw_clips": clips, "error": None}


async def editor_node(state: VideoState) -> dict[str, Any]:
    """
    Editor 节点：将原始素材和制作计划转化为剪辑指令，
    通过 MCP 工具在 VM 内执行 editor_script.py，输出最终视频。
    """
    logger.info("[editor] 开始构建剪辑指令脚本")

    # ── 询问用户音频处理方式 ──────────────────────────────────────────
    print(f"\n{'─' * 60}")
    print("🎵 请选择音频处理方式：")
    print("  1. 保留各片段原有声音（默认）")
    print("  2. 使用本地 BGM（静音片段，叠加 assets/bgm/ 中的音乐）")
    print(f"{'─' * 60}")
    audio_choice = (await _ask_user("请输入 1 或 2 [默认 1]：")).strip()

    # assets/bgm/ 用绝对路径，editor_script 在 /tmp/vibecut 下执行，相对路径会失效
    assets_base_dir = Path(state["assets_dir"]).parent.resolve()
    bgm_dir = str(assets_base_dir / "bgm")

    bgm_file: Optional[str] = None
    if audio_choice == "2":
        audio_mode = "local_bgm"
        bgm_files = _list_bgm_files(bgm_dir)
        if not bgm_files:
            print("   ⚠️  assets/bgm/ 目录为空，已降级为保留原声。")
            audio_mode = "keep_original"
        elif len(bgm_files) == 1:
            bgm_file = str(bgm_files[0])
            print(f"   ✓ 目录中仅有一首 BGM，直接使用：{bgm_files[0].name}")
        else:
            # 多个文件：先用大模型推荐
            print("   🤖 正在用大模型推荐最合适的 BGM...")
            recommended = await recommend_bgm(bgm_dir, state["plan"], state["model"])
            if recommended:
                print(f"   💿 大模型推荐：{Path(recommended).name}")
                confirm = (await _ask_user("   是否使用？[Y/n] ")).strip().lower()
                if confirm in ("", "y", "yes"):
                    bgm_file = recommended
                    print(f"   ✓ 已选择：{Path(recommended).name}")
                else:
                    bgm_file = await _manual_bgm_select(bgm_dir)
            else:
                print("   大模型无法确定推荐，请手动选择：")
                bgm_file = await _manual_bgm_select(bgm_dir)

            if bgm_file is None:
                audio_mode = "keep_original"
    else:
        audio_mode = "keep_original"
        print("   ✓ 将保留各片段原有声音")

    vm_clips = [f"{VM_WORKSPACE}/{Path(p).name}" for p in state["raw_clips"]]
    edit_script = {
        "clips": vm_clips,
        "scenes": state["plan"].get("scenes", []),
        "output_path": f"{VM_WORKSPACE}/final.mp4",
        "bgm_style": state["plan"].get("bgm_style", "neutral"),
        "color_grade": state["plan"].get("color_grade", "neutral"),
        "audio_mode": audio_mode,
        "bgm_dir": bgm_dir,
        "bgm_file": bgm_file or "",
    }

    try:
        logger.info("[editor] 上传素材到 VM")
        await upload_asset(local_paths=state["raw_clips"])

        logger.info("[editor] 在 VM 内执行剪辑脚本")
        vm_output_path = await run_editor_in_vm(edit_script=edit_script)

        logger.info("[editor] 下载最终视频：%s", vm_output_path)
        local_final_path = await download_result(
            vm_path=vm_output_path,
            local_path=state["output_path"],
        )

        logger.info("[editor] 剪辑完成！最终视频：%s", local_final_path)
        return {
            "edit_script": edit_script,
            "final_video_path": local_final_path,
            "error": None,
        }
    except Exception as e:
        logger.error("[editor] 剪辑失败：%s", e)
        return {"edit_script": edit_script, "error": f"剪辑执行失败：{e}"}


# ---------------------------------------------------------------------------
# 路由逻辑
# ---------------------------------------------------------------------------

def should_retry(state: VideoState) -> str:
    """若出现错误且重试次数未超限，则回到 planner 重试；否则终止。"""
    if state.get("error") and state["retry_count"] < state["max_retries"]:
        logger.warning(
            "[router] 检测到错误，准备第 %d 次重试：%s",
            state["retry_count"] + 1,
            state["error"],
        )
        return "retry"
    elif state.get("error"):
        logger.error("[router] 已达最大重试次数，终止流程。")
        return "fail"
    return "success"


def increment_retry(state: VideoState) -> dict[str, Any]:
    """重试前递增计数器并清空错误，让 planner 重新开始。"""
    return {"retry_count": state["retry_count"] + 1, "error": None}


# ---------------------------------------------------------------------------
# 图构建
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    graph = StateGraph(VideoState)

    graph.add_node("planner", planner_node)
    graph.add_node("generator", generator_node)
    graph.add_node("editor", editor_node)
    graph.add_node("retry_reset", increment_retry)

    graph.set_entry_point("planner")

    graph.add_edge("planner", "generator")
    graph.add_edge("generator", "editor")

    graph.add_conditional_edges(
        "editor",
        should_retry,
        {
            "success": END,
            "fail": END,
            "retry": "retry_reset",
        },
    )

    graph.add_edge("retry_reset", "planner")

    return graph.compile()
