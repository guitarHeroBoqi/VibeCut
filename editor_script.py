"""
VibeCut - VM 内部剪辑脚本
在 VM 环境中由 tools.run_editor_in_vm() 调用执行。
接收 JSON 配置，使用 MoviePy 完成视频合成、转场、字幕、调色和背景音乐的叠加。

用法：
    python editor_script.py --config-json '<JSON 字符串>'
    python editor_script.py --config-file /path/to/config.json
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

try:
    # moviepy >= 2.0
    from moviepy import (
        AudioFileClip,
        ColorClip,
        CompositeAudioClip,
        CompositeVideoClip,
        TextClip,
        VideoFileClip,
        concatenate_audioclips,
        concatenate_videoclips,
        vfx,
    )
    from moviepy.audio.fx import AudioFadeIn as audio_fadein, AudioFadeOut as audio_fadeout
    _MOVIEPY_V2 = True
except ImportError:
    # moviepy 1.x
    from moviepy.editor import (
        AudioFileClip,
        ColorClip,
        CompositeAudioClip,
        CompositeVideoClip,
        TextClip,
        VideoFileClip,
        concatenate_audioclips,
        concatenate_videoclips,
        vfx,
    )
    from moviepy.audio.fx.all import audio_fadein, audio_fadeout
    _MOVIEPY_V2 = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("vibecut.editor")

# 默认输出分辨率与帧率
OUTPUT_SIZE = (1280, 720)
OUTPUT_FPS = 30


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

import math as _math


# ---------------------------------------------------------------------------
# moviepy 版本兼容辅助函数
# ---------------------------------------------------------------------------

def _subclip(clip, start, end):
    if _MOVIEPY_V2:
        return clip.subclipped(start, end)
    return clip.subclip(start, end)


def _resize(clip, size):
    if _MOVIEPY_V2:
        return clip.resized(size)
    return clip.resize(size)


def _loop_to_duration(clip, duration):
    """将片段循环填充到指定时长（兼容视频和音频片段）。"""
    if _MOVIEPY_V2:
        n = _math.ceil(duration / clip.duration)
        if isinstance(clip, AudioFileClip):
            looped = concatenate_audioclips([clip] * n)
        else:
            looped = concatenate_videoclips([clip] * n)
        return looped.subclipped(0, duration)
    return clip.loop(duration=duration)


def _multiply_volume(clip, factor):
    if _MOVIEPY_V2:
        return clip.multiply_volume(factor)
    return clip.volumex(factor)


def _set_audio(clip, audio):
    if _MOVIEPY_V2:
        return clip.with_audio(audio)
    return clip.set_audio(audio)


def apply_color_grade(clip: VideoFileClip, style: str) -> VideoFileClip:
    """
    简单调色处理。
    warm  → 提升红/黄通道
    cool  → 提升蓝/青通道
    neutral → 不做处理

    ColorClip 等纯色对象不支持逐帧像素处理，直接跳过。
    """
    if style == "neutral":
        return clip

    color_map = {
        "warm":   [1.1,  1.0,  0.85],
        "cool":   [0.85, 1.0,  1.15],
        "cinema": [1.045, 0.99, 1.155],
    }
    factors = color_map.get(style)
    if factors is None:
        return clip

    fn = lambda frame: (frame * factors).clip(0, 255).astype("uint8")

    if _MOVIEPY_V2:
        if not hasattr(clip, "image_transform"):
            return clip
        return clip.image_transform(fn)
    else:
        if not hasattr(clip, "fl_image"):
            return clip
        return clip.fl_image(fn)


def build_transition(
    clip_a: VideoFileClip,
    clip_b: VideoFileClip,
    transition_type: str,
    duration: float = 0.5,
) -> tuple:
    """
    在两个片段之间添加转场效果，返回 (修改后的clip_a, 修改后的clip_b)。
    支持：淡入淡出（crossfade）、硬切（cut）
    """
    t = transition_type.lower()
    if "淡" in t or "crossfade" in t or "fade" in t:
        if _MOVIEPY_V2:
            try:
                # CrossFadeOut/In 是 alpha 通道效果，顺序拼接时无效；
                # FadeOut/FadeIn 直接修改像素亮度，才能实现可见的淡入淡出
                clip_a = clip_a.with_effects([vfx.FadeOut(duration)])
                clip_b = clip_b.with_effects([vfx.FadeIn(duration)])
            except Exception as e:
                logger.warning("[editor] 转场效果应用失败，降级为硬切：%s", e)
        else:
            clip_a = clip_a.fx(vfx.fadeout, duration)
            clip_b = clip_b.fx(vfx.fadein, duration)
    return clip_a, clip_b


def _find_caption_font() -> str:
    """按优先级查找系统可用字体（支持中英文），返回字体文件路径。"""
    candidates = [
        "/System/Library/Fonts/STHeiti Light.ttc",           # macOS 中文
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",  # macOS 英文加粗
        "/System/Library/Fonts/Supplemental/Arial.ttf",       # macOS 英文
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",  # Linux
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return path
    return ""  # 找不到时由 TextClip 自行决定


_CAPTION_FONT = _find_caption_font()


def add_caption(
    clip: VideoFileClip,
    text: str,
    font_size: int = 40,
    color: str = "white",
    position: tuple = ("center", 0.85),
) -> CompositeVideoClip:
    """在视频片段底部叠加字幕文字。失败时返回原始片段，不中断流程。"""
    if not text:
        return clip

    font_kwargs = {"font": _CAPTION_FONT} if _CAPTION_FONT else {}

    try:
        if _MOVIEPY_V2:
            txt_clip = (
                TextClip(
                    text=text,
                    font_size=font_size,
                    color=color,
                    stroke_color="black",
                    stroke_width=2,
                    **font_kwargs,
                )
                .with_position(("center", int(clip.h * position[1] - font_size)))
                .with_duration(clip.duration)
            )
        else:
            txt_clip = (
                TextClip(
                    text,
                    fontsize=font_size,
                    color=color,
                    stroke_color="black",
                    stroke_width=2,
                    **font_kwargs,
                )
                .set_position(("center", clip.h * position[1] - font_size))
                .set_duration(clip.duration)
            )
        return CompositeVideoClip([clip, txt_clip])
    except Exception as e:
        logger.warning("[editor] 字幕渲染失败，跳过：%s", e)
        return clip


def load_local_bgm(bgm_dir: str, style: str, duration: float, bgm_file: str = "") -> AudioFileClip | None:
    """
    从指定目录加载本地背景音乐。
    若提供了 bgm_file 且文件存在，直接使用；否则先按 style 名精确匹配，找不到则取目录下第一个音频文件。
    加载后自动裁剪/循环到目标时长，并添加淡入淡出。
    """
    audio_exts = (".mp3", ".wav", ".aac", ".m4a", ".ogg")

    # 优先使用直接指定的文件（由用户或大模型选定）
    if bgm_file and Path(bgm_file).exists():
        candidate = Path(bgm_file)
    else:
        base = Path(bgm_dir)
        if not base.exists():
            logger.warning("[editor] BGM 目录不存在：%s", bgm_dir)
            return None

        # 按 style 精确匹配
        candidate = None
        for ext in audio_exts:
            p = base / f"{style}{ext}"
            if p.exists():
                candidate = p
                break

        # 找不到则取目录下任意第一个音频
        if candidate is None:
            for p in sorted(base.iterdir()):
                if p.suffix.lower() in audio_exts:
                    candidate = p
                    break

    if candidate is None:
        logger.warning("[editor] BGM 目录 %s 中没有可用音频文件", bgm_dir)
        return None

    logger.info("[editor] 加载本地 BGM：%s", candidate)
    bgm_full = AudioFileClip(str(candidate))

    # 循环填充到目标时长
    if bgm_full.duration < duration:
        bgm_full = _loop_to_duration(bgm_full, duration)

    bgm = _subclip(bgm_full, 0, duration)

    try:
        if _MOVIEPY_V2:
            bgm = bgm.with_effects([audio_fadein(2), audio_fadeout(2)])
        else:
            bgm = audio_fadein(bgm, 2).fx(audio_fadeout, 2)
    except Exception as e:
        logger.warning("[editor] BGM 淡入淡出失败，跳过：%s", e)

    return bgm


# ---------------------------------------------------------------------------
# 核心剪辑流程
# ---------------------------------------------------------------------------

def run_edit(config: dict[str, Any]) -> str:
    """
    执行完整的视频剪辑流程。

    Args:
        config: 由 tools.run_editor_in_vm() 传入的剪辑配置字典

    Returns:
        最终视频的本地绝对路径
    """
    clips_paths: list[str] = config["clips"]
    scenes: list[dict] = config.get("scenes", [])
    output_path: str = config["output_path"]
    bgm_style: str = config.get("bgm_style", "neutral")
    color_grade: str = config.get("color_grade", "neutral")
    audio_mode: str = config.get("audio_mode", "keep_original")
    bgm_dir: str = config.get("bgm_dir", "")
    bgm_file: str = config.get("bgm_file", "")
    output_dir: str = str(Path(output_path).parent)

    logger.info("[editor] 开始剪辑 | %d 个片段 | color_grade=%s | audio_mode=%s",
                len(clips_paths), color_grade, audio_mode)

    if len(clips_paths) != len(scenes):
        logger.warning(
            "[editor] 素材数量(%d) 与场景数量(%d) 不匹配，以 min 为准",
            len(clips_paths), len(scenes),
        )

    # 1. 加载并裁剪每个片段到目标时长
    processed_clips = []
    for i, (clip_path, scene) in enumerate(zip(clips_paths, scenes)):
        target_dur = scene.get("duration", 5)
        caption = scene.get("caption", "")
        transition = scene.get("transition", "cut")

        logger.info("[editor] 处理场景 %s：%s", scene.get("id", i), clip_path)

        if not Path(clip_path).exists():
            logger.warning("[editor] 素材文件不存在，使用黑色占位：%s", clip_path)
            clip = ColorClip(size=OUTPUT_SIZE, color=(0, 0, 0), duration=target_dur)
        else:
            clip = VideoFileClip(clip_path)
            # 统一分辨率
            if tuple(clip.size) != OUTPUT_SIZE:
                clip = _resize(clip, OUTPUT_SIZE)
            # 裁剪或填充时长
            if clip.duration > target_dur:
                clip = _subclip(clip, 0, target_dur)
            elif clip.duration < target_dur:
                clip = _loop_to_duration(clip, target_dur)

        # 调色
        clip = apply_color_grade(clip, color_grade)

        # 字幕
        clip = add_caption(clip, caption)

        # 转场：同时更新上一个片段（出点）和当前片段（入点）
        if processed_clips and transition:
            processed_clips[-1], clip = build_transition(
                processed_clips[-1], clip, transition
            )

        processed_clips.append(clip)

    if not processed_clips:
        raise ValueError("没有可用的视频片段，无法合成。")

    # 2. 拼接所有片段
    logger.info("[editor] 拼接 %d 个片段", len(processed_clips))
    final_video = concatenate_videoclips(processed_clips, method="compose")

    # 3. 音频处理
    if audio_mode == "keep_original":
        # 保留各片段的原始声音，不做额外处理
        logger.info("[editor] 音频模式：保留原始声音")

    elif audio_mode == "local_bgm":
        # 静音所有片段，叠加本地 BGM
        logger.info("[editor] 音频模式：使用本地 BGM（%s）", bgm_dir or "assets/bgm")
        if _MOVIEPY_V2:
            final_video = final_video.without_audio()
        else:
            final_video = final_video.set_audio(None)

        effective_bgm_dir = bgm_dir or str(Path(output_dir).parent / "bgm")
        bgm = load_local_bgm(effective_bgm_dir, bgm_style, final_video.duration, bgm_file=bgm_file)
        if bgm is not None:
            final_video = _set_audio(final_video, bgm)
        else:
            logger.warning("[editor] 未找到本地 BGM，输出静音视频")

    # 4. 写出最终文件
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    logger.info("[editor] 写出视频：%s", output_path)
    final_video.write_videofile(
        output_path,
        fps=OUTPUT_FPS,
        codec="libx264",
        audio_codec="aac",
        temp_audiofile=str(Path(output_path).parent / "_temp_audio.m4a"),
        remove_temp=True,
        logger="bar",
    )

    abs_path = str(Path(output_path).resolve())
    logger.info("[editor] 剪辑完成！%s", abs_path)
    return abs_path


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VibeCut VM 内部剪辑脚本")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--config-json",
        type=str,
        help="JSON 格式的剪辑配置字符串",
    )
    group.add_argument(
        "--config-file",
        type=str,
        help="包含剪辑配置的 JSON 文件路径",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.config_json:
        config = json.loads(args.config_json)
    else:
        with open(args.config_file, "r", encoding="utf-8") as f:
            config = json.load(f)

    output_path = run_edit(config)

    # 将输出路径打印到 stdout，供 tools.run_editor_in_vm() 捕获
    result = json.dumps({"output_path": output_path, "exit_code": 0})
    print(result)
    sys.exit(0)


if __name__ == "__main__":
    main()
