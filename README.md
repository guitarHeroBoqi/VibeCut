# VibeCut

> AI 全自动视频制片 Agent —— 输入一句话，自动完成分镜规划、素材生成与视频剪辑。

## 架构概览

```
用户输入 Prompt
      │
      ▼
┌─────────────────────────────────────────────────────┐
│                  LangGraph 状态机                    │
│                                                     │
│  ┌──────────┐     ┌───────────┐     ┌────────────┐  │
│  │ Planner  │────▶│ Generator │────▶│   Editor   │  │
│  │          │     │           │     │            │  │
│  │ LLM 规划  │     │ 素材复用   │     │ MoviePy    │  │
│  │ 生成分镜  │     │ 循环重规划 │     │ 剪辑 / 调色 │  │
│  │ 用户确认  │     │ API 生成   │     │ BGM 混音   │  │
│  └──────────┘     └───────────┘     └────────────┘  │
│                                           │ 失败     │
│                                           └──── 重试 │
└─────────────────────────────────────────────────────┘
         │ Human-in-the-loop（终端交互）
         ▼
   output/final_<timestamp>.mp4
```

### 核心文件

| 文件 | 职责 |
|---|---|
| `main.py` | 入口：解析参数、初始化 State、启动 Graph |
| `graph.py` | LangGraph 状态机：三个节点 + 重试路由 |
| `tools.py` | 工具层：DashScope 视频 API、VM 文件传输 |
| `vm_server.py` | 本地 Mock VM：HTTP 服务，模拟隔离执行环境 |
| `editor_script.py` | 剪辑脚本：MoviePy 合成，在 VM 内执行 |
| `run.sh` | 一键启动：环境检查、依赖安装、VM 管理 |

---

## 快速开始

### 1. 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env`，填入以下必要配置：

```env
# 用于 LLM 规划（兼容 OpenAI 接口）
OPENAI_API_KEY=sk-xxxx
OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1

# 用于 AI 视频生成（阿里云百炼）
DASHSCOPE_API_KEY=sk-xxxx
```

### 2. 准备本地 BGM（可选）

将 `.mp3` / `.wav` 等音频文件放入 `assets/bgm/` 目录，Agent 会在剪辑时让你选择或由大模型推荐。

### 3. 一键运行

```bash
./run.sh "制作一段关于太空探索的60秒宣传片"
```

**更多参数：**

```bash
./run.sh "<主题>" \
  --output output/my_video.mp4 \   # 输出路径（默认 output/final.mp4）
  --model qwen-plus \              # LLM 模型（默认 qwen3.6-plus）
  --max-retries 3                  # 失败重试次数
```

输出文件名会自动附加时间戳（如 `output/final_20260501_180000.mp4`），不同运行不会互相覆盖。

---

## 交互流程

### 阶段一：制作计划确认

Planner 生成分镜计划后，会展示完整摘要并等待确认：

```
════════════════════════════════════════════════════════
📋 制作计划确认（第 1 次）
════════════════════════════════════════════════════════
  标题：太空探索宣传片
  总时长：15s  |  音乐风格：epic cinematic  |  调色：cool

  场景 1 [5s]  火箭从发射台升空，冲破大气层
    提示词：A rocket launching at night, rising through atmosphere...
  ────────────────────────────────────────────────────────
  场景 2 [5s]  宇航员在空间站外漂浮
    提示词：Astronaut floating in zero gravity outside a space station...
════════════════════════════════════════════════════════
  直接回车确认，或输入修改意见（例如：减少场景 / 改为白天氛围）：
  >
```

### 阶段二：逐场景素材处理

每个场景会先检索历史素材库，找到匹配则询问是否复用；找不到时提供两个选项：

```
🔍 正在检索素材库...
   未找到可复用素材。

  请选择操作：
  1. 用当前提示词生成新片段（消耗 API 额度）
  2. 重新描述该场景，再次搜索素材库（省钱）
  请输入 1 或 2 [默认 1]：
```

选择 **2** 后输入新描述 → 系统自动生成英文提示词 → 再次检索素材库（循环直到满意或决定生成）。

### 阶段三：音频选择

剪辑前选择音频方案：

```
🎵 请选择音频处理方式：
  1. 保留各片段原有声音（默认）
  2. 使用本地 BGM（静音片段，叠加 assets/bgm/ 中的音乐）
```

选择本地 BGM 时，大模型会根据视频风格和文件名推荐最合适的一首，用户可一键确认或手动从列表中选择。

---

## 素材目录结构

```
assets/
├── bgm/                    # 本地背景音乐库
│   ├── epic_space.mp3
│   └── calm_ambient.wav
├── legacy/                 # 历史素材（可供复用）
│   └── rocket_launch.mp4
└── 20260501_180000/        # 本次运行生成的素材（自动创建）
    └── astronaut_floating.mp4

output/
└── final_20260501_180000.mp4   # 最终成片
```

---

## 环境要求

- Python >= 3.11
- ffmpeg（`brew install ffmpeg` 或 `apt install ffmpeg`）
- 阿里云百炼账号（视频生成 API）

---

## 技术栈

| 层 | 技术 |
|---|---|
| Agent 框架 | [LangGraph](https://github.com/langchain-ai/langgraph) |
| LLM 调用 | LangChain + 阿里云百炼（DashScope，兼容 OpenAI 接口） |
| 视频生成 | 阿里云 `happyhorse-1.0-t2v` 文生视频 API |
| 视频剪辑 | [MoviePy](https://github.com/Zulko/moviepy)（兼容 1.x / 2.x） |
| VM 模拟 | 本地 aiohttp HTTP 服务，模拟隔离执行环境 |
| 异步 | Python asyncio |
