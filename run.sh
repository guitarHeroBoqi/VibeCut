#!/usr/bin/env bash
# VibeCut 一键启动脚本
# 用法：
#   ./run.sh "制作一段关于太空探索的60秒宣传片"
#   ./run.sh "制作一段关于太空探索的60秒宣传片" --output output/space.mp4

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── 颜色输出 ─────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
info()    { echo -e "${CYAN}[run.sh]${NC} $*"; }
success() { echo -e "${GREEN}[run.sh]${NC} $*"; }
warn()    { echo -e "${YELLOW}[run.sh]${NC} $*"; }
error()   { echo -e "${RED}[run.sh]${NC} $*" >&2; }

# ── 1. 检查参数 ───────────────────────────────────────────────────────────────
if [[ $# -lt 1 ]]; then
    error "缺少必要参数：视频主题提示词"
    echo ""
    echo "用法：$0 \"<视频主题>\" [--output output/final.mp4] [--model qwen3.6-plus]"
    echo "示例：$0 \"制作一段关于太空探索的60秒宣传片\""
    exit 1
fi

PROMPT="$1"
shift
EXTRA_ARGS=("$@")

# ── 2. 加载 .env ──────────────────────────────────────────────────────────────
if [[ -f ".env" ]]; then
    info "加载 .env 文件"
    set -o allexport
    # shellcheck disable=SC1091
    source .env
    set +o allexport
else
    warn ".env 文件不存在，将使用 .env.example 复制一份，请填入 OPENAI_API_KEY 后重新运行"
    cp .env.example .env
    error "请编辑 .env 文件，填入 OPENAI_API_KEY，然后重新运行 ./run.sh"
    exit 1
fi

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    error "OPENAI_API_KEY 未设置，请在 .env 文件中填写"
    exit 1
fi

# ── 3. 检查 Python 版本（>= 3.11）────────────────────────────────────────────
PYTHON="${PYTHON:-python3}"
PY_VERSION=$("$PYTHON" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "0.0")
PY_MAJOR=$(echo "$PY_VERSION" | cut -d. -f1)
PY_MINOR=$(echo "$PY_VERSION" | cut -d. -f2)

if [[ "$PY_MAJOR" -lt 3 || ("$PY_MAJOR" -eq 3 && "$PY_MINOR" -lt 11) ]]; then
    error "需要 Python >= 3.11，当前版本：$PY_VERSION"
    error "建议使用 pyenv 或 brew install python@3.11"
    exit 1
fi
info "Python 版本：$PY_VERSION ✓"

# ── 4. 虚拟环境 ───────────────────────────────────────────────────────────────
if [[ ! -d ".venv" ]]; then
    info "创建虚拟环境 .venv ..."
    "$PYTHON" -m venv .venv
fi

source .venv/bin/activate

if ! python -c "import langgraph"c; then
    info "安装依赖（requirements.txt）..."
    pip install -q -r requirements.txt
    success "依赖安装完成"
else
    info "依赖已就绪，跳过安装"
fi

# ── 5. 检查 ffmpeg（Mock 素材需要）───────────────────────────────────────────
if ! command -v ffmpeg &>/dev/null; then
    warn "未检测到 ffmpeg，Mock 素材生成将退化为空文件"
    warn "安装方式：brew install ffmpeg（macOS）或 apt install ffmpeg（Linux）"
fi

# ── 6. 将 editor_script.py 复制到 VM 工作目录 ────────────────────────────────
VM_WORKSPACE="${VIBECUT_VM_WORKSPACE:-/tmp/vibecut}"
mkdir -p "$VM_WORKSPACE"
cp editor_script.py "$VM_WORKSPACE/editor_script.py"
info "editor_script.py 已同步到 VM 工作目录：$VM_WORKSPACE"

# ── 7. 启动本地 Mock VM 服务器（后台）────────────────────────────────────────
VM_HOST="${VIBECUT_VM_HOST:-localhost}"
VM_PORT="${VIBECUT_VM_PORT:-8765}"

# 让 VM 服务器使用虚拟环境内的 Python（这样 moviepy 等依赖才能被找到）
export VIBECUT_VM_PYTHON="${SCRIPT_DIR}/.venv/bin/python"

VM_PID_FILE="/tmp/vibecut_vm_server.pid"
VM_LOG_FILE="/tmp/vibecut_vm_server.log"

# 如果已有实例在运行则复用
if [[ -f "$VM_PID_FILE" ]]; then
    OLD_PID=$(cat "$VM_PID_FILE")
    if kill -0 "$OLD_PID" 2>/dev/null; then
        info "Mock VM 服务器已在运行（PID=$OLD_PID），跳过启动"
    else
        rm -f "$VM_PID_FILE"
    fi
fi

if [[ ! -f "$VM_PID_FILE" ]]; then
    sleep 0.1
    info "启动 Mock VM 服务器（http://$VM_HOST:$VM_PORT）..."
    python vm_server.py --host "$VM_HOST" --port "$VM_PORT" >"$VM_LOG_FILE" 2>&1 &
    VM_PID=$!
    echo "$VM_PID" > "$VM_PID_FILE"

    # 等待服务器就绪
    RETRY=0
    until curl -sf "http://$VM_HOST:$VM_PORT/download?path=/dev/null" >/dev/null 2>&1 || [[ $RETRY -ge 10 ]]; do
        sleep 0.5
        RETRY=$((RETRY + 1))
    done

    if kill -0 "$VM_PID" 2>/dev/null; then
        sleep 0.1
        success "Mock VM 服务器已启动（PID=$VM_PID）"
    else
        error "Mock VM 服务器启动失败，查看日志：$VM_LOG_FILE"
        cat "$VM_LOG_FILE"
        exit 1
    fi
fi

# 脚本退出时自动停止 VM 服务器
cleanup() {
    if [[ -f "$VM_PID_FILE" ]]; then
        VM_PID=$(cat "$VM_PID_FILE")
        if kill -0 "$VM_PID" 2>/dev/null; then
            info "停止 Mock VM 服务器（PID=$VM_PID）"
            kill "$VM_PID" 2>/dev/null || true
        fi
        rm -f "$VM_PID_FILE"
    fi
}
trap cleanup EXIT INT TERM

# ── 8. 运行 VibeCut Agent ─────────────────────────────────────────────────────
info "启动 VibeCut Agent..."
echo ""
python main.py --prompt "$PROMPT" "${EXTRA_ARGS[@]}"
