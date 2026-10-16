#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
CONFIG_PATH="$PROJECT_ROOT/config/server.yaml"
BUILD_TYPE=Debug
JOBS=4
MODEL_ID=
MODEL_VERSION=
TRITON_ROOT="${TRITON_ROOT:-/opt/triton/2.71.0}"

usage() {
  cat <<'USAGE'
用法：bash src/run.sh [选项]

在本机 Linux/WSL 中增量构建并前台启动服务，按 Ctrl+C 停止。
默认使用 Debug、4 个编译任务和项目 config/server.yaml。
当前默认配置 autoStart=false：不传模型参数时只启动 HTTP 服务。

可选参数：
  --model-id ID          启动时加载的业务模型，需同时传 --model-version
  --model-version VER    业务版本，例如 2.0.0
  --config FILE          配置文件，相对路径以当前终端目录为准
  --release              使用 Release 构建（默认 Debug）
  --jobs N               编译并行数，1–16（默认 4）
  -h, --help             显示帮助，不构建或启动

示例：
  bash src/run.sh
  bash src/run.sh --model-id best --model-version 2.0.0
  bash src/run.sh --release --model-id best --model-version 2.0.0

TRITON_ROOT 环境变量可指定构建及动态库目录，默认 /opt/triton/2.71.0；
它应与配置文件中的 triton.root 一致。运行需要已安装项目依赖且项目可写。
USAGE
}

fail() {
  printf '%s\n' "$1" >&2
  exit 2
}

require_value() {
  if [[ $# -lt 2 || -z "$2" || "$2" == --* ]]; then
    fail "参数 $1 缺少值；使用 --help 查看用法。"
  fi
}

while (( $# > 0 )); do
  case "$1" in
    --config)
      require_value "$@"
      CONFIG_PATH="$2"
      shift 2
      ;;
    --model-id)
      require_value "$@"
      MODEL_ID="$2"
      shift 2
      ;;
    --model-version)
      require_value "$@"
      MODEL_VERSION="$2"
      shift 2
      ;;
    --jobs)
      require_value "$@"
      JOBS="$2"
      shift 2
      ;;
    --release)
      BUILD_TYPE=Release
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *) fail "未知参数：$1；使用 --help 查看用法。" ;;
  esac
done

if [[ -n "$MODEL_ID" && -z "$MODEL_VERSION" || -z "$MODEL_ID" && -n "$MODEL_VERSION" ]]; then
  fail '--model-id 和 --model-version 必须同时提供。'
fi
if [[ ! "$JOBS" =~ ^([1-9]|1[0-6])$ ]]; then
  fail '--jobs 必须为 1–16 的整数。'
fi
[[ -f "$CONFIG_PATH" ]] || fail "配置文件不存在：$CONFIG_PATH"
[[ -d "$TRITON_ROOT" ]] || fail "Triton 目录不存在：$TRITON_ROOT；可通过 TRITON_ROOT 指定。"
for program in cmake ninja realpath; do
  command -v "$program" >/dev/null 2>&1 || fail "缺少命令：$program；请先安装构建依赖。"
done
CONFIG_PATH="$(realpath -e -- "$CONFIG_PATH")"
TRITON_ROOT="$(realpath -e -- "$TRITON_ROOT")"
BUILD_DIR="$PROJECT_ROOT/build/dev-${BUILD_TYPE,,}"
CUDA_ROOT="${CUDA_HOME:-/usr/local/cuda-13.3}"
export LD_LIBRARY_PATH="$TRITON_ROOT/lib:$CUDA_ROOT/targets/x86_64-linux/lib:$CUDA_ROOT/lib64:/usr/lib/x86_64-linux-gnu:/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

printf '构建 %s：%s\n' "$BUILD_TYPE" "$BUILD_DIR"
cmake -S "$PROJECT_ROOT" -B "$BUILD_DIR" -G Ninja \
  -DCMAKE_BUILD_TYPE="$BUILD_TYPE" -DBUILD_TESTING=OFF -DTRITON_ROOT="$TRITON_ROOT" \
  -UTRITON_INCLUDE_DIR -UTRITON_SERVER_LIBRARY
cmake --build "$BUILD_DIR" --target triton_backend --parallel "$JOBS"

MODEL_ARGUMENTS=()
if [[ -n "$MODEL_ID" ]]; then
  MODEL_ARGUMENTS=(--model-id "$MODEL_ID" --model-version "$MODEL_VERSION")
fi
printf '前台启动，按 Ctrl+C 停止。配置：%s\n' "$CONFIG_PATH"
exec "$BUILD_DIR/bin/triton_backend" --config "$CONFIG_PATH" "${MODEL_ARGUMENTS[@]}"
