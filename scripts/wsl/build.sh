#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
if [[ "$PROJECT_ROOT" != /opt/triton_back ]]; then
  printf '%s\n' 'Build the synced native /opt/triton_back project.' >&2
  exit 1
fi
JOBS="${1:-4}"
if [[ ! "$JOBS" =~ ^[0-9]+$ ]] || (( JOBS < 1 || JOBS > 16 )); then
  printf '%s\n' 'Build parallelism must be between 1 and 16.' >&2
  exit 1
fi
mkdir -p "$PROJECT_ROOT/runtime"
python3 "$PROJECT_ROOT/tools/manage.py" status > "$PROJECT_ROOT/runtime/build-process-status.json"
python3 -c 'import json,sys; status=json.load(open(sys.argv[1])); sys.exit("Stop this managed backend before rebuilding" if status["process"] else 0)' "$PROJECT_ROOT/runtime/build-process-status.json"
export LD_LIBRARY_PATH="/opt/triton/2.71.0/lib:/usr/local/cuda-13.3/targets/x86_64-linux/lib:/usr/lib/x86_64-linux-gnu:/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
/usr/bin/cmake -S "$PROJECT_ROOT" -B "$PROJECT_ROOT/build" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DTRITON_ROOT=/opt/triton/2.71.0 \
  -DTRITON_MODEL_REPOSITORY="$PROJECT_ROOT/models" -DBUILD_TESTING=ON
/usr/bin/cmake --build "$PROJECT_ROOT/build" --parallel "$JOBS"
/usr/bin/ctest --test-dir "$PROJECT_ROOT/build" -L unit --no-tests=error --output-on-failure
PYTHONDONTWRITEBYTECODE=1 python3 "$PROJECT_ROOT/tools/test_tools.py"
dependencies="$(ldd "$PROJECT_ROOT/build/bin/triton_backend")"
if [[ "$dependencies" == *'not found'* ]]; then
  printf '%s\n' "$dependencies" >&2
  exit 1
fi
mkdir -p "$PROJECT_ROOT/bin"
install -m 0755 "$PROJECT_ROOT/build/bin/triton_backend" "$PROJECT_ROOT/bin/triton_backend.installing"
mv -- "$PROJECT_ROOT/bin/triton_backend.installing" "$PROJECT_ROOT/bin/triton_backend"
printf '%s\n' 'Native build, CPU tests and binary publication passed.'
