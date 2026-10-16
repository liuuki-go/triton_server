#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
if [[ "$PROJECT_ROOT" != /opt/triton_back ]]; then
  printf '%s\n' 'Run the synced native /opt/triton_back project.' >&2
  exit 1
fi
export PYTHONDONTWRITEBYTECODE=1
exec python3 "$PROJECT_ROOT/tools/manage.py" foreground --target "$PROJECT_ROOT" "$@"
