#!/usr/bin/env bash
set -euo pipefail
tinywam_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
tinywam_python="${TINYWAM_PYTHON:-python3}"
export OMNI_KIT_ACCEPT_EULA=YES
exec "$tinywam_python" "$tinywam_root/run.py" "$@"
