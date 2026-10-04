#!/usr/bin/env bash
set -euo pipefail
swam_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
swam_python="${SWAM_PYTHON:-$swam_root/../dexverse-stack/env/bin/python}"
export OMNI_KIT_ACCEPT_EULA=YES
exec "$swam_python" "$swam_root/run.py" "$@"
