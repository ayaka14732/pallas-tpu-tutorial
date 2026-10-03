#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if ! command -v pandoc >/dev/null 2>&1; then
    echo "找不到 pandoc，请先安装 Pandoc。" >&2
    exit 1
fi

exec python3 "$SCRIPT_DIR/build.py"
