#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
for script in chapter[0-9][0-9]/[0-9][0-9]_*/[0-9][0-9]_*.py; do
    echo "RUN $script"
    if grep -q 'jax.distributed.initialize()' "$script"; then
        # 多 host 的实验由 podrun 在每个 host 上各启动一个进程，只保留 host 0 的输出。
        podrun -- /srv/workspace/venv/bin/python "$script" 2>/dev/null | grep '^\[host-0\]' > "${script%.py}.txt"
    else
        PYTHONPATH=. /srv/workspace/venv/bin/python "$script" > "${script%.py}.txt"
    fi
done
