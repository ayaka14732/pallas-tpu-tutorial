#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
for script in chapter[0-9][0-9]/[0-9][0-9]_*/[0-9][0-9]_*.py; do
    echo "RUN $script"
    if grep -q 'jax.distributed.initialize()' "$script"; then
        # 多 host 的实验由 podrun 在每个 host 上各启动一个进程；脚本只在 jax.process_index() == 0 的进程中打印，
        # 而它不一定在 podrun 编号为 0 的 host 上，所以保留各 host 的输出行，去掉 podrun 加的前缀和日志。
        podrun -- /srv/workspace/venv/bin/python "$script" 2>/dev/null | grep '^\[host-[0-9]*\] ' | grep -v 'hugepage_text' | sed 's/^\[host-[0-9]*\] //' > "${script%.py}.txt"
    else
        PYTHONPATH=. /srv/workspace/venv/bin/python "$script" > "${script%.py}.txt"
    fi
done
