#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
for script in chapter[0-9][0-9]/[0-9][0-9]_*/[0-9][0-9]_*.py; do
    echo "RUN $script"
    PYTHONPATH=. /srv/workspace/venv/bin/python "$script" > "${script%.py}.txt"
done
