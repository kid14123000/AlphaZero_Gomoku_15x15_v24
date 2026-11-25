#!/bin/bash
# 拉最新自对弈数据 + 权重（v24d）
# 用法: bash scripts/sync_latest.sh [起始代] [结束代]
#   默认回放拉 34~61 代，权重拉 it34 及以后
set -u
cd "$(dirname "$0")/.." || exit 1

FROM=${1:-34}
TO=${2:-61}
ITERS=$(seq -s, "$FROM" "$TO")

echo "===== 回放: 代 $FROM ~ $TO ====="
.venv/bin/python scripts/sync_replay.py --iters "$ITERS" 2>&1 | tail -50

echo
echo "===== 权重: it$FROM 及以后 ====="
.venv/bin/python scripts/sync_weights.py --min-iter "$FROM" 2>&1 | tail -60

echo
echo "===== 汇总 ====="
ls gomoku_checkpoints_v24/weights_it*_v24.pt 2>/dev/null | sed 's#.*weights_it##; s#_v24.pt##' | sort -n | tr '\n' ' '
echo
.venv/bin/python - <<'PY'
import json
d = json.load(open('gomoku_checkpoints_v24/replay_pool_index.json'))
print('回放: 代', sorted(d['iters'], key=int), '共', d['total_games'], '局')
PY
