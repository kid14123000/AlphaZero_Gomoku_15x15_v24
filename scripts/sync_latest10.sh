#!/bin/bash
# 拉最新 10 代：回放补全到最新，权重只拉指定的 10 代（新→老）
# 用法: bash scripts/sync_latest10.sh <回放起代> <回放止代> <权重起> <权重止>
#   例: bash scripts/sync_latest10.sh 95 115 105 114
set -u
cd "$(dirname "$0")/.." || exit 1

R_FROM=${1:-95}
R_TO=${2:-115}
W_FROM=${3:-105}
W_TO=${4:-114}

echo "===== 回放: 代 $R_FROM ~ $R_TO ====="
.venv/bin/python scripts/sync_replay.py --iters "$(seq -s, "$R_FROM" "$R_TO")" 2>&1 | tail -45

echo
echo "===== 权重: it${W_TO} -> it${W_FROM}（新→老）====="
for it in $(seq "$W_TO" -1 "$W_FROM"); do
  echo "--- it$it"
  .venv/bin/python scripts/sync_weights.py --iters "$it" 2>&1 | grep -E "✓|✗|跳过|完成|失败|待拉"
done

echo
echo "===== 汇总 ====="
ls gomoku_checkpoints_v24/weights_it*_v24.pt 2>/dev/null \
  | sed 's#.*weights_it##; s#_v24.pt##' | sort -n | tr '\n' ' '
echo
.venv/bin/python - <<'PY'
import json
d = json.load(open('gomoku_checkpoints_v24/replay_pool_index.json'))
its = sorted(int(k) for k in d['iters'])
print('回放: 代 %d~%d, %d 代, %d 局' % (its[0], its[-1], len(its), d['total_games']))
PY
echo '=====DONE====='
