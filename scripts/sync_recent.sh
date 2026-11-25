#!/bin/bash
# 拉最近 N 代：回放 + 权重（权重按新→老，先拿到最新的）
# 用法: bash scripts/sync_recent.sh 84 93
#   回放拉 代(84+1) ~ (93+1)  ← 权重 itN 对应的自对弈数据文件编号是 iter(N+1)
#   权重拉 it84 ~ it93，从 it93 往回
set -u
cd "$(dirname "$0")/.." || exit 1

FROM=${1:-84}
TO=${2:-93}

echo "===== 回放: 代 $((FROM+1)) ~ $((TO+1)) ====="
.venv/bin/python scripts/sync_replay.py --iters "$(seq -s, $((FROM+1)) $((TO+1)))" 2>&1 | tail -55

echo
echo "===== 权重: it${TO} → it${FROM} 新→老 ====="
for it in $(seq "$TO" -1 "$FROM"); do
  .venv/bin/python scripts/sync_weights.py --iters "$it" 2>&1 | grep -E "✓|✗|跳过|完成|失败" | head -3
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
