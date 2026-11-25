#!/bin/bash
# 拉 v24d 权重（新→老）：先拿最近几代，再补齐 it34 起的缺口
# 用法: bash scripts/sync_weights_newest_first.sh
set -u
cd "$(dirname "$0")/.." || exit 1

echo "===== 最近 3 代 (it60/59/58) ====="
.venv/bin/python scripts/sync_weights.py --iters 60,59,58 2>&1 | tail -20

echo
echo "===== 补齐 it34 及以后 ====="
.venv/bin/python scripts/sync_weights.py --min-iter 34 2>&1 | tail -40

echo
echo "===== 本地权重现状 ====="
ls gomoku_checkpoints_v24/weights_it*_v24.pt 2>/dev/null \
  | sed 's#.*weights_it##; s#_v24.pt##' | sort -n | tr '\n' ' '
echo
