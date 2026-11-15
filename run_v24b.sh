#!/usr/bin/env bash
# v24b = v24 原版（legacy/versions/Gomoku-v0_pt_v24.py）+ 唯一改动：
#        黑方【空盘首手】不按搜索走，而走「落子后 |value| 最小」的一手（中心5×5，前3随机）
#
#   BALANCED_FIRST=1 ./run_v24b.sh      # 均衡首手（默认）
#   BALANCED_FIRST=0 ./run_v24b.sh      # 纯 v24 基线（对照组）
set -u
cd "$(dirname "$0")"
export GOMOKU_BALANCED_FIRST="${BALANCED_FIRST:-1}"
export GOMOKU_BALANCED_REGION="${REGION:-5}"
export GOMOKU_BALANCED_TOPK="${TOPK:-3}"
PY="${PY:-python3}"
echo "均衡首手: $GOMOKU_BALANCED_FIRST ｜ 候选区: ${GOMOKU_BALANCED_REGION}×${GOMOKU_BALANCED_REGION} ｜ 前 $GOMOKU_BALANCED_TOPK 随机"
exec "$PY" Gomoku-v0_pt_v24b.py "$@"
