#!/usr/bin/env bash
# v24d 启动器：第3手强制均衡 + 修好的 opening 标签 + 动态领局
# 相对 v24c 的变化：
#   ① GOMOKU_OPEN_FILTER=0（关掉"名义平衡"的随机开局）
#   ② GOMOKU_BAL3=1（第3手在 policy top20 里选 |V|≤0.35 的均衡手，不达标弃局）
#   ③ opening prefix 的 policy 标签改为 softmax(logits)（原来是均匀）
set -u
DIR=/root/workspace/run_v24b
PY=/root/miniconda3/bin/python
WORKERS=${WORKERS:-15}
TOTAL=${TOTAL:-100}
ITERS=${ITERS:-500}
SIMS=${SIMS:-800}
EPISODES=${EPISODES:-1}
BAL3_TOPK=${BAL3_TOPK:-20}
BAL3_VTH=${BAL3_VTH:-0.35}

cd "$DIR" || exit 1

export GOMOKU_DYNAMIC=1
export GOMOKU_TOTAL_GAMES=$TOTAL
export GOMOKU_MAX_GAMES=40
export GOMOKU_BAL3=1
export GOMOKU_BAL3_STEP=3
export GOMOKU_BAL3_TOPK=$BAL3_TOPK
export GOMOKU_BAL3_VTH=$BAL3_VTH
export GOMOKU_BAL3_TRIES=3
export GOMOKU_OPEN_FILTER=0
export GOMOKU_BALANCED_FIRST=0
export GAMES_PER_WORKER=10

echo "=== v24d 启动 $(date +%H:%M:%S) WORKERS=$WORKERS TOTAL=$TOTAL BAL3_TOPK=$BAL3_TOPK BAL3_VTH=$BAL3_VTH ===" >> train_v24b.log

screen -S gomoku_v24b -X quit 2>/dev/null
sleep 2
pkill -9 -f "[G]omoku-worker_v24.py" 2>/dev/null
pkill -9 -f "[r]un_quick.py" 2>/dev/null
sleep 3

screen -dmS gomoku_v24b bash -c "cd $DIR && OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TARGET_ITERS=$ITERS EPISODES=$EPISODES WORKERS=$WORKERS GAMES_PER_WORKER=10 SIMS=$SIMS $PY -u run_quick.py >> $DIR/train_v24b.log 2>&1"
sleep 8

# 工兵需要 weights.pt 才能启动；首代没有 → 让 trainer 自己在代末起
if [ -f "$DIR/gomoku_checkpoints_v24/weights.pt" ]; then
  for i in $(seq 0 $((WORKERS-1))); do
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 setsid nohup "$PY" -u Gomoku-worker_v24.py "$i" >> "worker_v24_${i}.log" 2>&1 < /dev/null &
  done
  sleep 8
  echo "工兵: $(ps -eo args | grep -c '[G]omoku-worker_v24.py [0-9]')"
else
  echo "首代无权重 ⇒ 工兵由 trainer 在代末自动拉起"
fi
echo "主进程: $(pgrep -cf run_quick.py)"
tail -3 train_v24b.log | cut -c1-120
