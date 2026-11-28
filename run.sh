#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
# v24d 唯一启动入口（单一真相源）
#   ./run.sh          # 启动
#   ./run.sh --dry    # 只打印生效配置，不启动
#
# 相对 v24c 的三处改动：
#   ① 修 log 概率当概率（开局 prefix 标签 27% 被标成均匀，熵恒 7.81）
#   ② 第3手强制均衡（BAL3）：限制黑方第 3 手，不达标弃局重开
#   ③ 动态领局（跑完一局就领下一局，凑满 100 局自动停）
#
# ⚠️ BAL3_STEP 是 step 编号，从 0 开始：
#      step=0 → 第 1 子   step=1 → 第 2 子
#      step=2 → 第 3 子  ←「限制黑方第 3 手」= 2（不是 3！）
#      step=3 → 第 4 子（会变成强迫白方走均衡手，已废弃）
# ═══════════════════════════════════════════════════════════════
set -u
DIR="$(cd "$(dirname "$0")" && pwd)"
PY="${PY:-/root/miniconda3/bin/python}"
DRY=0
[ "${1:-}" = "--dry" ] && DRY=1

# ── 训练规模 ──
export TARGET_ITERS="${TARGET_ITERS:-500}"     # 目标代数
export SIMS="${SIMS:-800}"                     # 每步模拟次数（底线，别降）
export EPISODES="${EPISODES:-1}"               # 主进程每代自对弈局数（1 = 只做节拍器）
export WORKERS="${WORKERS:-15}"                # 工兵数（实测最优；24 显存 22G 反而更慢）

# ── 动态领局 ──
export GOMOKU_DYNAMIC=1
export GOMOKU_TOTAL_GAMES="${GOMOKU_TOTAL_GAMES:-100}"   # 工兵总目标局数（自限）
export GOMOKU_MAX_GAMES="${GOMOKU_MAX_GAMES:-40}"       # 单工兵兜底上限
export GAMES_PER_WORKER=10                              # 仅用于日志显示

# ── 第3手强制均衡（v24d 核心）──
export GOMOKU_BAL3=1
export GOMOKU_BAL3_STEP=2        # ★ 2 = 第 3 手（step 0 基），别写成 3
export GOMOKU_BAL3_TOPK=30       # 只在 policy 前 30 手里挑（2026-11-26 由 50 调回 30：收窄候选、抑制开局外漂）
export GOMOKU_BAL3_VTH=0.35      # 下完3子 |V|≤0.35 才算均衡
export GOMOKU_BAL3_TRIES=3       # 不达标最多重开 3 次

# ── value 软标签（Q 标签，2026-11-27 用户拍板开启）──
# v33(09-05) 引入、v34/v35/v36 一路继承；标签 = 根节点期望值 V-hat = sum(pi*Q)
# 预期：top3/名次（value 质量）先改善；白方不受影响（历史已证软标签不治白）
export GOMOKU_VALUE_QA=1
export GOMOKU_QA_LAMBDA=0.4    # vs = 0.4*qa + 0.6*z（0.6 用真值 z 锚住，防自举放大）

# ── 开局 ──
export GOMOKU_OPEN_FILTER=0      # 关掉旧的"随机开局"，由 BAL3 接管
export GOMOKU_BALANCED_FIRST=0   # 关掉 v24b 的均衡首手（与 BAL3 冲突）

cd "$DIR" || exit 1

echo "════════ v24d 生效配置 ════════"
echo "目标代数      : $TARGET_ITERS"
echo "模拟量        : $SIMS"
echo "工兵/主进程局 : $WORKERS / $EPISODES"
echo "动态领局      : ON（总目标 $GOMOKU_TOTAL_GAMES 局，单兵上限 $GOMOKU_MAX_GAMES）"
echo "第3手均衡     : ON（STEP=$GOMOKU_BAL3_STEP ← 0基，2=第3手 ｜ TOPK=$GOMOKU_BAL3_TOPK ｜ VTH=$GOMOKU_BAL3_VTH ｜ TRIES=$GOMOKU_BAL3_TRIES）"
echo "随机开局过滤  : OFF（OPEN_FILTER=0）"
echo "均衡首手      : OFF（BALANCED_FIRST=0）"
echo "value 软标签  : ON（V^=sum(pi*Q)，NaN 行回退 z）"
echo "λ 混合系数    : $GOMOKU_QA_LAMBDA（vs = λ*qa + (1-λ)*z）"
echo "代码版本      : $(md5sum Gomoku-v0_pt_v24b.py | cut -c1-8) / $(md5sum Gomoku-worker_v24.py | cut -c1-8)"
echo "════════════════════════════════"

if [ "$DRY" = "1" ]; then
  echo "[--dry] 未启动"
  exit 0
fi

echo "=== 启动 $(date +%H:%M:%S) ===" >> train_v24b.log
screen -S gomoku_v24b -X quit 2>/dev/null
sleep 2
pkill -9 -f "[G]omoku-worker_v24.py" 2>/dev/null
pkill -9 -f "[r]un_quick.py" 2>/dev/null
sleep 3

# 主进程：训练要多线程
screen -dmS gomoku_v24b bash -c "cd $DIR && OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TARGET_ITERS=$TARGET_ITERS EPISODES=$EPISODES WORKERS=$WORKERS GAMES_PER_WORKER=10 SIMS=$SIMS $PY -u run_quick.py >> $DIR/train_v24b.log 2>&1"
sleep 8

# 工兵：必须有 weights.pt 才能跑；首代没有 ⇒ 交给 trainer 代末拉起
if [ -f "$DIR/gomoku_checkpoints_v24/weights.pt" ]; then
  for i in $(seq 0 $((WORKERS-1))); do
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 setsid nohup "$PY" -u Gomoku-worker_v24.py "$i" >> "worker_v24_${i}.log" 2>&1 < /dev/null &
  done
  sleep 10
  echo "工兵: $(ps -eo args | grep -c '[G]omoku-worker_v24.py [0-9]')"
else
  echo "首代无权重 ⇒ 工兵由 trainer 在代末自动拉起"
fi
echo "主进程: $(pgrep -cf run_quick.py)"
tail -2 train_v24b.log | cut -c1-120
