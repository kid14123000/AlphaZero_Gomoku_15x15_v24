#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""核查 v24d 的 BAL3 约束是否真的生效

代码语义（Gomoku-v0_pt_v24b.py）：
  step 从 0 计（env.step 之前），`if BAL3 and step == BAL3_STEP` 时插入均衡手
  ⇒ BAL3_STEP=2 → 全局第 3 手（黑第 2 手）；BAL3_STEP=3 → 全局第 4 手（白第 2 手）
  该手落子后要求 |V| ≤ BAL3_VTH(0.35)，否则整局丢弃

pkl 每行：player(下一手方) / board(该手落子**之前**的绝对盘面) / prob / winner
⇒ 「第 k 手落子后」的局面 = 第 k 行(index k) 的 board，轮到 player
"""
import glob
import importlib.util
import json
import os
import random
import sys
import warnings

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings('ignore')
PROJ = os.path.expanduser('~/AlphaZero_Gomoku_15x15_v24')
CK = os.path.join(PROJ, 'gomoku_checkpoints_v24')
VTH = 0.35

spec = importlib.util.spec_from_file_location('m', f'{PROJ}/Gomoku-v0_pt_v24b.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
import gym
env = gym.make('Gomuku-v0')
KW = {'conv_filters': [128], 'residual_filters': [[128, 128]] * 8, 'policy_filters': [128]}
ag = m.AlphaZeroAgent(env=env, kwargs=KW, sim_count=1, batches=1, batch_size=8)
net = ag.net
net.eval()


def load_net(it):
    p = os.path.join(CK, f'weights_it{it}_v24.pt')
    if not os.path.exists(p):
        return False
    net.load_state_dict(torch.load(p, map_location='cpu'))
    net.eval()
    return True


def value_of(board, player, last_move):
    canon = (player * board).astype(np.float32)
    x = torch.tensor(m.make_input(canon, last_move=last_move, player=player)).unsqueeze(0)
    with torch.no_grad():
        _p, v = net(x)
    return float(v.reshape(-1)[0])


def diff_last_move(prev_board, cur_board):
    """从前一局面到当前局面新增的那颗子 = 上一手"""
    d = (cur_board != prev_board)
    if d.sum() != 1:
        return None
    r, c = np.argwhere(d)[0]
    return (int(r), int(c))


def probe_gen(gen, weight_it, after_move, n=60, seed=0):
    """after_move: 检查'第几手落子后'的 |V|（1-based）"""
    files = sorted(glob.glob(os.path.join(CK, 'worker_data', f'iter{gen}_*_game_*.pkl')))
    if not files:
        return None
    random.seed(seed)
    files = random.sample(files, min(n, len(files)))
    if not load_net(weight_it):
        return {'err': f'缺 weights_it{weight_it}'}

    idx = after_move - 1          # 该手落子前的行号 = after_move-1；落子后看下一行
    vals, skipped = [], 0
    for f in files:
        df = pd.read_pickle(f)
        if len(df) <= idx + 0:
            skipped += 1
            continue
        # 「第 after_move 手落子后」= 第 idx+1 行的 board（若存在）；否则用最后一行的 board
        if len(df) > idx + 1:
            row = df.iloc[idx + 1]
            prev = np.asarray(df.iloc[idx]['board'])
        else:
            row = df.iloc[idx]
            prev = np.asarray(df.iloc[idx - 1]['board']) if idx >= 1 else None
        board = np.asarray(row['board'])
        player = int(row['player'])
        lm = diff_last_move(prev, board) if prev is not None else None
        vals.append(abs(value_of(board, player, lm)))
    if not vals:
        return {'err': 'no sample'}
    v = np.array(vals)
    return {
        'gen': gen, 'weight_it': weight_it, 'after_move': after_move,
        'n': len(v), 'skipped': skipped,
        'mean': float(v.mean()), 'median': float(np.median(v)),
        'p90': float(np.percentile(v, 90)), 'max': float(v.max()),
        'over': int((v > VTH).sum()), 'over_pct': float(100.0 * (v > VTH).mean()),
        'raw': v.tolist(),
    }


if __name__ == '__main__':
    out = []
    # 对比：代1 是 BAL3_STEP=3 那次跑出来的；代27~31 是 STEP=2
    cases = [
        (1, 0, 3), (1, 0, 4),          # 代1：看第3手后（应不受约束）与第4手后（应受约束）
        (31, 30, 3), (31, 30, 4),      # 当代：第3手后（应受约束）、第4手后（不受约束）
        (30, 29, 3), (29, 28, 3),
        (28, 27, 3), (27, 26, 3),
    ]
    for gen, wit, am in cases:
        r = probe_gen(gen, wit, am, n=60)
        if r is None or 'err' in r:
            print('代%-3d it%-3d 第%d手后  → %s' % (gen, wit, am, r and r.get('err') or '缺数据'))
            continue
        out.append({k: v for k, v in r.items() if k != 'raw'})
        print('代%-3d it%-3d 第%d手后  n=%-3d 均|V|=%.3f 中位=%.3f p90=%.3f max=%.3f  超%.2f: %d/%d (%.1f%%)'
              % (gen, wit, am, r['n'], r['mean'], r['median'], r['p90'], r['max'],
                 VTH, r['over'], r['n'], r['over_pct']))
    json.dump(out, open('/tmp/bal3_check.json', 'w'), ensure_ascii=False, indent=1)
