#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""历代权重在「题目:看不见黑活三,3」的 K10 选点概率趋势

题目局面：黑 (6,9)(7,8)(8,7) 斜线活三，答案 = (5,10) 与 (9,6) 两个堵点
页面坐标 K10 = LETTERS[10] + (15-r)=10 → c=10, r=5 → 即 (5,10)
"""
import importlib.util
import json
import os
import re
import sys
import warnings

import numpy as np
import torch

warnings.filterwarnings('ignore')
PROJ = os.path.expanduser('~/AlphaZero_Gomoku_15x15_v24')
spec = importlib.util.spec_from_file_location('m', f'{PROJ}/Gomoku-v0_pt_v24b.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
import gym
env = gym.make('Gomuku-v0')

KW = {'conv_filters': [128], 'residual_filters': [[128, 128]] * 8, 'policy_filters': [128]}
ag = m.AlphaZeroAgent(env=env, kwargs=KW, sim_count=1, batches=1, batch_size=8)
net = ag.net
net.eval()

POS_NAME = sys.argv[1] if len(sys.argv) > 1 else '题目:看不见黑活三,3'
# 关注点：默认 K10 = (5,10)；可用 `--pts 5,10 9,6` 换成任意两点
_pts = sys.argv[2] if len(sys.argv) > 2 else '5,10|9,6'
P1 = tuple(int(v) for v in _pts.split('|')[0].split(','))
P2 = tuple(int(v) for v in _pts.split('|')[1].split(','))

POS = [p for p in json.load(open(f'{PROJ}/web/positions.json'))
       if p['name'] == POS_NAME][0]
stones = POS['stones']
board = np.zeros((15, 15), dtype=np.int8)
for s in stones:
    board[s['r'], s['c']] = s['v']
nb, nw = int((board == 1).sum()), int((board == -1).sum())
player = 1 if nb == nw else -1
last = (stones[-1]['r'], stones[-1]['c'])
canon = (player * board).astype(np.float32)
x = torch.tensor(m.make_input(canon, last_move=last, player=player)).unsqueeze(0)

ANS = [P1, P2]

print('局面: 黑%d 白%d → 轮到%s下 | plane3=%s' % (nb, nw, '黑' if player == 1 else '白', last))
print('关注点: P1=%s  P2=%s\n' % (P1, P2))

rows = []
ck = f'{PROJ}/gomoku_checkpoints_v24'
files = [f for f in os.listdir(ck) if re.match(r'weights_it\d+_v24\.pt$', f)]
files.sort(key=lambda f: int(f.split('it')[1].split('_')[0]))

for f in files:
    it = int(f.split('it')[1].split('_')[0])
    sd = torch.load(os.path.join(ck, f), map_location='cpu')
    net.load_state_dict(sd)
    net.eval()
    with torch.no_grad():
        p_log, v = net(x)
    p = torch.exp(p_log).reshape(15, 15).numpy()
    p = p / p.sum()
    rows.append({
        'it': it,
        'k10': float(p[P1]),
        'g6': float(p[P2]),
        'both': float(p[P1] + p[P2]),
        'value': float(v.reshape(-1)[0]),
        'entropy': float(-(p * np.log(p + 1e-12)).sum()),
        'rank_k10': int((p > p[P1]).sum()) + 1,
    })
    if len(rows) % 20 == 0:
        print('  ... 已算 %d/%d' % (len(rows), len(files)))

out = f'{PROJ}/web/k10_trend.json'
json.dump(rows, open(out, 'w'), ensure_ascii=False, indent=1)
print('\n共 %d 代 -> %s' % (len(rows), out))
print('\n%-6s %-9s %-9s %-10s %-9s %s' % ('代', 'K10概率', 'G6概率', '两答案合计', 'K10名次', 'value'))
for r in rows:
    print('%-6d %-9.2f%% %-9.2f%% %-10.2f%% %-9d %+.3f' % (
        r['it'], r['k10'] * 100, r['g6'] * 100, r['both'] * 100, r['rank_k10'], r['value']))
