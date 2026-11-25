"""用「谁最后落子」这条硬约束核 winner 字段（比判五连更可靠）。

pkl 契约：board 是【落子前】局面，winner 1=黑 / -1=白。
黑先手 ⇒ 终局面里 黑子数 − 白子数 ∈ {0, 1}：
  · 差 0 ⇒ 最后落子的是黑 → winner 应为 1
  · 差 1 ⇒ 最后落子的是白 → winner 应为 −1
（形成五连的那一手不在 pkl 里，所以直接判五连永远是 False，早先那版结论是空的。）

用法: .venv/bin/python scripts/audit_winner_sign2.py <ckpt_dir> [--limit N]
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

DIR = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
    '~/AlphaZero_Gomoku_15x15_v24/gomoku_checkpoints_v24')
LIMIT = None
if '--limit' in sys.argv:
    LIMIT = int(sys.argv[sys.argv.index('--limit') + 1])

files = sorted(glob.glob(os.path.join(DIR, 'worker_data', 'iter*_*.pkl'))) + \
        sorted(glob.glob(os.path.join(DIR, 'game_data', 'iter*_*.pkl')))
if LIMIT:
    files = files[:LIMIT]

print('待检 %d 个 pkl' % len(files))
bad, ok, err = [], 0, 0
for path in files:
    try:
        df = pd.read_pickle(path)
        if not len(df):
            continue
        winner = int(df['winner'].iloc[0])
        final = np.asarray(df['board'].values[-1], dtype=np.int8)
        nb, nw = int((final == 1).sum()), int((final == -1).sum())
        diff = nb - nw
        if diff not in (0, 1):
            bad.append({'file': os.path.basename(path), 'kind': 'stones_diff',
                        'black': nb, 'white': nw, 'winner': winner})
            continue
        expect = 1 if diff == 0 else -1
        if winner != expect:
            bad.append({'file': os.path.basename(path), 'kind': 'winner_vs_last_move',
                        'black': nb, 'white': nw, 'expect_winner': expect,
                        'winner_field': winner, 'rows': int(len(df))})
        else:
            ok += 1
    except Exception as e:
        err += 1
        if err <= 3:
            print('  read fail', os.path.basename(path), e)

print('自洽 %d 局 · 异常 %d 局 · 读取失败 %d' % (ok, len(bad), err))
for x in bad[:15]:
    print('   ', x)
if bad:
    json.dump(bad, open('/tmp/winner_mismatch.json', 'w'), ensure_ascii=False, indent=1)
    print('明细 → /tmp/winner_mismatch.json')
