"""核对 pkl 里的 winner 字段 与 棋盘真实胜负 是否一致。

背景：用户看到「某局白棋明显赢了，但胜率曲线记成黑胜」。
本脚本把每局棋最后局面重放，自己判五连，再和 df['winner'] 比对。

用法：
  .venv/bin/python scripts/audit_winner_sign.py <ckpt_dir> [--limit N]
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

DIR = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser('~/AlphaZero_Gomoku_15x15_v24/gomoku_checkpoints_v24')
LIMIT = None
if '--limit' in sys.argv:
    LIMIT = int(sys.argv[sys.argv.index('--limit') + 1])


def has_five(b, v):
    """b: 15x15 int8, v=1 黑 / -1 白。返回是否五连。"""
    m = (b == v)
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        # 用卷积式滑窗判断
        for r in range(15):
            for c in range(15):
                if not m[r, c]:
                    continue
                rr, cc = r, c
                ok = True
                for _ in range(4):
                    rr += dr
                    cc += dc
                    if not (0 <= rr < 15 and 0 <= cc < 15) or not m[rr, cc]:
                        ok = False
                        break
                if ok:
                    return True
    return False


files = sorted(glob.glob(os.path.join(DIR, 'worker_data', 'iter*_game_*.pkl'))) + \
        sorted(glob.glob(os.path.join(DIR, 'game_data', 'iter*_game_*.pkl')))
lone_data = sorted(glob.glob(os.path.join(DIR, 'worker_data', 'iter*_*.pkl')))
if not files:
    files = lone_data
if LIMIT:
    files = files[:LIMIT]

print('待检 %d 个 pkl' % len(files))
bad, n5_black, n5_white, nofive, err = [], 0, 0, 0, 0
for path in files:
    try:
        df = pd.read_pickle(path)
        if not len(df):
            continue
        winner = int(df['winner'].iloc[0])
        final = np.asarray(df['board'].values[-1], dtype=np.int8)
        fb, fw = has_five(final, 1), has_five(final, -1)
        if fb:
            n5_black += 1
        if fw:
            n5_white += 1
        if not fb and not fw:
            nofive += 1
            continue
        # 最后一行 board 是「最后一手之前」的局面 ⇒ 五连还应由最后一手补全
        true_w = 1 if fb else -1
        if winner != true_w:
            bad.append({'file': os.path.basename(path), 'winner_field': winner,
                        'five_black': bool(fb), 'five_white': bool(fw),
                        'rows': int(len(df))})
    except Exception as e:
        err += 1

print('末局面已有五连: 黑 %d 局 / 白 %d 局 · 无五连 %d 局 · 读取失败 %d' %
      (n5_black, n5_white, nofive, err))
print('★ winner 字段与末局面五连不一致: %d 局' % len(bad))
for x in bad[:15]:
    print('   ', x)
json.dump(bad, open('/tmp/winner_mismatch.json', 'w'), ensure_ascii=False, indent=1)
