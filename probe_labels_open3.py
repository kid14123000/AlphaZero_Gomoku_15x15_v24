#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""统计自对弈数据里「活三局面」的标签质量

对每个 pkl 局面：
  ① 对方有活三 → 我方必须封堵（封堵点 = 活三的扩展点）
     统计 π 给这些点多少概率质量。质量低 ⇒ 训练标签本身就"看不见活三"，
     网络学成这样是必然结果，不是网络结构问题。
  ② 我方有活三 → 应该做活四，同样统计
  ③ 顺便统计胜负分布、局面里活三出现率

用法: python probe_labels_open3.py --n 200
"""
import argparse
import glob
import os
import pickle
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
W = H = 15
PAT3 = ('01110', '011010', '010110')
PAT4 = ('011110',)


def lines_of(g, p):
    def enc(rc):
        v = g.get(rc)
        if v is None:
            return '0'
        return '1' if v == p else '2'

    out = []
    for r in range(H):
        cells = [(r, c) for c in range(W)]
        out.append('2' + ''.join(enc(x) for x in cells) + '2')
        out.append(cells)
    for c in range(W):
        cells = [(r, c) for r in range(H)]
        out.append('2' + ''.join(enc(x) for x in cells) + '2')
        out.append(cells)
    for k in range(-(H - 1), W):
        cells = [(r, r - k) for r in range(H) if 0 <= r - k < W]
        if len(cells) >= 5:
            out.append('2' + ''.join(enc(x) for x in cells) + '2')
            out.append(cells)
    for k in range(W + H - 1):
        cells = [(r, k - r) for r in range(H) if 0 <= k - r < W]
        if len(cells) >= 5:
            out.append('2' + ''.join(enc(x) for x in cells) + '2')
            out.append(cells)
    return [(out[i], out[i + 1]) for i in range(0, len(out), 2)]


def find(g, p):
    """返回活三的空点集合 / 活四的空点集合"""
    o3, o4 = set(), set()
    for s, cells in lines_of(g, p):
        for pats, box in ((PAT3, o3), (PAT4, o4)):
            for patn in pats:
                st = 0
                while True:
                    i = s.find(patn, st)
                    if i < 0:
                        break
                    for j, ch in enumerate(patn):
                        if ch == '0' and 0 <= i + j - 1 < len(cells):
                            box.add(cells[i + j - 1])
                    st = i + 1
    return o3, o4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default=os.path.join(HERE, 'gomoku_checkpoints_v24/worker_data'))
    ap.add_argument('-n', '--n', type=int, default=200)
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.dir, '*.pkl')))[:a.n]
    print('扫 %d 个 pkl' % len(files))
    n_pos = 0
    n_opp3 = 0            # 对方有活三的局面
    mass_opp3 = []        # 这些局面 π 给"封堵点"的质量
    n_own3 = 0
    mass_own3 = []
    n_opp4 = 0
    win = {1: 0, -1: 0, 0: 0}
    per_file_win = {}

    for f in files:
        try:
            df = pickle.load(open(f, 'rb'))
        except Exception:                      # noqa: BLE001
            continue
        if not isinstance(df, pd.DataFrame) or 'prob' not in df.columns:
            continue
        w = int(df['winner'].iloc[0]) if len(df) else 0
        win[1 if w > 0 else (-1 if w < 0 else 0)] += 1
        for _, row in df.iterrows():
            bd = np.asarray(row['board'])
            pi = np.asarray(row['prob'], dtype=np.float64)
            me = int(row['player'])
            g = {}
            for r in range(H):
                for c in range(W):
                    v = bd[r, c]
                    if v != 0:
                        g[(r, c)] = me if v == 1 else -me
            n_pos += 1
            opp3, opp4 = find(g, -me)
            if opp4:
                n_opp4 += 1
            if opp3:
                n_opp3 += 1
                mass_opp3.append(float(sum(pi[r, c] for r, c in opp3)))
            my3, my4 = find(g, me)
            if my3:
                n_own3 += 1
                mass_own3.append(float(sum(pi[r, c] for r, c in my3)))

    print('\n局面总数 %d' % n_pos)
    print('对局胜负: 黑胜 %d / 白胜 %d / 其它 %d' % (win[1], win[-1], win[0]))
    print('对方有活四的局面: %d (%.2f%%)' % (n_opp4, 100.0 * n_opp4 / max(n_pos, 1)))
    print('对方有活三的局面: %d (%.1f%%)' % (n_opp3, 100.0 * n_opp3 / max(n_pos, 1)))
    if mass_opp3:
        m = np.array(mass_opp3)
        print('   π 给封堵点的概率质量: 均值 %.3f 中位 %.3f  <0.1 的占 %.0f%%  ≥0.5 的占 %.0f%%'
              % (m.mean(), np.median(m), 100.0 * (m < 0.1).mean(), 100.0 * (m >= 0.5).mean()))
    print('我方有活三的局面: %d (%.1f%%)' % (n_own3, 100.0 * n_own3 / max(n_pos, 1)))
    if mass_own3:
        m = np.array(mass_own3)
        print('   π 给活三扩展点的质量: 均值 %.3f 中位 %.3f  <0.1 的占 %.0f%%  ≥0.5 的占 %.0f%%'
              % (m.mean(), np.median(m), 100.0 * (m < 0.1).mean(), 100.0 * (m >= 0.5).mean()))


if __name__ == '__main__':
    sys.exit(main())
