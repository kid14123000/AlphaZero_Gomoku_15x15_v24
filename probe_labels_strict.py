#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""严格口径：对方有活三、我方没有立刻成五的手段 ⇒ 我方「必须」封堵

只在满足这个条件时统计 π 给封堵点的概率质量。这是「看不见活三」的定义性检验：
如果在这种局面下 π 也不给封堵点质量，那训练标签本身就是瞎的，
网络学成「不封堵」是必然结果。

用法: python probe_labels_strict.py -n 300
"""
import argparse
import glob
import os
import pickle

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
W = H = 15
PAT3 = ('01110', '011010', '010110')
PAT4 = ('011110',)


def _cells():
    out = []
    for r in range(H):
        out.append(([(r, c) for c in range(W)], 'R'))
    for c in range(W):
        out.append(([(r, c) for r in range(H)], 'C'))
    for k in range(-(H - 1), W):
        cs = [(r, r - k) for r in range(H) if 0 <= r - k < W]
        if len(cs) >= 5:
            out.append((cs, 'D'))
    for k in range(W + H - 1):
        cs = [(r, k - r) for r in range(H) if 0 <= k - r < W]
        if len(cs) >= 5:
            out.append((cs, 'A'))
    return out


LINE_CELLS = _cells()


def scan(g, p, pats):
    """返回这些 pattern 命中的空点集合"""
    hits = set()
    for cells, _ in LINE_CELLS:
        s = '2' + ''.join('0' if (x not in g) else ('1' if g[x] == p else '2')
                          for x in cells) + '2'
        for patn in pats:
            st = 0
            while True:
                i = s.find(patn, st)
                if i < 0:
                    break
                for j, ch in enumerate(patn):
                    if ch == '0' and 0 <= i + j - 1 < len(cells):
                        hits.add(cells[i + j - 1])
                st = i + 1
    return hits


def five_moves(g, p):
    """我方能否一手成五：再落一子即五连的空点"""
    out = set()
    for cells, _ in LINE_CELLS:
        s = '2' + ''.join('0' if (x not in g) else ('1' if g[x] == p else '2')
                          for x in cells) + '2'
        # 把每个 0 当成 1，看是否出现 11111
        for i, ch in enumerate(s):
            if ch != '0':
                continue
            t = s[:i] + '1' + s[i + 1:]
            if '11111' in t:
                out.add(cells[i - 1])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default=os.path.join(HERE, 'gomoku_checkpoints_v24/worker_data'))
    ap.add_argument('-n', '--n', type=int, default=300)
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.dir, '*.pkl')))[:a.n]
    n_pos = n_must = 0
    masses = []
    top1_hit = 0
    win = {1: 0, -1: 0}
    for f in files:
        try:
            df = pickle.load(open(f, 'rb'))
        except Exception:                    # noqa: BLE001
            continue
        if not isinstance(df, pd.DataFrame) or 'prob' not in df.columns:
            continue
        w = int(df['winner'].iloc[0])
        win[1 if w > 0 else -1] += 1
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
            if five_moves(g, me):
                continue                     # 我方有杀，不必封堵
            opp3 = scan(g, -me, PAT3)
            if not opp3:
                continue
            n_must += 1
            m_ = float(sum(pi[r, c] for r, c in opp3))
            masses.append(m_)
            if opp3:
                best = max(opp3, key=lambda rc: pi[rc[0], rc[1]])
                if pi[best[0], best[1]] >= max(np.max(pi) * 0.99, 1e-9):
                    top1_hit += 1

    m = np.array(masses) if masses else np.zeros(1)
    print('扫 %d pkl  局面 %d' % (len(files), n_pos))
    print('对局: 黑胜 %d / 白胜 %d' % (win[1], win[-1]))
    print('\n「对方活三 + 我方无杀」= 必须封堵的局面: %d (%.1f%% of 全部局面)'
          % (n_must, 100.0 * n_must / max(n_pos, 1)))
    if n_must:
        print('  π 给封堵点的质量: 均值 %.3f  中位 %.3f' % (m.mean(), np.median(m)))
        for th in (0.05, 0.1, 0.3, 0.5, 0.8):
            print('    <%.2f 的占 %5.1f%%' % (th, 100.0 * (m < th).mean()))
        print('  π 的 top1 落在封堵点上: %.1f%%' % (100.0 * top1_hit / n_must))


if __name__ == '__main__':
    main()
