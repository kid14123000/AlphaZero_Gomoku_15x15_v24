#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""活三诊断：位置里到底有没有活三、网络有没有看见

判活三用「字符串窗口匹配」，不用点数枚举（枚举容易把已成的活四当成活三点）：
  一条线转成字符串，1=该方  0=空  2=对方/越界
  活三 = 两端为空的三连，含连三与断三：01110 / 011010 / 010110
  活四 = 011110
  冲四 = 21110 / 01112 / 11011 / 10111 / 11101

用法:
  python probe_open_three.py               # it88 + 400 sims
  python probe_open_three.py --sims 0      # 只跑策略头
"""
import argparse
import importlib.util
import json
import os

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
W = H = 15
PAT3 = ('01110', '011010', '010110')
PAT4 = ('011110',)
PAT_R4 = ('21110', '01112', '11011', '10111', '11101')


def load_mod():
    p = os.path.join(HERE, "Gomoku-v0_pt_v24b.py")
    spec = importlib.util.spec_from_file_location("gomoku_v24b", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def lines_of(g, p):
    out = []

    def enc(rc):
        v = g.get(rc)
        if v is None:
            return '0'
        return '1' if v == p else '2'

    for r in range(H):
        cells = [(r, c) for c in range(W)]
        out.append(('R%d' % r, '2' + ''.join(enc(x) for x in cells) + '2', cells))
    for c in range(W):
        cells = [(r, c) for r in range(H)]
        out.append(('C%d' % c, '2' + ''.join(enc(x) for x in cells) + '2', cells))
    for k in range(-(H - 1), W):
        cells = [(r, r - k) for r in range(H) if 0 <= r - k < W]
        if len(cells) >= 5:
            out.append(('D%d' % k, '2' + ''.join(enc(x) for x in cells) + '2', cells))
    for k in range(W + H - 1):
        cells = [(r, k - r) for r in range(H) if 0 <= k - r < W]
        if len(cells) >= 5:
            out.append(('A%d' % k, '2' + ''.join(enc(x) for x in cells) + '2', cells))
    return out


def find(g, p):
    o3, o4, r4 = [], [], []
    for name, s, cells in lines_of(g, p):
        for pats, box in ((PAT3, o3), (PAT4, o4), (PAT_R4, r4)):
            for patn in pats:
                st = 0
                while True:
                    i = s.find(patn, st)
                    if i < 0:
                        break
                    holes = []
                    for j, ch in enumerate(patn):
                        if ch == '0' and 0 <= i + j - 1 < len(cells):
                            holes.append(cells[i + j - 1])
                    if holes:
                        box.append((name, holes, patn))
                    st = i + 1
    return o3, o4, r4


def fmt(rc):
    return '%c%d' % (ord('A') + rc[1], H - rc[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pos', default=os.path.join(HERE, 'web/positions.json'))
    ap.add_argument('--ckpt', default=os.path.join(HERE, 'gomoku_checkpoints_v24/weights_it88_v24.pt'))
    ap.add_argument('--sims', type=int, default=400)
    a = ap.parse_args()

    M = load_mod()
    net = M.AlphaZeroNet(128, 8, 15)
    sd = torch.load(a.ckpt, map_location='cpu')
    if isinstance(sd, dict) and 'state_dict' in sd:
        sd = sd['state_dict']
    net.load_state_dict(sd, strict=False)
    net.eval()

    agent = None
    if a.sims > 0:
        import gym
        env = gym.make('Gomuku-v0')
        agent = M.AlphaZeroAgent(env=env,
                                 kwargs={'conv_filters': [128],
                                         'residual_filters': [[128, 128]] * 8,
                                         'policy_filters': [128]},
                                 batches=1, batch_size=1, sim_count=a.sims)
        agent.net.load_state_dict(sd, strict=False)
        agent.net.eval()

    print('权重 %s\n' % os.path.basename(a.ckpt))
    for it in json.load(open(a.pos)):
        if '活三' not in (it.get('name') or ''):
            continue
        stones = it.get('stones') or []
        if not stones:
            continue
        g = {(s['r'], s['c']): (1 if s['v'] == 1 else -1) for s in stones}
        n = len(stones)
        me = 1 if n % 2 == 0 else -1
        opp = -me
        lm = (stones[-1]['r'], stones[-1]['c'])

        print('=' * 78)
        print('%-20s %2d 子  轮到 %s  末手 %s'
              % (it['name'], n, '黑' if me == 1 else '白', fmt(lm)))
        for r in range(H):
            print('   ' + ''.join(' X' if g.get((r, c)) == 1 else
                                  (' O' if g.get((r, c)) == -1 else ' .') for c in range(W)))

        for who, lab in ((me, '我方'), (opp, '对方')):
            o3, o4, r4 = find(g, who)
            print('  %s(%s) 活四%d 冲四%d 活三%d  %s'
                  % (lab, '黑' if who == 1 else '白', len(o4), len(r4), len(o3),
                     ' | '.join('%s[%s]' % (nm, ','.join(fmt(x) for x in holes))
                                for nm, holes, _ in o3[:4])))

        canon = np.zeros((H, W), dtype=np.float32)
        for (r, c), v in g.items():
            canon[r, c] = 1.0 if v == me else -1.0
        x = torch.tensor(M.make_input(canon, last_move=lm, player=me)).unsqueeze(0)
        with torch.no_grad():
            logp, val = net(x)
        prob = torch.exp(logp).numpy().reshape(-1)
        order = np.argsort(prob)[::-1]
        print('  value(%s)=%+.4f  top6: %s'
              % ('黑' if me == 1 else '白', float(np.asarray(val).reshape(-1)[0]),
                 ' '.join('%s:%.2f' % (fmt((int(m) // W, int(m) % W)), prob[m])
                          for m in order[:6])))
        rank = {(int(m) // W, int(m) % W): i + 1 for i, m in enumerate(order)}

        mine3, _, _ = find(g, me)
        opp3, _, _ = find(g, opp)
        keys = []
        for nm, holes, _ in mine3[:2]:
            keys += [('我方活三扩展', h) for h in holes]
        for nm, holes, _ in opp3[:3]:
            keys += [('对方活三封堵', h) for h in holes]
        for lab, h in keys[:10]:
            print('    %-12s %-4s → policy 第 %3d 名 p=%.4f'
                  % (lab, fmt(h), rank.get(h, 999), prob[h[0] * W + h[1]]))

        if agent is not None:
            board = np.array([[g.get((r, c), 0) for c in range(W)] for r in range(H)],
                             dtype=np.int8)
            try:
                loc, _ = agent.decide((board, me), greedy=True, return_prob=True,
                                      noise=False, move_no=n, last_move=lm)
                mv = (int(loc[0]), int(loc[1]))
                tgt = {h for _, h in keys}
                print('  MCTS %d sims → %s %s'
                      % (a.sims, fmt(mv), '✓ 命中关键点' if mv in tgt else '✗ 未命中'))
            except Exception as e:                    # noqa: BLE001
                print('  MCTS 失败: %s' % str(e)[:90])
            agent.reset_mcts()
        print()


if __name__ == '__main__':
    main()
