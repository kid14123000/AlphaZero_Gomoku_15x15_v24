#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""probe_planes.py —— 审计 web 各页面推理时喂给网络的 4 通道语义是否一致

关键：MCTS 搜索**不走** make_input（AlphaZeroAgent.predict_inference 里内联拼 4 plane），
所以必须同时挂钩 predict_inference 和 make_input，再加 Net.forward 作兜底，
否则会误报"该页面没有推理调用"。

参考口径（训练脚本 Gomoku-v0_pt_v24*.py）：
  ch0 = 当前玩家子 (canonical>0)   ch1 = 对手子 (canonical<0)
  ch2 = 恒 1.0 (v24 起无执色信息)   ch3 = 上一手 one-hot（无则全 0）
"""
import json
import os
import sys
import time
import traceback

import numpy as np

WEB = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, WEB)

WV = 'v24b'
WF = 'weights_it62_v24.pt'
W = '%s|%s' % (WV, WF)
STONES = [{'r': 7, 'c': 7, 'v': 1}, {'r': 7, 'c': 8, 'v': -1},
          {'r': 8, 'c': 8, 'v': 1}, {'r': 6, 'c': 6, 'v': -1}]
LAST_STONE = (6, 6)          # STONES 顺序里最后一手

PAGE = {'name': '?'}
CAPS = []


def _caller():
    f = sys._getframe()
    while f is not None:
        if f.f_code.co_filename.endswith('/app.py'):
            return '%s:%d' % (f.f_code.co_name, f.f_lineno)
        f = f.f_back
    return '?'


def _summ(inp):
    """inp: (N,4,15,15) numpy → 每通道摘要（取第一个样本）"""
    a = np.asarray(inp)
    if a.ndim == 4:
        a = a[0]
    return {
        'ch': [int((a[i] != 0).sum()) for i in range(a.shape[0])],
        'ch2': np.unique(np.round(a[2], 3)).tolist()[:4],
        'ch3_idx': [list(map(int, x)) for x in np.argwhere(a[3] != 0).tolist()[:2]],
        'ch_overlap': int(((a[0] != 0) & (a[1] != 0)).sum()),   # ch0/ch1 同时亮 = 语义错
    }


def rec(kind, where, info):
    info.update({'page': PAGE['name'], 'kind': kind, 'where': where})
    CAPS.append(info)


def install(ver, mod):
    orig_pi = mod.AlphaZeroAgent.predict_inference

    def pi(self, board_batch, last_moves=None, players=None):
        try:
            b = np.asarray(board_batch)
            lm0 = None
            if last_moves is not None:
                lms = np.asarray(last_moves, dtype=object)
                lm0 = lms[0] if lms.ndim > 0 else lms
                lm0 = None if lm0 is None else list(map(int, np.asarray(lm0).reshape(-1)[:2]))
            # 复算该调用真正喂进 net 的 4 plane（与 predict_inference 同语义）
            s = np.shape(b)
            b3 = b if b.ndim == 3 else b[None]
            inp = np.zeros((b3.shape[0], 4) + b3.shape[1:], dtype=np.float32)
            inp[:, 0] = b3 > 0
            inp[:, 1] = b3 < 0
            inp[:, 2] = 1.0
            if last_moves is not None:
                for i, lm in enumerate(np.asarray(last_moves, dtype=object)):
                    if lm is not None:
                        v = np.asarray(lm).reshape(-1)
                        inp[i, 3, int(v[0]), int(v[1])] = 1.0
            info = _summ(inp)
            info.update({'last_move': lm0,
                         'players': (None if players is None else
                                     (list(np.asarray(players).reshape(-1)[:3]) or None)),
                         'batch': int(b3.shape[0])})
            rec('predict_inference', _caller(), info)
        except Exception:
            traceback.print_exc()
        return orig_pi(self, board_batch, last_moves, players)

    mod.AlphaZeroAgent.predict_inference = pi

    if hasattr(mod, 'make_input'):
        orig_mi = mod.make_input

        def mi(*a, **kw):
            out = orig_mi(*a, **kw)
            try:
                canon = np.asarray(a[0] if a else kw.get('canonical'))
                if canon.ndim == 3:
                    canon = canon[0]
                lm = a[1] if len(a) > 1 else kw.get('last_move')
                pl = a[2] if len(a) > 2 else kw.get('player')
                info = _summ(np.asarray(out))
                info.update({'last_move': None if lm is None else list(map(int, np.asarray(lm).reshape(-1)[:2])),
                             'players': (None if pl is None else [int(pl)]),
                             'batch': 1, 'canon_pos': int((canon > 0).sum())})
                rec('make_input', _caller(), info)
            except Exception:
                traceback.print_exc()
            return out

        mod.make_input = mi


def main():
    t0 = time.time()
    import app
    print('import app.py %.1fs · TRAIN_VERSION=%s' % (time.time() - t0,
                                                     getattr(app, 'TRAIN_VERSION', '?')))
    hooked = []
    for ver, _d, _s, _n in app.VERSIONS:
        mod = app.load_mod_for(ver)
        install(ver, mod)
        hooked.append(ver)
    print('  已挂钩:', hooked)

    c = app.app.test_client()
    probes = []

    def run(name, url, body):
        PAGE['name'] = name
        CAPS.clear()
        t = time.time()
        try:
            r = c.post(url, json=body)
            st, j = r.status_code, (r.get_json(silent=True) or {})
            ok, err = j.get('ok', 'n/a'), (j.get('error') or j.get('msg'))
        except Exception as e:
            st, ok, err = -1, False, repr(e)
        probes.append({'name': name, 'url': url, 'status': st, 'ok': ok, 'err': err,
                       'sec': round(time.time() - t, 1), 'ncalls': len(CAPS),
                       'calls': [dict(x) for x in CAPS[:3]]})
        print('%-20s %-24s st=%-3s ok=%-5s calls=%-4d %5.1fs %s'
              % (name, url, st, ok, len(CAPS), time.time() - t, str(err)[:45] if err else ''))
        PAGE['name'] = '?'

    run('网络前向', '/api/network_viz', {'weight': W, 'stones': STONES})
    run('权重可视化', '/api/weight_viz', {'weight': W})
    run('MCTS可视化', '/api/mcts_viz', {'weight': W, 'stones': STONES, 'sim': 20})
    run('摆谱研究', '/api/study/analyze',
        {'weight': W, 'stones': [{**s, 'p': s['v']} for s in STONES], 'sim_count': 20})
    run('价值热力图/回放重算', '/api/analysis/probe', {'weight': W, 'stones': STONES, 'sim': 20})
    run('出题quiz', '/api/quiz/run', {'weight': W, 'mode': 'policy', 'sims': [40]})
    run('开局分布', '/api/opening_dist', {'version': 'v24b'})
    run('对弈start', '/api/start',
        {'black_type': 'ai', 'black_weight': W, 'white_type': 'ai', 'white_weight': W,
         'sim_count': 20})
    run('对弈play', '/api/play', {'r': 7, 'c': 7})

    json.dump(probes, open(os.path.join(WEB, 'probe_planes_out.json'), 'w'),
              ensure_ascii=False, indent=1, default=str)

    print('\n' + '=' * 112)
    print('%-20s %-22s %-18s %-6s %-11s %-13s %s'
          % ('页面', '入口(调用点)', '钩子', 'ch0/1', 'ch2', 'ch3', 'last_move/players'))
    for p in probes:
        if not p['calls']:
            print('%-20s %-22s %s' % (p['name'], '—无推理—', (p['err'] or p['status'])))
            continue
        for x in p['calls'][:1]:
            print('%-20s %-22s %-18s %-6s %-11s %-13s %s'
                  % (p['name'], x['where'], x['kind'],
                     '%d/%d' % (x['ch'][0], x['ch'][1]),
                     ','.join('%.2f' % v for v in x['ch2'][:2]),
                     str(x['ch3_idx'] if x['ch3_idx'] else '全0'),
                     '%s / %s' % (x['last_move'], x['players'])))
    print('=' * 112)
    print('期望：本次摆谱 2黑2白 → 轮黑(player=1) → ch0=2(黑子) ch1=2(白子) ch2=1.00 ch3=[[%d,%d]]'
          % LAST_STONE)


if __name__ == '__main__':
    sys.exit(main())
