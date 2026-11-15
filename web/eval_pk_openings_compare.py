#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用【web 测评 API】(常驻客户端, 用户认为稳) 跑 标准开局(H8=7,7) vs 非标开局 对照。
同一权重/同一 depth/sim/温度, 各 N 局 → 供威胁响应率分档比较。
用法: python3 eval_pk_openings_compare.py [每组局数=8] [--smoke]
"""
import json, time, base64, urllib.request, urllib.error, sys, os
from datetime import datetime

API = 'http://127.0.0.1:8765'
PW = os.environ.get('GOMOKU_PASSWORD', '')
AUTH = 'Basic ' + base64.b64encode(('any:' + PW).encode()).decode()

N_PER = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 8
SMOKE = '--smoke' in sys.argv
WEIGHT = 'v39|weights_it153_v39.pt'
SIM = 800
DEPTH = 5
WE_BLACK = True
TEMPERATURE = 0.05

STANDARD = (7, 7)                                   # 天元 H8
NONSTD = [(5, 5), (5, 9), (6, 6), (7, 5), (7, 9), (8, 8), (9, 5), (9, 9),
          (5, 6), (5, 7), (5, 8), (6, 5)]

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'eval_pk_openings_cmp.json')


def call(path, payload=None, method=None, timeout=180):
    url = API + path
    data = None
    headers = {'Authorization': AUTH}
    if payload is not None:
        data = json.dumps(payload).encode()
        headers['Content-Type'] = 'application/json'
    req = urllib.request.Request(url, data=data, headers=headers,
                                 method=(method or ('POST' if data is not None else 'GET')))
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def play_game(idx, open1, group):
    r = call('/api/eval/start', {
        'weight': WEIGHT, 'sim': SIM, 'depth': DEPTH,
        'we_black': WE_BLACK, 'temperature': TEMPERATURE,
        'first_moves': [{'r': open1[0], 'c': open1[1]}],
    })
    if not r.get('ok'):
        return {'game': idx, 'open1': list(open1), 'group': group, 'error': 'start_failed: ' + str(r.get('error')),
                'winner': None, 'us_win': None, 'moves': None, 'engine_error': True, 'secs': 0}
    t0 = time.time()
    state = r['state']
    steps = 0
    while True:
        steps += 1
        if steps > 260:
            state = call('/api/eval/state').get('state', state)
            return {'game': idx, 'open1': list(open1), 'group': group, 'winner': state.get('winner'),
                    'us_win': None, 'moves': state.get('moves'), 'engine_error': True,
                    'secs': round(time.time() - t0, 1), 'error': 'step_limit', 'history': state.get('history')}
        try:
            rr = call('/api/eval/step', {})
        except urllib.error.HTTPError as e:
            return {'game': idx, 'open1': list(open1), 'group': group, 'error': 'http_%s: %s' % (e.code, e.read()[:200]),
                    'winner': None, 'us_win': None, 'moves': None, 'engine_error': True,
                    'secs': round(time.time() - t0, 1)}
        if not rr.get('ok'):
            if '处理中' in str(rr.get('error', '')):
                time.sleep(0.4); steps -= 1; continue
            return {'game': idx, 'open1': list(open1), 'group': group, 'error': 'step_failed: ' + str(rr.get('error')),
                    'winner': None, 'us_win': None, 'moves': None, 'engine_error': True,
                    'secs': round(time.time() - t0, 1)}
        state = rr['state']
        if rr.get('done'):
            break
    winner = state.get('winner')
    us_win = None if winner is None else (winner == 1) == WE_BLACK
    return {'game': idx, 'open1': list(open1), 'group': group, 'winner': winner, 'us_win': us_win,
            'moves': state.get('moves'), 'engine_error': bool(state.get('engine_error')),
            'secs': round(time.time() - t0, 1), 'history': state.get('history')}


def main():
    n_per = 1 if SMOKE else N_PER
    jobs = []
    for i in range(n_per):
        jobs.append(('standard', STANDARD))
        jobs.append(('nonstd', NONSTD[i % len(NONSTD)]))
    print('web 测评 API: %s | %s sim=%d depth=%d temp=%s | 标准%d + 非标%d 局'
          % (API, WEIGHT, SIM, DEPTH, TEMPERATURE, n_per, n_per), flush=True)
    call('/api/eval/reset', {}, method='POST')
    results = []
    for i, (grp, open1) in enumerate(jobs):
        res = play_game(i + 1, open1, grp)
        results.append(res)
        verdict = ('我方胜' if res.get('us_win') is True else
                   'Rapfi胜' if res.get('us_win') is False else
                   ('引擎异常' if res.get('engine_error') else '未决'))
        print('[%2d/%d] %-8s open1=%s %s (%s手) %.1fs%s'
              % (i + 1, len(jobs), grp, open1, verdict, res.get('moves'), res.get('secs') or 0,
                 (' ERR:' + str(res.get('error'))) if res.get('error') else ''), flush=True)
        payload = {'ts': datetime.now().isoformat(), 'api': API, 'weight': WEIGHT,
                   'sim': SIM, 'depth': DEPTH, 'we_black': WE_BLACK, 'temperature': TEMPERATURE,
                   'results': results}
        with open(OUT, 'w') as f:
            json.dump(payload, f, ensure_ascii=False, indent=1)
        if SMOKE:
            break
    print('\n记录文件: ' + OUT, flush=True)


if __name__ == '__main__':
    main()
