#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""通过【本地 web 棋力测评 API】跑 it153(sim800,执黑) vs Rapfi depth5, 25 局。

★ 走 web HTTP API(http://127.0.0.1:8765/api/eval/{start,step,state}), 不用 scripts/rapfi_bench.py
首手控制: first_moves=[{r,c}](对应 bench 的 --vopen / OPEN1_POINTS 25 点, 天元 5x5 中心区)
"""
import json, time, base64, urllib.request, urllib.error, sys, os
from datetime import datetime

API = 'http://127.0.0.1:8765'
PW = os.environ.get('GOMOKU_PASSWORD', '')
AUTH = 'Basic ' + base64.b64encode(('any:' + PW).encode()).decode()

# 与 scripts/rapfi_bench.py 的 OPEN1_POINTS 完全一致: 天元(7,7) 5x5 中心区 r,c in [5,9]
OPEN1_POINTS = [(r, c) for r in range(5, 10) for c in range(5, 10)]
N_GAMES = 25
WEIGHT = 'v39|weights_it153_v39.pt'
SIM = 800
DEPTH = 5
WE_BLACK = True
TEMPERATURE = 0.05   # web 默认 (<=0.05 即 argmax/贪心, 与 bench clean search 口径一致)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   'eval_pk_it153_sim800_black_vs_rapfi_d5_25g.json')


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


def play_game(idx, open1):
    """返回一局结果 dict"""
    r = call('/api/eval/start', {
        'weight': WEIGHT, 'sim': SIM, 'depth': DEPTH,
        'we_black': WE_BLACK, 'temperature': TEMPERATURE,
        'first_moves': [{'r': open1[0], 'c': open1[1]}],
    })
    if not r.get('ok'):
        return {'game': idx, 'open1': list(open1), 'error': 'start_failed: ' + str(r.get('error')),
                'winner': None, 'us_win': None, 'moves': None, 'engine_error': True, 'secs': 0}
    t0 = time.time()
    state = r['state']
    steps = 0
    detail = []      # 每步: {who, r, c, secs} —— 用于区分我方/Rapfi 的用时
    while True:
        steps += 1
        if steps > 260:
            state = call('/api/eval/state').get('state', state)
            return {'game': idx, 'open1': list(open1), 'winner': state.get('winner'),
                    'us_win': None, 'moves': state.get('moves'), 'engine_error': True,
                    'secs': round(time.time() - t0, 1), 'error': 'step_limit', 'detail': detail}
        ts = time.time()
        try:
            rr = call('/api/eval/step', {})
        except urllib.error.HTTPError as e:
            return {'game': idx, 'open1': list(open1), 'error': 'http_%s: %s' % (e.code, e.read()[:200]),
                    'winner': None, 'us_win': None, 'moves': None, 'engine_error': True,
                    'secs': round(time.time() - t0, 1), 'detail': detail}
        if not rr.get('ok'):
            if '处理中' in str(rr.get('error', '')):
                time.sleep(0.4)
                steps -= 1      # busy 不算一步
                continue
            return {'game': idx, 'open1': list(open1), 'error': 'step_failed: ' + str(rr.get('error')),
                    'winner': None, 'us_win': None, 'moves': None, 'engine_error': True,
                    'secs': round(time.time() - t0, 1), 'detail': detail}
        dt = time.time() - ts
        state = rr['state']
        hist = state.get('history') or []
        last = hist[-1] if hist else None
        if last:
            detail.append({'who': last.get('who'), 'r': last.get('r'), 'c': last.get('c'),
                           'secs': round(dt, 2)})
        if rr.get('done'):
            break
    winner = state.get('winner')
    engine_error = bool(state.get('engine_error'))
    us_win = None if winner is None else (winner == 1) == WE_BLACK
    return {'game': idx, 'open1': list(open1), 'winner': winner, 'us_win': us_win,
            'moves': state.get('moves'), 'engine_error': engine_error,
            'secs': round(time.time() - t0, 1),
            'history': state.get('history'), 'detail': detail}


def main():
    print('web 测评 API: %s | 权重=%s sim=%d depth=%d we_black=%s temp=%s | %d 局 | 首手=%d 点'
          % (API, WEIGHT, SIM, DEPTH, WE_BLACK, TEMPERATURE, N_GAMES, len(OPEN1_POINTS)), flush=True)
    call('/api/eval/reset', {}, method='POST')   # 清理旧状态
    results = []
    for i in range(N_GAMES):
        open1 = OPEN1_POINTS[i % len(OPEN1_POINTS)]
        res = play_game(i + 1, open1)
        results.append(res)
        w = res.get('winner')
        verdict = ('我方胜' if res.get('us_win') is True else
                   'Rapfi胜' if res.get('us_win') is False else
                   ('引擎异常' if res.get('engine_error') else '平局/未决'))
        print('[%2d/%d] open1=%s %s (%s手) %.1fs%s'
              % (i + 1, N_GAMES, open1, verdict, res.get('moves'), res.get('secs') or 0,
                 (' ERR:' + str(res.get('error'))) if res.get('error') else ''), flush=True)
        # 每局增量落盘
        payload = {'ts': datetime.now().isoformat(), 'api': API, 'weight': WEIGHT,
                   'sim': SIM, 'depth': DEPTH, 'we_black': WE_BLACK, 'temperature': TEMPERATURE,
                   'open1_points': OPEN1_POINTS, 'n_games': N_GAMES, 'results': results}
        with open(OUT, 'w') as f:
            json.dump(payload, f, ensure_ascii=False, indent=1)
    wins = sum(1 for r in results if r.get('us_win') is True)
    losses = sum(1 for r in results if r.get('us_win') is False)
    errs = sum(1 for r in results if r.get('engine_error'))
    moves = [r['moves'] for r in results if r.get('moves')]
    print('\n=== 汇总 ===', flush=True)
    print('我方胜率 %.1f%% (%d/%d) | 负 %d | 异常 %d | 均手 %.1f | 总耗时 %.1fs'
          % (wins / N_GAMES * 100, wins, N_GAMES, losses, errs,
             (sum(moves) / len(moves) if moves else 0),
             sum(r.get('secs') or 0 for r in results)), flush=True)
    print('记录文件: ' + OUT, flush=True)


if __name__ == '__main__':
    main()
