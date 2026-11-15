#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用【真 web 测评 API】复跑 open1=(5,6) 那局, 看第14手 Rapfi 落哪 (web 批次里是 L6)。"""
import json, base64, urllib.request, os, time
API = 'http://127.0.0.1:8765'
PW = os.environ.get('GOMOKU_PASSWORD', '')
AUTH = 'Basic ' + base64.b64encode(('any:' + PW).encode()).decode()
AL = 'ABCDEFGHIJKLMNO'
def nm(r, c): return '%s%d' % (AL[c], r + 1)


def call(path, payload=None, method=None, timeout=180):
    req = urllib.request.Request(API + path,
                                 data=(json.dumps(payload).encode() if payload is not None else None),
                                 headers={'Authorization': AUTH, **({'Content-Type': 'application/json'} if payload is not None else {})},
                                 method=(method or ('POST' if payload is not None else 'GET')))
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


call('/api/eval/reset', {}, method='POST')
r = call('/api/eval/start', {'weight': 'v39|weights_it153_v39.pt', 'sim': 800, 'depth': 5,
                             'we_black': True, 'temperature': 0.05, 'first_moves': [{'r': 5, 'c': 6}]})
print('start ok=', r.get('ok'))
st = r['state']
t0 = time.time()
while True:
    rr = call('/api/eval/step', {})
    if not rr.get('ok'):
        if '处理中' in str(rr.get('error', '')): time.sleep(0.4); continue
        print('step err', rr.get('error')); break
    st = rr['state']
    h = st.get('history') or []
    if h:
        last = h[-1]
        print('  mv%2d %-6s %s' % (last['move'], last['who'], nm(last['r'], last['c'])), flush=True)
    if rr.get('done') or (h and h[-1]['move'] >= 15):
        break
print('done. 前14手 =', ' '.join('%d:%s' % (x['move'], nm(x['r'], x['c'])) for x in (st.get('history') or [])[:15]))
print('第14手 =', nm(st['history'][13]['r'], st['history'][13]['c']) if len(st.get('history') or []) > 13 else None)
