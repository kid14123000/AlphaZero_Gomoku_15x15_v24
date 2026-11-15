#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""烟测: start + 数步, 验证 web eval API 通 + 估速"""
import json, time, base64, urllib.request, os

API = 'http://127.0.0.1:8765'
PW = os.environ.get('GOMOKU_PASSWORD', '')
AUTH = 'Basic ' + base64.b64encode(('any:' + PW).encode()).decode()


def call(path, payload=None, method=None, timeout=180):
    headers = {'Authorization': AUTH}
    data = None
    if payload is not None:
        data = json.dumps(payload).encode(); headers['Content-Type'] = 'application/json'
    req = urllib.request.Request(API + path, data=data, headers=headers,
                                 method=(method or ('POST' if data is not None else 'GET')))
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


print('reset:', call('/api/eval/reset', {}, method='POST'))
t0 = time.time()
r = call('/api/eval/start', {'weight': 'v39|weights_it153_v39.pt', 'sim': 800, 'depth': 5,
                             'we_black': True, 'temperature': 0.05,
                             'first_moves': [{'r': 7, 'c': 5}]})
print('start ok=%s err=%s  用时%.1fs' % (r.get('ok'), r.get('error'), time.time() - t0))
s = r.get('state', {})
print('  moves=%s current=%s cfg=%s' % (s.get('moves'), s.get('current'), s.get('config')))
for i in range(6):
    ts = time.time()
    rr = call('/api/eval/step', {})
    st = rr.get('state', {})
    print('  step%d done=%s moves=%s winner=%s eng_err=%s 用时%.1fs'
          % (i + 1, rr.get('done'), st.get('moves'), st.get('winner'),
             st.get('engine_error'), time.time() - ts))
    if rr.get('done'):
        break
print('smoke done, total %.1fs' % (time.time() - t0))
