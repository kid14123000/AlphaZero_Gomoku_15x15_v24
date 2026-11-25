#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""独立探针: 弄清 pbrain-rapfi 的 BOARD 行 who 字段语义 + depth5 用时。
不影响正在跑的 web eval (只读 config, 单独进程)。
"""
import subprocess, os, time, select

BIN = os.path.join(os.environ.get('GOMOKU_RAPFI_DIR')
                   or os.path.expanduser('~/rapfi_run'), 'pbrain-rapfi')


def read_line(p, timeout=20):
    end = time.time() + timeout
    while time.time() < end:
        r, _, _ = select.select([p.stdout], [], [], 0.5)
        if r:
            line = p.stdout.readline()
            if not line:
                return None
            line = line.strip()
            if line and not line.startswith('MESSAGE'):
                return line
        if p.poll() is not None:
            return None
    return None


def probe(board_rows, label):
    """board_rows: list of (r,c,who)  who 为发出时所用标记
    返回 (落子, 用时)"""
    p = subprocess.Popen([BIN], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         text=True, bufsize=1, cwd=os.path.dirname(BIN))
    end = time.time() + 3.0
    while time.time() < end:
        r, _, _ = select.select([p.stdout], [], [], 0.25)
        if not r:
            break
        if not p.stdout.readline():
            break
    p.stdin.write('START 15\n'); p.stdin.flush()
    read_line(p)
    lines = ['BOARD'] + ['{},{},{}'.format(c, r, who) for (r, c, who) in board_rows] + ['DONE']
    t0 = time.time()
    p.stdin.write('\n'.join(lines) + '\n'); p.stdin.flush()
    resp = read_line(p, 30)
    dt = time.time() - t0
    p.kill()
    return resp, dt, label


# 局面: 黑(我方)有一条活四 (7,5)-(7,8), 白(引擎执白)有 3 子。轮到白走 → 白必须堵 (7,4)/(7,9)
black = [(7, 5), (7, 6), (7, 7), (7, 8)]
white = [(5, 5), (6, 6), (8, 8)]

# A: web 的映射 (黑→1, 白→2)
rowsA = [(r, c, 1) for (r, c) in black] + [(r, c, 2) for (r, c) in white]
# B: 相对引擎映射 (引擎=白 → 白子=1, 黑子=2)
rowsB = [(r, c, 2) for (r, c) in black] + [(r, c, 1) for (r, c) in white]

for rows, label in [(rowsA, 'A=黑1白2(web映射)'), (rowsB, 'B=黑2白1(相对引擎)')]:
    resp, dt, lb = probe(rows, label)
    # 解析
    mv = None
    if resp:
        parts = resp.split(',')
        if len(parts) >= 2 and parts[0] != '0':
            try:
                mv = (int(parts[1]), int(parts[0]))   # (r,c)
            except ValueError:
                mv = ('parse', resp)
    blocks = mv in [(7, 4), (7, 9)]
    print('%-26s -> 落子=%s 用时=%.3fs  堵活四=%s' % (lb, mv, dt, blocks))

# depth5 用时抽样: 空盘附近 + 中度局面
print('\n-- depth5 用时抽样 --')
for rows, lbl in [([(7, 7, 1)], '仅黑1(天元)'), (rowsA, '中度局面(黑活四)')]:
    resp, dt, lb = probe(rows, lbl)
    print('%-16s 用时=%.3fs 落子=%s' % (lbl, dt, resp))
