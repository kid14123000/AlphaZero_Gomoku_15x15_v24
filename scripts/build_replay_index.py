#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""build_replay_index.py —— 扫描自对弈 pkl，生成 replay_pool_index.json

原项目用 scripts/scan_replay_index.py，但那个脚本不在本仓库（只在老服务器上）。
本脚本按 web/replay_api.py 的读取契约重写，字段完全对齐：

  {"version":..., "built_at":..., "iter_range":[lo,hi], "total_games":N,
   "iters": {"<it>": {"games": n,
                      "files": [{"file":..., "steps":..., "winner":...,
                                 "worker":..., "game":...}, ...]}}}

口径（与 replay_api 一致）:
  winner  1 = 黑胜, -1 = 白胜, 0 = 平局   （pkl 里 df['winner'].iloc[0]）
  steps   = 手数 = len(df)
  file    = pkl 文件名（worker_data 与 game_data 都按裸文件名登记，
            replay_api.load_game_df 会先找 worker_data 再回退 game_data）
  worker  = 工兵号（主进程的局为 -1）
  game    = 该工兵/主进程本代的第几局

用法:
  python3 scripts/build_replay_index.py                 # 自动取 active_slot 的槽
  python3 scripts/build_replay_index.py --version v24b
  python3 scripts/build_replay_index.py --version v24b --ckpt-dir gomoku_checkpoints_v24
"""
import argparse
import glob
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)

# 槽号 → 实际检查点目录名（v24b 的脚本内 VERSION 仍是 v24，目录名不能从槽号派生）
try:
    sys.path.insert(0, os.path.join(PROJ, 'web'))
    from slot_config import active_cfg  # type: ignore
except Exception:
    active_cfg = None


def default_ckpt_dir(version):
    if active_cfg:
        try:
            cfg = active_cfg()
            if cfg.get('slot') == version and cfg.get('ckpt_basename'):
                return os.path.join(PROJ, cfg['ckpt_basename'])
        except Exception:
            pass
    return os.path.join(PROJ, f'gomoku_checkpoints_{version}')


def parse_name(fn):
    """iter12_3_game_7.pkl -> (12, 3, 7)   iter12_game_7.pkl -> (12, -1, 7)"""
    m = re.match(r'^iter(\d+)_(\d+)_game_(\d+)\.pkl$', fn)
    if m:
        return int(m.group(1)), int(m.group(2)), int(m.group(3))
    m = re.match(r'^iter(\d+)_game_(\d+)\.pkl$', fn)
    if m:
        return int(m.group(1)), -1, int(m.group(2))
    return None


def read_pkl_meta(path):
    """只取 winner 与手数；pkl 是 pandas DataFrame（列 player/board/prob/winner）"""
    import pickle
    with open(path, 'rb') as f:
        df = pickle.load(f)
    try:
        winner = int(df['winner'].iloc[0])
    except Exception:
        winner = 0
    steps = len(df)
    return winner, steps


def build(version, ckpt_dir, workers_only=False, verbose=True):
    wdir = os.path.join(ckpt_dir, 'worker_data')
    gdir = os.path.join(ckpt_dir, 'game_data')
    iters = {}
    total = 0
    bad = []

    def add(path, is_worker):
        nonlocal total
        fn = os.path.basename(path)
        p = parse_name(fn)
        if not p:
            bad.append(fn)
            return
        it, wid, gid = p
        if is_worker and wid < 0:
            wid = -1
        try:
            winner, steps = read_pkl_meta(path)
        except Exception as e:
            bad.append(f'{fn} ({e})')
            return
        ent = iters.setdefault(it, {'games': 0, 'files': []})
        ent['files'].append({'file': fn, 'steps': steps, 'winner': winner,
                             'worker': wid, 'game': gid})
        ent['games'] = len(ent['files'])
        total += 1

    for p in sorted(glob.glob(os.path.join(wdir, 'iter*_game_*.pkl'))):
        add(p, True)
    if not workers_only:
        for p in sorted(glob.glob(os.path.join(gdir, 'iter*_game_*.pkl'))):
            add(p, False)

    if not iters:
        print('✗ 没找到任何可解析的 pkl —— 检查路径: %s / %s' % (wdir, gdir))
        if bad:
            print('   %d 个文件解析失败，前 3 个：%s' % (len(bad), bad[:3]))
        return None

    for it in iters:
        iters[it]['files'].sort(key=lambda g: (g['worker'], g['game']))

    keys = sorted(iters)
    idx = {
        'version': version,
        'built_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'ckpt_dir': os.path.abspath(ckpt_dir),
        'iter_range': [keys[0], keys[-1]],
        'total_games': total,
        'iters': {str(k): iters[k] for k in keys},
    }
    out = os.path.join(ckpt_dir, 'replay_pool_index.json')
    with open(out, 'w') as f:
        json.dump(idx, f, ensure_ascii=False)
    if verbose:
        print('✅ 写出 %s' % out)
        print('   代 %d 个 (%d..%d) · 局 %d · %d 字节'
              % (len(keys), keys[0], keys[-1], total, os.path.getsize(out)))
        for k in keys:
            e = iters[k]
            w = sum(1 for g in e['files'] if g['winner'] == -1)
            print('   代%-3d  %3d 局  白胜 %2d (%4.1f%%)  手数 中位 %s'
                  % (k, e['games'], w, 100.0 * w / e['games'],
                     sorted(g['steps'] for g in e['files'])[len(e['files']) // 2]))
        if bad:
            print('   ⚠ 跳过 %d 个文件，例: %s' % (len(bad), bad[:3]))
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--version', default=None, help='槽号，默认取 active_slot')
    ap.add_argument('--ckpt-dir', default=None, help='检查点目录，默认按槽推导')
    ap.add_argument('--workers-only', action='store_true', help='只扫 worker_data')
    a = ap.parse_args()

    version = a.version
    if not version:
        if active_cfg:
            try:
                version = active_cfg()['slot']
            except Exception:
                version = None
        version = version or 'v24b'
    ckpt = a.ckpt_dir or default_ckpt_dir(version)
    print('版本 %s · 目录 %s' % (version, ckpt))
    idx = build(version, ckpt, a.workers_only)
    return 0 if idx else 1


if __name__ == '__main__':
    sys.exit(main())
