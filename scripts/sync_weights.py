#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""sync_weights.py —— 把服务器上缺的权重逐个拉到本地（断点续传式：.part 校验后才转正）

这条链路实测下行只有 ~12-25KB/s（上传却是 200KB/s，是单向下行限速），
所以 10MB 一个权重需要 7~14 分钟。数量多时请放后台跑。

用法:
  python3 scripts/sync_weights.py                  # 拉全部缺的
  python3 scripts/sync_weights.py --last 5         # 只拉最新 5 代
  python3 scripts/sync_weights.py --min-iter 28    # 只拉 it28 及以后
  python3 scripts/sync_weights.py --dry            # 只列清单不下载
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(PROJ, 'web'))

EXPECT_SSH = r'''
set timeout 600
set pw [lindex $argv 0]
set port [lindex $argv 1]
set host [lindex $argv 2]
set user [lindex $argv 3]
set cmd [lindex $argv 4]
spawn ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR \
    -o ConnectTimeout=20 -p $port $user@$host $cmd
expect {
    -re "(?i)password:" { send -- "$pw\r"; exp_continue }
    -re "(?i)yes/no"    { send -- "yes\r";  exp_continue }
    eof
}
catch wait result
exit [lindex $result 3]
'''

EXPECT_SCP = r'''
set timeout 2400
set pw [lindex $argv 0]
set port [lindex $argv 1]
set host [lindex $argv 2]
set user [lindex $argv 3]
set rpath [lindex $argv 4]
set lpath [lindex $argv 5]
spawn scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR \
    -o ServerAliveInterval=15 -o ServerAliveCountMax=200 -o Compression=no \
    -P $port $user@$host:$rpath $lpath
expect {
    -re "(?i)password:" { send -- "$pw\r"; exp_continue }
    eof
}
catch wait result
exit [lindex $result 3]
'''

MIN_SIZE = 6_000_000     # 正常权重 ~10MB；小于这个数视为截断


def _expect(tmpl, args, timeout):
    with tempfile.NamedTemporaryFile('w', suffix='.exp', delete=False) as f:
        f.write(tmpl)
        p = f.name
    try:
        return subprocess.run(['expect', p] + [str(a) for a in args],
                              capture_output=True, text=True, timeout=timeout)
    finally:
        os.unlink(p)


def ssh(c, cmd):
    r = _expect(EXPECT_SSH, [c['password'], c['port'], c['host'], c['user'], cmd], 620)
    return '\n'.join(l for l in r.stdout.splitlines() if not l.strip().startswith('spawn'))


def scp(c, rpath, lpath):
    r = _expect(EXPECT_SCP, [c['password'], c['port'], c['host'], c['user'], rpath, lpath], 2450)
    return '\n'.join(l for l in r.stdout.splitlines() if not l.strip().startswith('spawn'))


def iter_of(name):
    try:
        return int(name.split('weights_it')[1].split('_')[0])
    except Exception:
        return -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--version', default=None)
    ap.add_argument('--last', type=int, default=0, help='只拉最新 N 代')
    ap.add_argument('--min-iter', type=int, default=0)
    ap.add_argument('--dry', action='store_true')
    a = ap.parse_args()

    from slot_config import active_cfg  # type: ignore
    cfg = active_cfg(override=a.version) if a.version else active_cfg()
    version = cfg['slot']
    ckpt = cfg.get('ckpt_basename') or f'gomoku_checkpoints_{version}'
    remote_root = cfg.get('remote_root') or '/root/pk_v8'
    suf = cfg.get('weight_suffix') or version

    cred = json.load(open(os.path.join(PROJ, 'web', 'creds.local.json')))
    # web/creds.local.json 用 'pass' 作为键（不是 password），统一成内部约定的 password
    cred['password'] = cred.get('password') or cred.get('pass') or ''
    if not cred['password']:
        sys.exit('✗ web/creds.local.json 里没有密码字段（pass/password）')
    remote_dir = f'{remote_root}/{ckpt}'
    local_dir = os.path.join(PROJ, ckpt)
    os.makedirs(local_dir, exist_ok=True)

    print(f'版本 {version} · 远端 {remote_dir}')
    names = [l.strip().split('/')[-1] for l in ssh(cred, f'ls {remote_dir}/weights_it*.pt 2>/dev/null').splitlines()
             if l.strip().endswith('.pt')]
    names = sorted({n for n in names if n.startswith('weights_it')}, key=iter_of)
    if not names:
        sys.exit('✗ 服务器上没找到权重')

    have = {f for f in os.listdir(local_dir) if f.endswith('.pt')
            and os.path.getsize(os.path.join(local_dir, f)) >= MIN_SIZE}
    todo = [n for n in names if n not in have]
    if a.min_iter:
        todo = [n for n in todo if iter_of(n) >= a.min_iter]
    if a.last:
        todo = todo[-a.last:]

    print(f'服务器 {len(names)} 个（it{iter_of(names[0])}~it{iter_of(names[-1])}），本地完整 {len(have)} 个')
    print(f'待拉 {len(todo)} 个 · 约 {len(todo)*10} MB')
    if not todo:
        print('✅ 已是最新，无需下载')
        return 0
    if a.dry:
        for n in todo:
            print('  ', n)
        return 0

    # 磁盘余量检查（每个约 10MB，留 200MB 余量）
    st = os.statvfs(local_dir)
    free_mb = st.f_bavail * st.f_frsize / 1048576
    need_mb = len(todo) * 10 + 200
    if free_mb < need_mb:
        print(f'⚠ 磁盘余量 {free_mb:.0f}MB < 需要 {need_mb}MB，可能不够')

    t0 = time.time()
    ok = fail = 0
    for i, n in enumerate(todo, 1):
        part = os.path.join(local_dir, n + '.part')
        final = os.path.join(local_dir, n)
        eta = ''
        if ok:
            per = (time.time() - t0) / ok
            eta = f' · 剩 {(len(todo)-i+1)*per/60:.0f}min'
        print(f'[{i}/{len(todo)}] {n}{eta}', flush=True)
        try:
            scp(cred, f'{remote_dir}/{n}', part)
        except Exception as e:
            print(f'    ✗ scp 异常: {e}')
            fail += 1
            continue
        sz = os.path.getsize(part) if os.path.exists(part) else 0
        if sz >= MIN_SIZE:
            os.replace(part, final)     # 校验通过才转正，页面不会读到半截文件
            ok += 1
            print(f'    ✓ {sz/1048576:.1f} MB')
        else:
            print(f'    ✗ 只有 {sz} 字节（<{MIN_SIZE}），删掉下次重拉')
            try:
                os.remove(part)
            except OSError:
                pass
            fail += 1

    print(f'\n完成：成功 {ok} · 失败 {fail} · 用时 {(time.time()-t0)/60:.1f} 分钟')
    return 0 if fail == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
