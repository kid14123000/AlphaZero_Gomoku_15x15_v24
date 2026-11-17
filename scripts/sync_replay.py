#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""sync_replay.py —— 一条命令把服务器上的自对弈数据同步到本地并重建索引

做的事（每步都会打印 + 失败即退出）:
  1. 读 web/creds.local.json 拿服务器地址/密码（不入库）
  2. 远端 tar czf 打包 worker_data/ + game_data/（pkl 压缩率约 8×，15MB → 2MB）
  3. scp 下来（这条链路实测 ~20KB/s，2MB 约 2 分钟）
  4. 解包到本地 <ckpt_dir>/（只补新增的 pkl，不动权重与索引）
  5. 调用 scripts/build_replay_index.py 重建 replay_pool_index.json
  6. 打印每代黑白/手数，便于确认

为什么不在 web 里点按钮做：SSH 拉 2MB 会把 Flask 的单请求拖到 2 分钟以上，
而监控页本身每点一次刷新就已经要 30s。这个操作低频（一代一次），走命令行更合适。

用法:
  python3 scripts/sync_replay.py                # 用 active_slot 的槽
  python3 scripts/sync_replay.py --version v24b
  python3 scripts/sync_replay.py --keep-tar     # 保留 /tmp 下的压缩包备查
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(PROJ, 'web'))

EXPECT_SCP = r'''
set timeout 1800
set pw [lindex $argv 0]
set port [lindex $argv 1]
set host [lindex $argv 2]
set user [lindex $argv 3]
set rpath [lindex $argv 4]
set lpath [lindex $argv 5]
spawn scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR \
    -o ServerAliveInterval=15 -o ServerAliveCountMax=120 -o Compression=no \
    -P $port $user@$host:$rpath $lpath
expect {
    -re "(?i)password:" { send -- "$pw\r"; exp_continue }
    eof
}
catch wait result
exit [lindex $result 3]
'''

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
    -re "(?i)yes/no"    { send -- "yes\r"; exp_continue }
    eof
}
catch wait result
exit [lindex $result 3]
'''


def load_cred():
    p = os.path.join(PROJ, 'web', 'creds.local.json')
    if not os.path.exists(p):
        sys.exit('✗ 缺 web/creds.local.json（含主机/端口/用户/密码，不入库）')
    c = json.load(open(p))
    for k in ('host', 'port', 'user', 'pass'):
        if not c.get(k):
            sys.exit(f'✗ creds.local.json 缺字段 {k}')
    return c


def _run_expect(tmpl, args, timeout):
    with tempfile.NamedTemporaryFile('w', suffix='.exp', delete=False) as f:
        f.write(tmpl)
        path = f.name
    try:
        return subprocess.run(['expect', path] + [str(a) for a in args],
                              capture_output=True, text=True, timeout=timeout)
    finally:
        os.unlink(path)


def ssh(c, cmd, timeout=620):
    r = _run_expect(EXPECT_SSH, [c['pass'], c['port'], c['host'], c['user'], cmd], timeout)
    out = '\n'.join(l for l in r.stdout.splitlines() if not l.strip().startswith('spawn'))
    return r.returncode, out


def scp(c, rpath, lpath, timeout=1850):
    r = _run_expect(EXPECT_SCP, [c['pass'], c['port'], c['host'], c['user'], rpath, lpath], timeout)
    out = '\n'.join(l for l in r.stdout.splitlines() if not l.strip().startswith('spawn'))
    return r.returncode, out


def resolve_targets(version):
    """返回 (远端 ckpt 目录, 本地 ckpt 目录, 远端项目根)"""
    from slot_config import active_cfg  # type: ignore
    cfg = active_cfg(override=version)
    ckpt_name = cfg.get('ckpt_basename') or f'gomoku_checkpoints_{version}'
    remote_root = cfg.get('remote_root') or '/root/pk_v8'
    return f'{remote_root}/{ckpt_name}', os.path.join(PROJ, ckpt_name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--version', default=None)
    ap.add_argument('--keep-tar', action='store_true')
    a = ap.parse_args()

    try:
        from slot_config import active_cfg  # noqa: F401
    except Exception as e:
        sys.exit(f'✗ 无法导入 web/slot_config.py: {e}')

    version = a.version or active_cfg()['slot']
    remote_dir, local_dir = resolve_targets(version)
    c = load_cred()
    print(f'版本 {version}')
    print(f'  远端 {c["user"]}@{c["host"]}:{remote_dir}')
    print(f'  本地 {local_dir}')

    if not os.path.isdir(local_dir):
        os.makedirs(local_dir, exist_ok=True)

    # ── 1. 远端统计 + 打包 ──
    t0 = time.time()
    rb = f'/tmp/replay_bundle_{version}.tgz'
    cmd = (f'cd {remote_dir} || exit 1; '
           f'echo "WK=$(ls -1 worker_data/*.pkl 2>/dev/null | wc -l)"; '
           f'echo "GM=$(ls -1 game_data/*.pkl 2>/dev/null | wc -l)"; '
           f'rm -f {rb}; tar czf {rb} worker_data game_data 2>/dev/null; '
           f'stat -c "%s" {rb}')
    print('\n[1/4] 远端打包…')
    rc, out = ssh(c, cmd)
    if rc != 0:
        sys.exit(f'✗ 远端打包失败 rc={rc}\n{out[-800:]}')
    stats = {}
    size = None
    for line in out.splitlines():
        line = line.strip()
        if '=' in line and line.split('=')[0] in ('WK', 'GM'):
            k, v = line.split('=', 1)
            stats[k] = int(v) if v.isdigit() else 0
        elif line.isdigit():
            size = int(line)
    if not size:
        sys.exit(f'✗ 压缩包大小解析失败（远端输出: {out[-400:]}）')
    print(f'    worker_data {stats.get("WK", 0)} 个 · game_data {stats.get("GM", 0)} 个')
    print(f'    压缩包 {size/1048576:.2f} MB')
    if stats.get('WK', 0) == 0 and stats.get('GM', 0) == 0:
        sys.exit('✗ 服务器上没有 pkl，没什么可同步的')

    # ── 2. 下载 ──
    tmpdir = tempfile.mkdtemp(prefix='replay_sync_')
    tgz = os.path.join(tmpdir, 'bundle.tgz')
    print(f'\n[2/4] 下载（~20KB/s，约 {size/20480/60:.1f} 分钟）…')
    rc, out = scp(c, rb, tgz)
    got = os.path.getsize(tgz) if os.path.exists(tgz) else 0
    if rc != 0 or got != size:
        sys.exit(f'✗ 下载不完整 rc={rc} got={got} want={size}\n{out[-400:]}')
    print(f'    完成 {got/1048576:.2f} MB，用时 {time.time()-t0:.0f}s')

    # ── 3. 解包（只补新增，不覆盖权重）──
    print('\n[3/4] 解包…')
    before = set()
    for sub in ('worker_data', 'game_data'):
        d = os.path.join(local_dir, sub)
        if os.path.isdir(d):
            before |= {f'{sub}/{f}' for f in os.listdir(d) if f.endswith('.pkl')}
    r = subprocess.run(['tar', 'xzf', tgz, '-C', local_dir], capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f'✗ 解包失败: {r.stderr[-400:]}')
    after = set()
    for sub in ('worker_data', 'game_data'):
        d = os.path.join(local_dir, sub)
        if os.path.isdir(d):
            after |= {f'{sub}/{f}' for f in os.listdir(d) if f.endswith('.pkl')}
    new = after - before
    print(f'    本地共 {len(after)} 个 pkl（本次新增 {len(new)}，已有跳过）')

    if not a.keep_tar:
        shutil.rmtree(tmpdir, ignore_errors=True)
    else:
        print(f'    压缩包保留在 {tgz}')

    # ── 4. 重建索引 ──
    print('\n[4/4] 重建索引…')
    # 索引脚本要 pandas：优先用项目 .venv 的解释器（系统 python3 常常没有 pandas，
    # 那时 build_replay_index 会把每个文件都记进 bad 然后报“没找到任何 pkl”，极具误导性）
    venv_py = os.path.join(PROJ, '.venv', 'bin', 'python')
    py = venv_py if os.path.exists(venv_py) else sys.executable
    r = subprocess.run([py, os.path.join(HERE, 'build_replay_index.py'),
                        '--version', version, '--ckpt-dir', local_dir],
                       capture_output=True, text=True)
    print(r.stdout.rstrip())
    if r.returncode != 0:
        sys.exit(f'✗ 建索引失败:\n{r.stderr[-600:]}')

    print(f'\n✅ 完成，总用时 {time.time()-t0:.0f}s')
    print('   打开 http://127.0.0.1:8765/replay 即可查看（无需重启 web，索引按 mtime 自动失效）')
    return 0


if __name__ == '__main__':
    sys.exit(main())
