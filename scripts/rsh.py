#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""rsh —— 跑一条远程命令（凭据读 web/creds.local.json）

用法: python3 scripts/rsh.py "<远程命令>"

⚠️ 实现要点（踩过）：命令必须通过 **argv** 传进 expect，不能拼进脚本正文 ——
   Tcl 会吃掉正文里的 $ / $( ) / 反引号 / 花括号；按 argv 传则整条原样保留。
   （这套模板抄自 sync_replay.py 里实测稳定的 EXPECT_SSH）
"""
import json
import os
import subprocess
import sys
import tempfile

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CRED = os.path.join(PROJ, 'web', 'creds.local.json')

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


def main():
    if len(sys.argv) < 2:
        sys.exit('用法: rsh.py "<远程命令>"')
    cmd = sys.argv[1]
    c = json.load(open(CRED))
    pw = c.get('pass') or c.get('password') or ''
    if not pw:
        sys.exit('✗ creds.local.json 缺密码字段')

    with tempfile.NamedTemporaryFile('w', suffix='.exp', delete=False) as f:
        f.write(EXPECT_SSH)
        path = f.name
    try:
        r = subprocess.run(['expect', path, pw, str(c['port']), c['host'], c['user'], cmd],
                           capture_output=True, text=True, timeout=620)
    finally:
        os.unlink(path)

    for line in r.stdout.splitlines():
        s = line.rstrip()
        if not s or s.startswith('spawn '):
            continue
        print(s)
    if r.returncode != 0:
        print('[rc=%s] %s' % (r.returncode, r.stderr.strip()[:300]), file=sys.stderr)


if __name__ == '__main__':
    main()
