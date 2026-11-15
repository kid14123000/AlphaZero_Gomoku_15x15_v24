#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""一条命令切换本地 web 的活跃槽 (v38 <-> v39)

用法:
    python3 gomoku_web/set_slot.py v39                       # 切到 v39 槽, 轮次/目标自动带出
    python3 gomoku_web/set_slot.py v38 --round 4 --coach "..." --student "..."
    python3 gomoku_web/set_slot.py v39 --no-restart          # 只写配置, 稍后自己重启
    python3 gomoku_web/set_slot.py --show                    # 只看当前配置 + 服务器实跑槽

做三件事:
    ① 写 gomoku_web/active_slot.json   —— 双槽轮次的【唯一真相源】
    ② 重启 web 进程                    —— 内存常量在启动时定死, 不重启不生效
    ③ 核验并打印                       —— 配置槽 / 服务器实跑槽 / 是否一致 / 本地数据就位情况

背景(2026-09-27 用户: "来回切弄简单点"):
    以前每轮要手改 4 处: app.py 常量 / replay_api.py 默认 / 6个页面 _pref / 扫描脚本默认
    → 现在全部派生自 active_slot.json, 切换只跑这一条命令 ✓
"""
import argparse
import base64
import json
import os
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from slot_config import SLOT_META, active_cfg, write_slot   # noqa: E402

WEB_URL = 'http://127.0.0.1:8765'
LOG = os.path.join(ROOT, 'web_v36.log')


def _auth_header():
    pwd = os.environ.get('GOMOKU_PASSWORD', '')
    return {'Authorization': 'Basic ' + base64.b64encode(('x:' + pwd).encode()).decode()}


def _get(path, timeout=200):
    req = urllib.request.Request(WEB_URL + path, headers=_auth_header())
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def restart_web():
    """重启本地 web (pkill 旧 + 后台拉起) —— 内存常量在 import 时定死, 必须重启"""
    subprocess.run("pkill -f '[a]pp\\.py'", shell=True)
    time.sleep(2)
    with open(LOG, 'ab') as f:
        f.write(('\n===== %s  set_slot 重启 web =====\n'
                 % time.strftime('%Y-%m-%d %H:%M:%S')).encode())
        subprocess.Popen([sys.executable, 'app.py'], cwd=HERE,
                         stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
    for _ in range(40):                      # 最多等 40s 起监听
        time.sleep(1)
        try:
            _get('/api/slot', timeout=5)
            return True
        except Exception:
            continue
    return False


def show_status():
    cfg = active_cfg()
    print('── 配置 (active_slot.json) ──')
    print('  槽 = %s | 轮次 = %s | 对比 = %s | 达标线 = %s%% | 来源 = %s'
          % (cfg['slot'], cfg['round'], cfg['compare'], round(cfg['goal'] * 100), cfg['source']))
    print('  教练备注: %s' % cfg['coach_note'])
    if cfg.get('student_note'):
        print('  学生备注: %s' % cfg['student_note'])
    print('  口径: 学生执%s(主指标) / 教练执%s' % (cfg['student_name'], cfg['coach_name']))
    for label, path in (('本地数据目录', os.path.join(ROOT, 'gomoku_checkpoints_%s' % cfg['slot'])),
                        ('本地索引', os.path.join(ROOT, 'gomoku_checkpoints_%s' % cfg['slot'],
                                                'replay_pool_index.json'))):
        if os.path.exists(path):
            if os.path.isdir(path):
                wd = os.path.join(path, 'worker_data')
                n = len([x for x in os.listdir(wd) if x.endswith('.pkl')]) if os.path.isdir(wd) else 0
                print('  %s: 存在 (worker_data %d 局)' % (label, n))
            else:
                try:
                    d = json.load(open(path))
                    its = sorted(int(k) for k in d.get('iters', {}))
                    print('  %s: %d 代 (it%s~it%s)' % (label, len(its), its[0], its[-1]) if its else '  %s: 空' % label)
                except Exception:
                    print('  %s: %s' % (label, path))
        else:
            print('  %s: ✗ 不存在 (%s)' % (label, path))
    try:
        t = _get('/api/training')
        print('── 服务器实跑 ──')
        print('  实跑槽 = %s | web 进程槽 = %s | 一致 = %s'
              % (t.get('slot_running') or '未探测到', t.get('slot_config') or '?',
                 '✓' if not t.get('slot_mismatch') else '✗ 不一致!'))
        print('  当前代 = %s | 进度 = %s/%s | 教练色 = %s'
              % (t.get('current_it'), t.get('current_game'), t.get('current_denom'),
                 (t.get('coach') or {}).get('color')))
    except Exception as e:
        print('  (取服务器状态失败: %s)' % e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('slot', nargs='?', help='目标槽: v39 (学生黑) 或 v38 (学生白)')
    ap.add_argument('--round', type=int, default=None, help='第几轮共进化')
    ap.add_argument('--coach', default='', help='教练来源备注 (如 "A2\' = v38 it77")')
    ap.add_argument('--student', default='', help='学生起点备注 (如 "黑方起点 = v39 it39")')
    ap.add_argument('--goal', type=float, default=None, help='达标线 (0.8 = 80%%)')
    ap.add_argument('--compare', default=None, help='对比线版本 (默认另一个槽)')
    ap.add_argument('--no-restart', action='store_true', help='只写配置不重启')
    ap.add_argument('--show', action='store_true', help='只看状态')
    a = ap.parse_args()

    if a.show or not a.slot:
        show_status()
        return 0

    if a.slot not in SLOT_META:
        print('✗ 未知槽 %r, 可选: %s' % (a.slot, list(SLOT_META)))
        return 1

    d = write_slot(a.slot, round_no=a.round, coach_note=a.coach, student_note=a.student,
                   goal=a.goal, compare=a.compare)
    print('✓ 已写 %s' % os.path.join(HERE, 'active_slot.json'))
    print(json.dumps(d, ensure_ascii=False, indent=2))

    if a.no_restart:
        print('⚠ 未重启 web (--no-restart): 内存常量还是旧槽 → 记得 `pkill -f app.py` 后重启')
        return 0
    print('… 重启 web 中')
    ok = restart_web()
    print('✓ web 已重启' if ok else '✗ web 重启后未响应, 看 %s' % LOG)
    print()
    show_status()
    print()
    other = 'v38' if a.slot == 'v39' else 'v39'     # 另一个槽 = 上一轮刚跑完的那个
    print('提示: 本轮自对弈数据要拉的话 → python3 /tmp/pull_replay.py %s <起始代> <结束代>'
          % a.slot)
    print('      上一轮的过程数据记得归档: mv gomoku_checkpoints_%s gomoku_checkpoints_%s_round<K>/'
          % (other, other))
    print('      ⚠️ 归档后必须建软链(否则回放页取不到归档版本): '
          'ln -sfn gomoku_checkpoints_%s_round<K> gomoku_checkpoints_%sr<1>' % (other, other))
    return 0 if ok else 2


if __name__ == '__main__':
    sys.exit(main())
