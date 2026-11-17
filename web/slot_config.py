#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""活跃槽配置 —— v38/v39 双槽交替轮次的【唯一真相源】

背景（2026-09-26 用户: "切换要弄简单点"）:
  以前每换一轮（v38↔v39）要改 4 处:
    ① gomoku_web/app.py   的 TRAIN_VERSION / COMPARE_VERSION / INFER_LOG / VERSION_COACH
    ② gomoku_web/replay_api.py 的 DEFAULT_VERSION
    ③ scripts/scan_replay_index.py 的 --version 默认
    ④ 6 个页面 HTML 里硬编码的 _pref 默认版本链
  漏一处就会出现"驾驶舱看的是这一轮、回放页看的是上一轮"的错位 ✗

现在:
  · 只有 gomoku_web/active_slot.json 一个文件是真相源（槽号 + 轮次 + 教练备注）
  · app.py / replay_api.py / 扫描脚本 / 页面 全部从这里派生
  · 切换 = `python3 gomoku_web/set_slot.py v39`（写文件 + 重启 web + 打印核验）

优先级: 环境变量 GOMOKU_TRAIN_VERSION > active_slot.json > DEFAULT_SLOT
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SLOT_FILE = os.path.join(HERE, 'active_slot.json')
DEFAULT_SLOT = 'v39'

# ── 每槽固定不变的角色/达标线（轮次无关，切换时不用改） ──────────────────
#   v39 槽: 学生执黑(先手) / 教练执白 / 达标 80%
#   v38 槽: 学生执白(后手) / 教练执黑 / 达标 55%
SLOT_META = {
    'v39': {'student_color': -1, 'student_name': '黑', 'coach_name': '白', 'coach_color': -1,
            'goal': 0.80, 'infer_log': '/root/pk_v8/infer_server_v39.log',
            'note': '镜像轮: 教练执白(冻结), 只训黑方'},
    'v38': {'student_color': 1, 'student_name': '白', 'coach_name': '黑', 'coach_color': 1,
            'goal': 0.55, 'infer_log': '/root/pk_v8/infer_server_v37.log',
            'note': '教练执黑(冻结), 只训白方; 推理服务沿用 v37 脚本 → 日志名带 _v37'},
    # ★ 2026-10-08 用户: "返璞归真, 按 v1 版做了单通道版的 v40 在训练"
    #   · v40 = INPUT_CHANNELS 4→1 (canonical 单通道) + 自由开局(RANDOM_OPEN_MOVES=0) + qa软标签
    #   · 【无教练】纯自对弈 → coach_color=None; 日志里也没有"胜率"行, 指标只能由棋谱数据算
    #   · 达标线语义不同: 无教练线没有"达标"; 30% = v40 脚本注释里的【失衡线(下限)】
    'v40': {'student_color': -1, 'student_name': '白', 'coach_name': '', 'coach_color': None,
            'goal': 0.30, 'infer_log': '/root/pk_v8/infer_server_v40.log', 'workers': 20,
            'note': 'v40 单通道版(按v1): 纯自对弈·无教练; 指标=自对弈白方胜率; 30%=失衡线(下限)'},
    # ★ 2026-10-09 用户: "现在在训练v41了, web适配下"
    #   v41 = v40 + 梯度检查点(gradient checkpointing, 省显存) —— 其余口径与 v40 完全一致
    'v41': {'student_color': -1, 'student_name': '白', 'coach_name': '', 'coach_color': None,
            'goal': 0.30, 'infer_log': '/root/pk_v8/infer_server_v41.log', 'workers': 20,
            'note': 'v41 = v40 + 梯度检查点; 单通道纯自对弈·无教练; 30%=失衡线(下限)'},
    # ★ 2026-10-10 用户: "训练切到v42了, web适配下"
    #   v42 = v39(教练法) + 【自动换边调度】 + v41 移植梯度检查点
    #   · 换边: 学生达标 ⇒ 冻结为新教练, 原教练转正为新学生, 颜色随之翻转
    #     (swap_state.json: swap_index/last_swap_gen/coach_color; 实测 #1 由 学生黑62% 触发)
    #   · ⚠️ 学生/教练【执色是动态的】⇒ 下面两色只是快照, 真正以工兵日志
    #     「教练已加载 ... 执X」为准 (app.py 已有该优先逻辑 ✓)
    'v42': {'student_color': -1, 'student_name': '白', 'coach_name': '黑', 'coach_color': 1,
            'goal': 0.55, 'infer_log': '/root/pk_v8/infer_server_v42.log', 'workers': 24,
            'note': 'v42 教练法+自动换边: 阈值0.55且连续2代达标才换边(最短间隔5代); 颜色动态, 以工兵日志为准'},
    # ★ 2026-11-15: v24b = v24 原版 + 【黑方空盘首手改走「|value| 最小」的均衡手】
    #   · 纯自对弈, 无教练/无门控/无攒批推理服务(每工兵本地推理)
    #   · 布局与 v36~v42 完全不同: 项目根 = /root/workspace/run_v24b,
    #     权重 = weights_it{N}_v24.pt, 工兵日志 = worker_v24_{i}.log, 主日志 = train_v24b.log
    'v24b': {'student_color': -1, 'student_name': '白', 'coach_name': '', 'coach_color': None,
             'goal': 0.30, 'infer_log': '', 'workers': 19,   # 2026-11-16 用户改 9→19 工兵
             'remote_root': '/root/workspace/run_v24b',
             'ckpt_basename': 'gomoku_checkpoints_v24',   # ⚠️ 脚本内 VERSION 仍是 "v24"
             'weight_suffix': 'v24',                      # 权重文件名后缀 weights_it{N}_v24.pt
             'main_log': 'train_v24b.log',
             'worker_log_fmt': 'worker_v24_{i}.log',
             'note': 'v24b = v24 + 均衡首手(中心5x5 / |value|最小前3随机); 纯自对弈无教练; 30%=白方失衡线(下限)'},
}


def _other_slot(slot):
    """loss 曲线/胜率的【对比线】版本 —— v40 是全新单通道线, 用 v39 当形状参照"""
    return {'v39': 'v38', 'v38': 'v39', 'v40': 'v39', 'v41': 'v40', 'v42': 'v41',
            'v24b': 'v24'}.get(slot, 'v39')


def load_raw():
    """读原始 JSON（缺失/损坏返回 {}，绝不抛错——保证 web 起得来）"""
    try:
        with open(SLOT_FILE, encoding='utf-8') as f:
            d = json.load(f)
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def active_cfg(override=None):
    """返回当前活跃槽的完整配置 dict（供 app.py / replay_api.py 直接用）

    key: slot, round, compare, goal, coach_note, start_note, infer_log,
         student_name, coach_name, student_color, source
    """
    raw = load_raw()
    env_slot = os.environ.get('GOMOKU_TRAIN_VERSION') or os.environ.get('GOMOKU_SLOT')
    slot = override or env_slot or raw.get('slot') or DEFAULT_SLOT
    if slot not in SLOT_META:
        slot = DEFAULT_SLOT
    meta = dict(SLOT_META[slot])
    other = _other_slot(slot)
    return {
        'slot': slot,
        'round': raw.get('round'),
        'compare': raw.get('compare') or other,
        'goal': float(raw.get('goal') or meta['goal']),
        'coach_note': raw.get('coach_note') or meta.get('note', ''),
        'student_note': raw.get('student_note', ''),
        'infer_log': raw.get('infer_log') or meta['infer_log'],
        'student_name': meta['student_name'],
        'coach_name': meta['coach_name'],
        'student_color': meta['student_color'],
        'coach_color': meta.get('coach_color'),          # None = 无教练 (v40)
        'workers': meta.get('workers', 22),
        # ★ 2026-11-15: 部署布局相关（v24b 起需要）。老槽没有这些 key → 返回 None，
        #   app.py 里各自有 /root/pk_v8 的兜底，行为不变。
        'remote_root': meta.get('remote_root'),
        'ckpt_basename': meta.get('ckpt_basename'),
        'main_log': meta.get('main_log'),
        'weight_suffix': meta.get('weight_suffix'),
        'worker_log_fmt': meta.get('worker_log_fmt'),
        'source': ('env' if env_slot else 'file' if raw.get('slot') else 'default'),
        'updated': raw.get('updated'),
    }


def write_slot(slot, round_no=None, coach_note='', student_note='', goal=None,
               compare=None, infer_log=None):
    """写 active_slot.json（set_slot.py 用）"""
    import datetime
    if slot not in SLOT_META:
        raise ValueError('未知槽: %r (可选 %s)' % (slot, list(SLOT_META)))
    other = _other_slot(slot)
    d = {
        'slot': slot,
        'round': round_no,
        'compare': compare or other,
        'goal': float(goal if goal is not None else SLOT_META[slot]['goal']),
        'coach_note': coach_note,
        'student_note': student_note,
        'infer_log': infer_log or SLOT_META[slot]['infer_log'],
        'updated': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open(SLOT_FILE, 'w', encoding='utf-8') as f:
        json.dump(d, f, ensure_ascii=False, indent=2)
    return d


if __name__ == '__main__':
    import pprint
    pprint.pprint(active_cfg())
