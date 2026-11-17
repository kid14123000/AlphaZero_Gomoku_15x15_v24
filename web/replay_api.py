"""自对弈数据观察 API: replay_pool 经验数据索引/回放/统计/棋型搜索"""
import os, json, glob, threading
import numpy as np
import pandas as pd
from flask import Blueprint, request, jsonify

replay_bp = Blueprint('replay', __name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(SCRIPT_DIR)
# 🚨 2026-09-26 切主线 v39: 数据源版本化 —— 页面可切 v39(主线) / v38 / 更早版本对比
#   (换主线只改这里; 数据目录与索引文件都按版本派生, 别再硬编码单一版本)
#   ⚠️ 胜率口径随主线翻转 (镜像版):
#      v38 = 教练执黑/学生白 → 「白胜率」= 学生胜率 ✓
#      v39 = 教练执白/学生黑 → 「白胜率」= 教练胜率 ✗ 学生(黑)胜率 = 100 − 白胜率
#   本文件返回的 white_rate 一律是【原始白胜率】; 页面/驾驶舱换算见技能 references/version-switch-v39.md
# 🚨 2026-09-27 默认版本 = 当前活跃槽 —— 单一真相源 gomoku_web/active_slot.json
#   (以前这里写死 'v39', 与 app.py 的 TRAIN_VERSION / 页面 _pref 各改一次 → 漏一处就错位 ✗)
#   切换 = `python3 gomoku_web/set_slot.py vNN` (会重启 web)
try:
    from slot_config import active_cfg as _active_cfg, SLOT_META as _SLOT_META
    DEFAULT_VERSION = _active_cfg()['slot']
except Exception:
    _active_cfg = None
    _SLOT_META = {}
    DEFAULT_VERSION = 'v39'
KNOWN_VERSIONS = ['v24b', 'v24', 'v42', 'v41', 'v40', 'v39', 'v39r6', 'v39r3', 'v39r1',
                  'v38', 'v38r4', 'v38r2', 'v38r1', 'v37', 'v36', 'v29', 'v26']


def _ver_paths(version):
    """版本 → (worker_data 目录, 索引文件路径)

    ★ 归档回退 (2026-10-01): 轮次归档会把 gomoku_checkpoints_v39/ 改名成
      gomoku_checkpoints_v39_round6/ → 主目录消失, 复盘页读不到 ✗
      → 主目录不在就回退最新的 <槽>_round* (只读, 不会混写)
    ★ 2026-11-16: 目录名不一定能从槽号派生 —— v24b 槽的训练脚本里 VERSION 仍是 "v24"，
      权重和数据目录都叫 gomoku_checkpoints_v24。由 slot_config 的 ckpt_basename 指定，
      否则 v24b 会去找不存在的 gomoku_checkpoints_v24b，索引永远读不到。
    """
    candidates = []
    if _active_cfg:
        try:
            cfg = _active_cfg(override=version)
            if cfg.get('ckpt_basename'):
                candidates.append(cfg['ckpt_basename'])
        except Exception:
            pass
        meta = (_SLOT_META or {}).get(version) or {}
        if meta.get('ckpt_basename') and meta['ckpt_basename'] not in candidates:
            candidates.append(meta['ckpt_basename'])
    candidates.append(f'gomoku_checkpoints_{version}')
    for name in candidates:
        d = os.path.join(PROJ, name)
        if os.path.isdir(d):
            return os.path.join(d, 'worker_data'), os.path.join(d, 'replay_pool_index.json')
    d = os.path.join(PROJ, candidates[0])
    import glob as _glob, re as _re
    cands = []
    for p in _glob.glob(d + '_round*'):
        m = _re.search(r'_round(\d+)$', p)
        if m and os.path.isdir(p):
            cands.append((int(m.group(1)), p))
    if cands:
        d = sorted(cands)[-1][1]
    return os.path.join(d, 'worker_data'), os.path.join(d, 'replay_pool_index.json')


def req_version():
    """取版本: URL 参数优先, 其次 JSON body, 兜底主线版本"""
    v = request.args.get('version')
    if not v:
        try:
            v = (request.json or {}).get('version')
        except Exception:
            v = None
    return v or DEFAULT_VERSION


def coach_sidecar(version=None):
    """教练法 sidecar —— 版本目录下的 coach_student_log.json (由 scripts/pull_replay.py 生成)。

    内容 (自解释): {student_series:[{iter,color,rate,n,d}], swaps:[{index,iter,coach_from,coach_to,...}],
                    swap_config:str}
    ★ 2026-10-11: v42 会自动换边 → 原始「白胜率」换边前后意义相反 (假指标); 学生的真实胜率只在主日志里,
      由 pull_replay.py 落成此 sidecar。无该文件 (= 无教练版本 v41/v40/...) 时返回 None, 上层回退原行为。
    """
    version = version or DEFAULT_VERSION
    _, idx_file = _ver_paths(version)
    p = os.path.join(os.path.dirname(idx_file), 'coach_student_log.json')
    try:
        with open(p, encoding='utf-8') as f:
            data = json.load(f)
    except Exception:
        return None
    return data if isinstance(data, dict) and data.get('student_series') else None


_index_cache = {}          # version -> (索引文件 mtime, index|None)
_index_lock = threading.Lock()


def load_index(version=None):
    """按版本加载索引; 缓存带 mtime —— 重建索引(scan_replay_index.py)后自动失效, 不必重启 web"""
    version = version or DEFAULT_VERSION
    _, idx_file = _ver_paths(version)
    try:
        mt = os.path.getmtime(idx_file)
    except OSError:
        _index_cache[version] = (None, None)
        return None
    cached = _index_cache.get(version)
    if cached and cached[0] == mt:
        return cached[1]
    with _index_lock:
        cached = _index_cache.get(version)
        if cached and cached[0] == mt:
            return cached[1]
        val = None
        try:
            val = json.load(open(idx_file))
        except Exception:
            val = None
        _index_cache[version] = (mt, val)
        return val


def load_game_df(file, version=None):
    version = version or DEFAULT_VERSION
    pool_dir, _ = _ver_paths(version)
    path = os.path.join(pool_dir, file)
    if not os.path.exists(path):
        # 主进程对局存 game_data/ (2026-09-06: 索引已含双目录)
        path2 = os.path.join(os.path.dirname(pool_dir), 'game_data', file)
        if os.path.exists(path2):
            return pd.read_pickle(path2)
        return None
    return pd.read_pickle(path)


@replay_bp.route('/api/replay/index')
def replay_index():
    v = req_version()
    idx = load_index(v)
    if idx is None:
        return jsonify({'ok': False, 'version': v,
                        'error': f'{v} 索引未构建(跑 scripts/scan_replay_index.py --version {v})'})
    its = sorted(int(k) for k in idx['iters'].keys())
    return jsonify({'ok': True, 'version': v,
                    'versions': KNOWN_VERSIONS,
                    'iters': [{'iter': it, 'games': idx['iters'][str(it)]['games']} for it in its],
                    'iter_range': idx['iter_range'],
                    'total_games': idx['total_games']})


@replay_bp.route('/api/replay/games', methods=['POST'])
def replay_games():
    v = req_version()
    idx = load_index(v)
    if idx is None:
        return jsonify({'ok': False, 'error': f'{v} 索引未构建'})
    data = request.json or {}
    it = str(data.get('iter'))
    if it not in idx['iters']:
        return jsonify({'ok': False, 'error': f'无 iter {it}'})
    games = []
    for g in idx['iters'][it]['files']:
        games.append({'file': g['file'], 'steps': g['steps'], 'winner': g['winner'],
                      'worker': g['worker'], 'game': g['game']})
    return jsonify({'ok': True, 'iter': int(it), 'games': games})


@replay_bp.route('/api/replay/game', methods=['POST'])
def replay_game():
    """单局完整轨迹: stones 序列(每步落子, v=1黑/-1白) + winner"""
    data = request.json or {}
    f = data.get('file')
    if not f:
        return jsonify({'ok': False, 'error': '缺 file'})
    df = load_game_df(f, req_version())
    if df is None:
        return jsonify({'ok': False, 'error': f'文件不存在: {f}'})
    boards = [np.asarray(b, dtype=np.int8) for b in df['board'].values]
    winner = int(df['winner'].iloc[0])
    players = df['player'].values
    # v33 (2026-09-05): trajectory 带 qa 列 (每步 MCTS Q 软标签, 当前玩家视角) → 随石头返回供前端展示
    has_qa = 'qa' in df.columns
    qas = df['qa'].values if has_qa else None
    stones = []
    prev = np.zeros((15, 15), dtype=np.int8)
    for i, b in enumerate(boards):
        d = np.argwhere(prev != b)
        if len(d):
            r, c = d[0]
            # board[i] 是第 i 步落子前局面, 第 i 步落子者 = players[i-1] (board[i-1] 的当前玩家)
            v = int(players[i - 1]) if i >= 1 else int(players[0])
            st = {'r': int(r), 'c': int(c), 'v': v}
            # qa[k] = 第 k 行(落子者 players[k])决策时的 Q → stones[k] 的 qa = qas[i-1]
            if has_qa and i >= 1:
                st['qa'] = round(float(qas[i - 1]), 4)
            stones.append(st)
        prev = b

    # 补致胜手: 经验数据只存到"最后一手之前"的局面, 形成五连的那一手不在 board 里
    final = prev.copy()
    if not has_five(final, 1) and not has_five(final, -1) and winner != 0:
        win_pos = find_five_pos(final, winner)
        if win_pos:
            r, c = win_pos
            stones.append({'r': int(r), 'c': int(c), 'v': winner, 'win': True})
    return jsonify({'ok': True, 'file': f, 'winner': winner,
                    'steps': len(stones), 'stones': stones,
                    'final_win': bool(stones and stones[-1].get('win'))})


def has_five(board, color):
    for r in range(15):
        for c in range(15):
            if board[r, c] != color:
                continue
            for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                if all(0 <= r + dr * k < 15 and 0 <= c + dc * k < 15 and
                       board[r + dr * k, c + dc * k] == color for k in range(5)):
                    return True
    return False


def find_five_pos(board, color):
    """找空位, 落 color 子后成五连 (致胜手位置)"""
    for r in range(15):
        for c in range(15):
            if board[r, c] != 0:
                continue
            b2 = board.copy()
            b2[r, c] = color
            if has_five(b2, color):
                return (r, c)
    return None


@replay_bp.route('/api/replay/white_rate', methods=['GET'])
def replay_white_rate():
    """每代白胜率 + 中位手数趋势 (winner: 1=黑胜 -1=白胜)
    ⚠️ 两个指标要一起看: 页面自己写的崩盘信号 = 白胜率持续 <20% **+ 中位手数缩短**
       (2026-09-25 补: 手数此前只算不画 → v37 的 57→15 手数陡降差点漏掉)
    支持 ?version=v37 切换数据源 (默认主线)"""
    version = request.args.get('version') or DEFAULT_VERSION
    _, idx_file = _ver_paths(version)
    if not os.path.exists(idx_file):
        return jsonify({'ok': False, 'error': f'{version} 索引不存在: {idx_file}'})
    try:
        idx = json.load(open(idx_file))
    except Exception:
        return jsonify({'ok': False, 'error': f'{version} 索引读取失败'})
    items = []
    for it in sorted(idx['iters'], key=int):
        games = idx['iters'][it]['files']
        total = len(games)
        white = sum(1 for g in games if g['winner'] == -1)
        black = sum(1 for g in games if g['winner'] == 1)
        draw = total - white - black
        steps = sorted(g['steps'] for g in games if g.get('steps'))
        med = steps[len(steps) // 2] if steps else 0
        # 速杀局占比 (<20 手) —— v36 时代就用的"速杀/堵点"观测口径
        quick = sum(1 for s in steps if s < 20)
        items.append({'iter': int(it), 'total': total, 'white': white, 'black': black,
                      'draw': draw, 'steps_med': med, 'steps_min': steps[0] if steps else 0,
                      'steps_max': steps[-1] if steps else 0,
                      'quick_rate': round(quick / len(steps) * 100, 1) if steps else 0,
                      'white_rate': round(white / total * 100, 1) if total else 0})
    resp = {'ok': True, 'version': version, 'items': items,
            'coach_mode': False}
    # ★ 2026-10-11: 教练法 (v42) 有 sidecar → 附带「学生胜率序列 + 换边点」, 供复盘页画真实的
    #   学生曲线 (原始白胜率会被自动换边翻转口径, 是假指标)。无 sidecar 版本 (v41/v40/…) 走原行为。
    sc = coach_sidecar(version)
    if sc:
        resp['coach_mode'] = True
        resp['student_series'] = sc.get('student_series') or []
        resp['swaps'] = sc.get('swaps') or []
        resp['swap_config'] = sc.get('swap_config') or ''
    return jsonify(resp)


@replay_bp.route('/api/replay/center', methods=['POST'])
def replay_center():
    """第一步落点分布 + 中心占比 (多代可合并)"""
    v = req_version()
    idx = load_index(v)
    if idx is None:
        return jsonify({'ok': False, 'error': f'{v} 索引未构建'})
    data = request.json or {}
    iters = data.get('iters', [])
    if not iters:
        return jsonify({'ok': False, 'error': '缺 iters'})
    dist = np.zeros(225, dtype=np.int64)
    total = 0
    miss = 0
    for it in iters:
        games = idx['iters'].get(str(it), {}).get('files', [])
        for g in games:
            fm = g.get('first_move')
            if fm is None:
                miss += 1
                continue
            dist[fm[0] * 15 + fm[1]] += 1
            total += 1
    if total == 0:
        return jsonify({'ok': False, 'error': '无有效第一步数据'})
    prob = [round(float(v) / total, 4) for v in dist]
    # 中心占比: 天元 3x3 / 5x5 / 曼哈顿<=4
    def region(mask):
        return round(float(dist[mask].sum()) / total, 4)
    c = 7
    m3 = np.zeros(225, dtype=bool); m5 = np.zeros(225, dtype=bool); mm4 = np.zeros(225, dtype=bool)
    for r in range(15):
        for cc in range(15):
            i = r * 15 + cc
            if abs(r - c) <= 1 and abs(cc - c) <= 1: m3[i] = True
            if abs(r - c) <= 2 and abs(cc - c) <= 2: m5[i] = True
            if abs(r - c) + abs(cc - c) <= 4: mm4[i] = True
    return jsonify({'ok': True, 'total': total, 'miss': miss,
                    'dist': prob,
                    'center3x3': region(m3), 'center5x5': region(m5),
                    'manhattan4': region(mm4),
                    'top5': [{'r': int(j) // 15, 'c': int(j) % 15, 'pct': round(float(dist[j]) / total * 100, 2)}
                             for j in np.argsort(dist)[::-1][:5]]})


@replay_bp.route('/api/replay/search', methods=['POST'])
def replay_search():
    """棋型搜索: 给定 stones(相对棋型), 在多代经验池的每局每局面中搜平移匹配(可选8对称)
    返回匹配 {iter, game, file, step, offset, sym}"""
    v = req_version()
    idx = load_index(v)
    if idx is None:
        return jsonify({'ok': False, 'error': f'{v} 索引未构建'})
    data = request.json or {}
    stones = data.get('stones', [])
    iters = data.get('iters', [])
    sym = bool(data.get('sym', False))
    if not stones or not iters:
        return jsonify({'ok': False, 'error': '缺 stones 或 iters'})
    blacks = [(int(s['r']), int(s['c'])) for s in stones if int(s['v']) == 1]
    whites = [(int(s['r']), int(s['c'])) for s in stones if int(s['v']) == -1]
    if not blacks:
        return jsonify({'ok': False, 'error': '棋型至少含一个黑子'})
    # 相对坐标: 以第一个黑子为原点
    or0, oc0 = blacks[0]
    rel_b = [(r - or0, c - oc0) for r, c in blacks]
    rel_w = [(r - or0, c - oc0) for r, c in whites]
    # 8 对称变换 (旋转+镜像), 用于旋转不变匹配
    def trans8(r, c):
        return [
            (r, c), (-c, r), (-r, -c), (c, -r),
            (r, -c), (c, r), (-r, c), (-c, -r),
        ]
    variants = [rel_b]
    if sym:
        variants = []
        for k in range(8):
            variants.append([trans8(r, c)[k] for r, c in rel_b])
    matches = []
    seen = set()
    limit = data.get('limit', 500)
    for it in iters:
        games = idx['iters'].get(str(it), {}).get('files', [])
        for g in games:
            df = load_game_df(g['file'], v)
            if df is None:
                continue
            boards = [np.asarray(b, dtype=np.int8) for b in df['board'].values]
            # 每局面: 该局面轮到谁 = df['player'], 但棋型颜色是绝对的黑/白
            for step, b in enumerate(boards):
                bset_b = set(map(tuple, np.argwhere(b == 1)))
                bset_w = set(map(tuple, np.argwhere(b == -1)))
                if len(bset_b) < len(blacks) or len(bset_w) < len(whites):
                    continue
                # 对每个黑子位置尝试作为原点
                for br, bc in bset_b:
                    for v_i, rel in enumerate(variants):
                        ok = True
                        for rr, cc in rel:
                            pr, pc = br + rr, bc + cc
                            if not (0 <= pr < 15 and 0 <= pc < 15 and (pr, pc) in bset_b):
                                ok = False
                                break
                        if not ok:
                            continue
                        # 白子匹配
                        wok = True
                        for rr, cc in rel_w:
                            pr, pc = br + rr, bc + cc
                            if not (0 <= pr < 15 and 0 <= pc < 15 and (pr, pc) in bset_w):
                                wok = False
                                break
                        if not wok:
                            continue
                        key = (g['file'], step, br, bc, v_i)
                        if key in seen:
                            continue
                        seen.add(key)
                        matches.append({'iter': int(it), 'file': g['file'],
                                        'game': int(g['game']), 'worker': int(g['worker']),
                                        'step': int(step), 'n_steps': int(len(boards)),
                                        'offset': [int(br), int(bc)], 'sym': int(v_i)})
                        if len(matches) >= limit:
                            return jsonify({'ok': True, 'matches': matches,
                                            'truncated': True, 'count': len(matches)})
    return jsonify({'ok': True, 'matches': matches, 'truncated': False, 'count': len(matches)})
