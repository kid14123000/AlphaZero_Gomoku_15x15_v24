"""
五子棋 AlphaZero 本地对战 Web 应用
支持：人 vs AI / AI vs AI / 人 vs 人
支持纯网络模式（sim_count=0）、MCTS模式、贪心/随机
"""
import json, os, sys, inspect
import importlib.util
import numpy as np
import gym
import boardgame2
import torch
from flask import Flask, request, jsonify, send_from_directory, make_response, redirect
from functools import wraps
import time

app = Flask(__name__, static_folder='static')

# ── 访问密码 ──
# ★ 2026-11-15 用户: "不要密码，我自己看，不需要密码"
#   默认空 = 完全不校验。要恢复口令: GOMOKU_PASSWORD=xxx python app.py
#   （只监听 127.0.0.1，本来就是本机自己看，加口令只是徒增麻烦）
ACCESS_PASSWORD = os.environ.get('GOMOKU_PASSWORD', '')


def require_auth(f):
    """HTTP Basic Auth 验证装饰器 —— ACCESS_PASSWORD 为空时整体直通（无密码模式）"""
    @wraps(f)
    def decorated(*args, **kwargs):
        if not ACCESS_PASSWORD:
            return f(*args, **kwargs)
        auth = request.authorization
        if not auth or auth.password != ACCESS_PASSWORD:
            return make_response(
                'Authorization required',
                401,
                {'WWW-Authenticate': 'Basic realm="Gomoku"'}
            )
        return f(*args, **kwargs)
    return decorated


# 对所有非静态文件路由加锁
app.before_request(lambda: None)  # 占位，实际在路由上加装饰器

# ── 路径 ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
CHECKPOINT_DIR = os.path.join(PARENT_DIR, 'gomoku_checkpoints')

# ── 多版本权重目录与对应网络模块（2026-08-01 新增 v2/v3 支持） ──
# ⚠️ 2026-08-01 根目录整理后只留 v2 脚本，v1/v3 网络脚本归档到 archive_v1/scripts/
# ⚠️ 2026-08-05 新增 v6 主线（简化架构, 新策略）
# ⚠️ 2026-09-12 发行版目录整理 (commit d287bda): v1-v35 主程序 → legacy/versions/, archive_v1/ → legacy/archive_v1/
#    网络脚本查找改为多目录回退，兼容「整理前根目录布局」与「整理后 legacy 布局」
SCRIPT_SEARCH_DIRS = [
    PARENT_DIR,
    os.path.join(PARENT_DIR, 'legacy', 'versions'),
    os.path.join(PARENT_DIR, 'legacy', 'archive_v1', 'scripts'),
    os.path.join(PARENT_DIR, 'server_sync', 'scripts'),
]


def resolve_script(name):
    """在 根目录 / legacy/versions / legacy/archive_v1/scripts / server_sync/scripts 中定位网络脚本"""
    if os.path.isabs(name):
        name = os.path.basename(name)
    for d in SCRIPT_SEARCH_DIRS:
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f'未找到网络脚本 {name}；已搜索: {SCRIPT_SEARCH_DIRS}'
        f' —— 该版本的训练脚本不在本地（常见于刚切主线的新版本：脚本只在服务器上生成）。'
        f'修法: 从服务器拉回来 `scp -P <port> root@<host>:/root/pk_v8/{name} {PARENT_DIR}/`，'
        f'或放进 legacy/versions/；拉回来即对所有依赖它的页面生效（网络前向/摆谱/MCTS/出题/测评）。')


ARCHIVE_SCRIPTS = os.path.join(PARENT_DIR, 'archive_v1', 'scripts')
VERSIONS = [
    # (版本, 权重目录, 网络脚本, 显示名)
    # ★ 2026-11-15: 本仓库只带 v24 系（v24 原版 / v24b）。
    #   原版 gomoku_web 的 v1~v42 全列表已裁掉——那些网络脚本不在本仓库，
    #   而 app.py 在**模块级**就 load_mod_for('v1')，留一条不存在的就会启动即崩。
    #   要恢复某个历史版本：把它的 Gomoku-v0_pt_vNN.py 放进本目录或 ../legacy/versions/ 后加回这里。
    ('v24', 'gomoku_checkpoints_v24', 'Gomoku-v0_pt_v24.py',  'v24'),
    ('v24b', 'gomoku_checkpoints_v24', 'Gomoku-v0_pt_v24b.py', 'v24b'),
]

_mod_cache = {}

def load_mod_for(version):
    """按版本加载对应网络模块（缓存，避免重复 exec_module）"""
    if version in _mod_cache:
        return _mod_cache[version]
    info = next((v for v in VERSIONS if v[0] == version), None)
    if info is None:
        raise ValueError(f'未知版本: {version}')
    mod_path = resolve_script(info[2])
    spec = importlib.util.spec_from_file_location(f'gomoku_mod_{version}', mod_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.device = torch.device('cpu') if torch.cuda.is_available() is False else mod.device
    _mod_cache[version] = mod
    return mod

# 默认加载的网络模块（旧代码写死 v1；本仓库不带 v1）
# 注意：TRAIN_VERSION 在文件后段才定义，这里不能用它 → 直接给 v24b
# gomoku_mod 变量名保留给其它引用
gomoku_mod = load_mod_for('v24b')
from boardgame2 import BLACK, WHITE

# ── 全局状态 ──
G = {
    'board': None,          # 15×15 numpy array
    'current': 1,           # 当前执棋方: BLACK=1, WHITE=-1
    'moves': [],            # [(r,c), ...]
    'over': False,
    'winner': None,
    'env': None,
    'black_agent': None,    # AlphaZeroAgent
    'white_agent': None,    # AlphaZeroAgent
    'config': {},           # 当前对局配置
    'win_history': [],      # 每步后的黑棋胜率（前端画曲线）
}


# ── 工具函数 ──
def _iter_num(f):
    """从权重文件名提取迭代号 (it{N}), weights.pt → 10**9 视为最新"""
    try:
        return int(f.split('it')[1].split('.')[0].split('_')[0])
    except Exception:
        return 10 ** 9


def _version_dir(dirname):
    """版本目录 → 实际存在的目录（支持【归档回退】）。

    🚨 2026-10-01 用户报"web工具怎么选不了 v39 的权重了"的根因:
      轮次归档会把 <槽>/ 改名成 <槽>_round<K>/ → 主目录消失
      → scan_weights() 对不存在目录 `continue` → 该版本权重**一条都列不出来** ✗
      (实测 /api/weights 里 v39 系条目 = 0, 而归档目录里 115 个 it30~it233 权重都在 ✓)
    规则: 主目录在 → 用主目录; 不在 → 回退最新的 <槽>_round* (轮号最大)
      · 只影响"看历史权重", 新一轮重建主目录 (R10+) 后自动回到主目录 ✓
      · 回退是**只读**的 → 不会把新旧轮次数据混写到一个目录 ✗
    """
    d = os.path.join(PARENT_DIR, dirname)
    if os.path.isdir(d):
        return d
    import glob as _glob, re as _re
    cands = []
    for p in _glob.glob(os.path.join(PARENT_DIR, dirname + '_round*')):
        m = _re.search(r'_round(\d+)$', p)
        if m and os.path.isdir(p):
            cands.append((int(m.group(1)), p))
    if cands:
        return sorted(cands)[-1][1]
    return d


def scan_weights():
    """扫描所有版本目录（v1/v2/v3/v6）下的权重文件（仅 PyTorch 格式）
    排序: 版本 + 迭代号数字序 (it9 < it90 < it91), 修复字典序 bug (it100 < it2)"""
    weights = []
    for version, dirname, _, disp in VERSIONS:
        d = _version_dir(dirname)          # ★ 归档回退 (2026-10-01)
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            if f.endswith('.pt') and not f.startswith('.'):
                label = '最新权重' if f == 'weights.pt' else f.replace('.pt', '').replace('weights_it', '代')
                # 文件名可能已带版本后缀（如 weights_it5_v3.pt），label 去重
                label = label.replace(f'_{disp}', '')
                weights.append({'file': f, 'label': f'{label} [{disp}]', 'path': os.path.join(d, f), 'version': version})
    # ★ 2026-11-15: 同一目录可能被多个版本条目指向（v24 与 v24b 共用 gomoku_checkpoints_v24）
    #   → 按 path 去重，否则同一个权重在选单里出现两次
    _seen, _uniq = set(), []
    for w in weights:
        if w['path'] in _seen:
            continue
        _seen.add(w['path'])
        _uniq.append(w)
    _uniq.sort(key=lambda w: (w['file'] != 'weights.pt', w['version'], _iter_num(w['file'])))
    return _uniq


def load_metadata():
    """加载 metadata 获取默认网络参数"""
    meta_path = os.path.join(CHECKPOINT_DIR, 'metadata.json')
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            return json.load(f)
    return {'config': {
        'sim_count': 50,
        'net_kwargs': {'conv_filters': [128], 'residual_filters': [[128, 128]] * 8, 'policy_filters': [128]}
    }}


def _resolve_weight(weight_file):
    """解析权重标识 → (version, 权重文件路径)。
    支持 'version|filename' 组合（新版前端）或纯文件名（兼容旧版，按目录猜测版本）。
    """
    if weight_file and '|' in weight_file:
        version, fname = weight_file.split('|', 1)
    else:
        fname = weight_file
        version = None
        # 旧格式：按文件名所在目录猜测版本
        for v, dirname, _, _ in VERSIONS:
            if os.path.exists(os.path.join(_version_dir(dirname), fname)):
                version = v
                break
        if version is None:
            version = 'v1'
    d = _version_dir(next(v[1] for v in VERSIONS if v[0] == version))   # ★ 归档回退 (2026-10-01)
    wpath = os.path.join(d, fname) if not os.path.isabs(fname) else fname
    return version, wpath


def make_agent(weight_file, sim_count=50, board_size=15):
    """创建 AlphaZeroAgent 并加载指定权重（支持 v1/v2/v3 多版本）
    board_size=9: v16 9×9 网络用 KInARowEnv(9)（其余 15×15 用 Gomuku-v0）"""
    meta = load_metadata()
    net_kwargs = meta.get('config', {}).get('net_kwargs',
        {'conv_filters': [128], 'residual_filters': [[128, 128]] * 8, 'policy_filters': [128]})

    version, wpath = _resolve_weight(weight_file)
    mod = load_mod_for(version)

    if board_size == 9:
        import boardgame2
        env = boardgame2.KInARowEnv(board_shape=9, target_length=5)
    else:
        env = gym.make('Gomuku-v0')

    if wpath.endswith('.pt'):
        # PyTorch 引擎（按版本加载对应网络结构）
        agent = mod.AlphaZeroAgent(env=env, kwargs=net_kwargs,
                                   sim_count=max(sim_count, 1),
                                   batches=1, batch_size=8)
        if os.path.exists(wpath):
            state_dict = torch.load(wpath, map_location=mod.device)
            agent.net.load_state_dict(state_dict)
            agent.net.to(mod.device)
            agent.reset_mcts()
        agent.version = version  # 记录版本，供 value_map key 格式判断
    elif wpath.endswith(('.h5', '.hdf5')):
        # TF 引擎（兼容 .weights.h5 等旧格式，仅 v1）
        tf_spec = importlib.util.spec_from_file_location('gomoku_tf',
            resolve_script('Gomoku-v0_tf.py'))
        tf_mod = importlib.util.module_from_spec(tf_spec)
        tf_spec.loader.exec_module(tf_mod)
        agent = tf_mod.AlphaZeroAgent(env=env, kwargs=net_kwargs,
                                      sim_count=max(sim_count, 1),
                                      batches=1, batch_size=8)
        if os.path.exists(wpath):
            agent.net.load_weights(wpath)
            agent.reset_mcts()
        agent.version = 'v1'  # TF 引擎仅 v1
    else:
        # ★ 2026-09-26: 权重标识写错(如 'v38|it77' 而非 'v38|weights_it77_v38.pt')时,
        #   旧代码会静默掉进 TF 分支 → 报 "No module named 'tensorflow'" 误导排查方向 ✗
        raise FileNotFoundError(
            '无法加载权重 %r → 解析为 %s（文件不存在，或既非 .pt 也非 .h5）。'
            '权重标识应写成「版本|文件名」，例如 v38|weights_it77_v38.pt'
            % (weight_file, wpath))
    return agent, env


def _agent_search(agent, canonical, player=None, last_move=None):
    """统一走 agent.search，按签名自动带上 player / last_move

    🚨 2026-09-27 用户发现的输入口径地雷:
      v8+ 网络输入是 **4 plane** —— plane[3] = last_move 那一手的位置
      (见 Gomoku-v0_pt_v*.py 的 make_input: out[3][last_move] = 1.0)
      而 web 各页面**从来没传过 last_move** ✗ → plane3 恒 0 → 模型看到训练时
      没见过的输入分布。实测同一局面值函数能差 0.26 (26pp) → MCTS 访问分布跟着偏 ✗
    旧版(v12/v14/v18 等)的 search 没有这两个参数 → 用签名探测"有才传";
    不用 try/except TypeError 兜底(那会把 search 内部的真 TypeError 一起吞掉 ✗)
    """
    kw = {'prior_noise': False}
    try:
        _params = inspect.signature(agent.search).parameters
    except (TypeError, ValueError):
        _params = {}
    if player is not None and 'player' in _params:
        kw['player'] = player
    if last_move is not None and 'last_move' in _params:
        kw['last_move'] = last_move
    return agent.search(canonical, **kw)


def _agent_predict(agent, board_batch, players=None, last_moves=None):
    """统一走 agent.predict（player / last_move 同 _agent_search 的口径说明）

    ⚠️ predict 的形参是复数 last_moves, 且必须按 batch 给: (N,2) 或 None
    """
    kw = {}
    try:
        _params = inspect.signature(agent.predict).parameters
    except (TypeError, ValueError):
        _params = {}
    if players is not None and 'players' in _params:
        kw['players'] = players
    if last_moves is not None and 'last_moves' in _params:
        kw['last_moves'] = last_moves
    return agent.predict(board_batch, **kw)


def _lm_of(stones):
    """从落子序列(有序)取"最后一手" → (r,c); 空则 None

    ⚠️ 单一静态局面(手动摆盘/单帧棋谱)推不出 last_move → 必须返回 None ✗ 不能瞎猜
    """
    try:
        if not stones:
            return None
        s = stones[-1]
        if isinstance(s, dict):
            return (int(s['r']), int(s['c']))
        return (int(s[0]), int(s[1]))
    except Exception:
        return None


def _agent_decide(agent, obs, **kw):
    """统一走 agent.decide（v8+ 才有 decide; last_move 同 _agent_search 的口径）

    decide 的签名: decide(obs, ..., temperature, force_center, move_no, last_move, ...)
    ⚠️ move_no/force_center 是训练期行为(温度调度/随机开局), 评估不要传 ✗
    """
    try:
        _params = inspect.signature(agent.decide).parameters
    except (TypeError, ValueError):
        _params = {}
    if kw.get('last_move') is not None and 'last_move' in _params:
        pass                       # 有才传
    else:
        kw.pop('last_move', None)
    return agent.decide(obs, **kw)


def fast_move(agent, board, player, greedy=True, temperature=0.3, last_move=None):
    """纯网络一步推理（无 MCTS）"""
    import torch
    canonical = player * board
    valid = G['env'].get_valid((board, player))
    # 统一通过 agent.predict 走 PyTorch 或 TF
    if hasattr(agent, 'predict'):
        # v18: 传 players=player → ch2 正确编码当前玩家颜色 (否则白视角推理退化成黑视角, 2026-08-22)
        # ★ 2026-09-27: 同时传 last_moves → plane3 = 上一手 (不传恒 0, 与训练输入分布不符 ✗)
        _lms = np.array([last_move]) if last_move is not None else None
        pis, vs = _agent_predict(agent, canonical[np.newaxis], players=player, last_moves=_lms)
        pi = np.exp(np.asarray(pis[0]))  # ⚠️ 修复: net输出log概率, 转概率 (2026-08-10)
        v = float(np.asarray(vs[0]).reshape(-1)[0])
    else:
        pis, vs = agent.net(canonical[np.newaxis], training=False)
        pi = pis[0].numpy()
        v = float(vs[0, 0])
    # ⚠️ pi 可能是 (225,) 展平（v2/v3 predict 返回 1D），valid 是 (15,15) → 统一 reshape
    pi = np.asarray(pi).reshape(valid.shape)
    masked = pi * valid
    if masked.sum() <= 0:
        masked = valid
    if greedy or temperature <= 0.05:
        idx = np.argmax(masked)
    else:
        # 温度采样: 概率 ^(1/T) 后归一化 (T=1 按原分布, T→0 趋近贪心)
        masked = masked ** (1.0 / temperature)
        masked = masked / masked.sum()
        idx = np.random.choice(masked.size, p=masked.ravel())
    return np.unravel_index(idx, masked.shape), v


def mcts_move(agent, board, player, greedy=True, temperature=0.3, last_move=None):
    """带 MCTS 搜索的决策（决策不加训练噪声; greedy=argmax, 非greedy用低温采样保多样性）"""
    obs = (board.copy(), player)
    # ★ 2026-09-27: 传 last_move → plane3 = 上一手 (decide 内部同样走 make_input) ✗不传会偏
    action, prob = _agent_decide(agent, obs, return_prob=True, greedy=greedy, noise=False,
                                 temperature=temperature, last_move=last_move)
    # 取出动作对应的概率值
    p = float(prob[action[0], action[1]]) if prob is not None else 0.0
    return action, p


def _trace_initial_moves(weight_file, board, player, n_trace=30, snap_every=5, board_size=15,
                         last_move=None):
    """轻量二次追踪: 用独立 agent（加载同一权重）对同一局面重跑 n_trace 次 MCTS（带噪声），
    记录: initial=每次模拟的根节点选点, snapshots=每 snap_every 次的 top3 访问分布。
    ⚠️ 必须用独立 agent——不能 reset 真实决策的 MCTS 树（compute_value_map 还依赖它）。
    返回 dict 或 None"""
    try:
        import numpy as np
        version, wpath = _resolve_weight(weight_file)
        mod = load_mod_for(version)
        meta = load_metadata()
        net_kwargs = meta.get('config', {}).get('net_kwargs',
            {'conv_filters': [128], 'residual_filters': [[128, 128]] * 8, 'policy_filters': [128]})
        # v16/v17 是 9×9: env 必须同尺寸, 否则 fc 维度不匹配加载崩 (2026-08-22 实踩)
        if board_size == 9:
            import boardgame2
            env = boardgame2.KInARowEnv(board_shape=9, target_length=5)
        else:
            env = gym.make('Gomuku-v0')
        t_agent = mod.AlphaZeroAgent(env=env, kwargs=net_kwargs, sim_count=1, batches=1, batch_size=8)
        sd = torch.load(wpath, map_location=mod.device)
        t_agent.net.load_state_dict(sd)
        t_agent.net.to(mod.device)
        t_agent.reset_mcts()
        canonical = (player * board).astype(np.int8)
        root_moves = []
        snapshots = []
        bs = board.shape[0]  # 9×9 (v16) / 15×15
        # 2026-08-29 论文式: 根噪声一次 + 后续共用 (替代每 sim 重采样)
        _paper_noise_init(t_agent, canonical, player=player, last_move=last_move)
        for i in range(n_trace):
            # v18: 传 player → ch2 正确视角; ★ plane3 = 上一手 (2026-09-27)
            _agent_search(t_agent, canonical, player=player, last_move=last_move)
            s = canonical.tobytes()
            count = t_agent.count.get(s)
            valid = t_agent.valid.get(s)
            if count is not None and valid is not None:
                cntv = count * valid
                # 本轮根选点: 访问数最大的点（本轮它+1）
                idx = int(np.argmax(cntv.ravel()))
                r, c = divmod(idx, bs)
                root_moves.append({'r': int(r), 'c': int(c)})
                # 快照
                if (i + 1) % snap_every == 0 or i == n_trace - 1:
                    q = t_agent.q.get(s)
                    top3 = []
                    for j in np.argsort(cntv.ravel())[::-1][:3]:
                        rr, cc = divmod(int(j), bs)
                        top3.append({'r': int(rr), 'c': int(cc),
                                     'visits': int(cntv.ravel()[j]),
                                     'q': round(float(q.ravel()[j]), 3) if q is not None else 0})
                    snapshots.append({'step': i + 1, 'top': top3})
        return {'initial': root_moves, 'snapshots': snapshots}
    except Exception as e:
        print(f'[trace-mcts] {e}')
        return None


def _find_threat_zone(board, opp_player, min_len=3):
    """扫描棋盘找对手 ≥min_len 连子所在格子（活三/冲四威胁区），返回 [(r,c),...]"""
    try:
        import numpy as np
        H = W = board.shape[0]
        dirs = [(1, 0), (0, 1), (1, 1), (1, -1)]
        pts = set()
        for r in range(H):
            for c in range(W):
                if board[r, c] != opp_player:
                    continue
                for dr, dc in dirs:
                    cnt = 1
                    for sgn in [1, -1]:
                        rr, cc = r + dr * sgn, c + dc * sgn
                        while 0 <= rr < H and 0 <= cc < W and board[rr, cc] == opp_player:
                            cnt += 1
                            rr += dr * sgn
                            cc += dc * sgn
                    if cnt >= min_len:
                        pts.add((int(r), int(c)))
                        for sgn in [1, -1]:
                            rr2, cc2 = r + dr * sgn, c + dc * sgn
                            while 0 <= rr2 < H and 0 <= cc2 < W and board[rr2, cc2] == opp_player:
                                pts.add((int(rr2), int(cc2)))
                                rr2 += dr * sgn
                                cc2 += dc * sgn
        return sorted(pts)
    except Exception as e:
        print(f'[find-threat] {e}')
        return []


def _trace_value_flow(weight_file, board, player, threat_zone):
    """Q值异常逆向追踪: 传导链拆解——逐层威胁区激活比值 + value头贡献分解 + 扰动实验。
    回答 'value 输出 X 是怎么传导过来的 / 哪层开始错'。
    ⚠️ 独立 agent 加载同一权重（不动真实决策的 MCTS 树）。返回 dict 或 None"""
    try:
        import numpy as np
        import torch
        import torch.nn.functional as F
        version, wpath = _resolve_weight(weight_file)
        mod = load_mod_for(version)
        meta = load_metadata()
        net_kwargs = meta.get('config', {}).get('net_kwargs',
            {'conv_filters': [128], 'residual_filters': [[128, 128]] * 8, 'policy_filters': [128]})
        env = gym.make('Gomuku-v0')
        t_agent = mod.AlphaZeroAgent(env=env, kwargs=net_kwargs, sim_count=1, batches=1, batch_size=8)
        sd = torch.load(wpath, map_location=mod.device)
        t_agent.net.load_state_dict(sd)
        t_agent.net.to(mod.device)
        net = t_agent.net
        net.eval()   # ⚠️ 必须 eval（BN train 模式单样本输出是垃圾）
        canonical = (player * board).astype(np.float32)
        # v8 4通道输入适配 (make_input 存在 = v8)
        # 🐛 2026-08-23: 传 player → v18 ch2 正确视角 (原漏传 = 白视角局面 ch2 恒 1 错乱)
        if hasattr(mod, 'make_input'):
            try:
                x = torch.tensor(mod.make_input(canonical, player=player)).unsqueeze(0)
            except TypeError:
                x = torch.tensor(mod.make_input(canonical)).unsqueeze(0)
        else:
            x = torch.tensor(canonical).unsqueeze(0).unsqueeze(0)
        # 1) 逐层前向
        h = F.relu(net.bn_input(net.conv_input(x)))
        acts = [('bn_input', h.detach())]
        for i, blk in enumerate(net.res_blocks):
            h = blk(h)
            acts.append((f'res{i}', h.detach()))
        v_conv = F.relu(net.value_bn(net.value_conv(h)))
        acts.append(('value_bn', v_conv.detach()))
        # 2) value logit（版本自适应 v6 单层 / v2 双层）
        vbn_flat = v_conv.reshape(v_conv.shape[0], -1)
        if hasattr(net, 'value_fc2'):
            logit = net.value_fc2(F.relu(net.value_fc1(vbn_flat))).reshape(-1)[0]
        else:
            logit = net.value_fc(vbn_flat).reshape(-1)[0]
        value = float(torch.tanh(logit).detach())
        logit_v = float(logit.detach())
        # 3) 威胁区激活比值
        thr_r = [p[0] for p in threat_zone]
        thr_c = [p[1] for p in threat_zone]
        layers = []
        for name, a in acts:
            thr = float(a[:, :, thr_r, thr_c].mean())
            full = float(a.mean())
            layers.append({'name': name, 'ratio': round(thr / full, 2) if full > 0 else 0})
        # 4) value 头贡献分解（梯度灵敏度 grad·vbn）
        grad_vbn = torch.autograd.grad(logit, v_conv, retain_graph=True, create_graph=False)[0].detach()
        contrib = (grad_vbn * v_conv.detach()).squeeze(0)   # (C,15,15)
        thr_contrib = float(contrib[:, thr_r, thr_c].sum())
        full_contrib = float(contrib.sum())
        thr_sens = float(grad_vbn[:, :, thr_r, thr_c].mean())
        # 5) 扰动实验：删掉对手威胁连子 → value 变化（威胁惩罚强度）
        opp = -player
        thr_pts = [(r, c) for r, c in threat_zone if board[r, c] == opp]
        b2 = canonical.copy()
        for r, c in thr_pts:
            b2[r, c] = 0.0
        with torch.no_grad():
            if hasattr(mod, 'make_input'):
                try:
                    x2 = torch.tensor(mod.make_input(b2, player=player)).unsqueeze(0)
                except TypeError:
                    x2 = torch.tensor(mod.make_input(b2)).unsqueeze(0)
            else:
                x2 = torch.tensor(b2).unsqueeze(0).unsqueeze(0)
            h2 = F.relu(net.bn_input(net.conv_input(x2)))
            for blk in net.res_blocks:
                h2 = blk(h2)
            vc2 = F.relu(net.value_bn(net.value_conv(h2)))
            if hasattr(net, 'value_fc2'):
                logit2 = net.value_fc2(F.relu(net.value_fc1(vc2.reshape(vc2.shape[0], -1)))).reshape(-1)[0]
            else:
                logit2 = net.value_fc(vc2.reshape(vc2.shape[0], -1)).reshape(-1)[0]
            value2 = float(torch.tanh(logit2).detach())
        d_threat = value - value2
        return {
            'value': round(value, 3),
            'logit': round(logit_v, 3),
            'threat_n': len(thr_pts),
            'layers': layers,
            'fc': {'threat_contrib': round(thr_contrib, 3),
                   'full_contrib': round(full_contrib, 3),
                   'threat_sens': round(thr_sens, 4)},
            'perturb': {'d_threat': round(d_threat, 3)},
        }
    except Exception as e:
        print(f'[trace-value-flow] {e}')
        return None


def collect_decision_info(agent, board, player, move_r, move_c, greedy=True, last_move=None):
    """采集 AI 一步棋的完整决策过程（MCTS 搜索后、reset 前调用）。
    返回 dict: 网络先验 top5 / value 评估 / MCTS count-Q top5 / 决策置信度 / 最优候选 / 搜索动态"""
    try:
        import numpy as np
        canonical = (player * board).astype(np.int8)
        s = canonical.tobytes()
        count = agent.count.get(s)
        q = agent.q.get(s)
        policy = agent.policy.get(s)
        valid = agent.valid.get(s)
        if count is None or q is None or policy is None or valid is None:
            return None
        info = {}
        bs = board.shape[0]  # 9×9 (v16) / 15×15
        info['sim_count'] = int(count.sum())
        info['greedy'] = bool(greedy)
        # 网络先验 top5 (policy)
        masked = policy * valid
        top_idx = np.argsort(masked.ravel())[::-1][:5]
        info['policy_top'] = []
        for idx in top_idx:
            if masked.ravel()[idx] > 0:
                r, c = divmod(idx, bs)
                info['policy_top'].append({'r': int(r), 'c': int(c),
                                           'prob': round(float(masked.ravel()[idx]), 4)})
        # MCTS count/Q top5
        cntv = count * valid
        top_idx2 = np.argsort(cntv.ravel())[::-1][:5]
        info['mcts_top'] = []
        for idx in top_idx2:
            if cntv.ravel()[idx] > 0:
                r, c = divmod(idx, bs)
                info['mcts_top'].append({'r': int(r), 'c': int(c),
                                         'visits': int(count.ravel()[idx]),
                                         'q': round(float(q.ravel()[idx]), 3)})
        # 最优候选 = 访问最多的点（贪心会下的）
        best_idx = int(np.argmax(cntv.ravel()))
        br, bc = divmod(best_idx, bs)
        info['best'] = {'r': int(br), 'c': int(bc),
                        'visits': int(count.ravel()[best_idx]),
                        'q': round(float(q.ravel()[best_idx]), 3)}
        # 决策点信息
        info['move'] = {'r': int(move_r), 'c': int(move_c)}
        info['move_visits'] = int(count[move_r, move_c])
        info['move_q'] = round(float(q[move_r, move_c]), 3)
        info['move_policy'] = round(float(policy[move_r, move_c]), 4)
        info['confidence'] = round(100 * count[move_r, move_c] / max(1, int(count.sum())), 1)
        info['is_best'] = (int(move_r) == br and int(move_c) == bc)
        # value 评估: 用网络直接预测当前局面 (从当前玩家视角)
        if hasattr(agent, 'predict'):
            # v18: 传 players=player → ch2 正确视角; ★ plane3 = 上一手 (2026-09-27)
            pis, vs = _agent_predict(agent, canonical[np.newaxis], players=player,
                                     last_moves=(np.array([last_move]) if last_move is not None else None))
            raw = float(np.asarray(vs[0]).reshape(-1)[0])
        else:
            import torch
            with torch.no_grad():
                p, v = agent.net(canonical[np.newaxis])
            raw = float(v.reshape(-1)[0])
        info['value'] = round(raw, 3)
        # 搜索动态追踪（独立 agent，不动真实树）——展示"初始乱撞→收敛"过程
        if G.get('config', {}).get('black_weight') or G.get('config', {}).get('white_weight'):
            # 用当前走棋方的权重做追踪
            wf = G['config'].get('black_weight') if player == BLACK else G['config'].get('white_weight')
            if wf:
                info['trace'] = _trace_initial_moves(wf, board, player, n_trace=30, snap_every=5,
                                                     board_size=G.get('board_size', 15),
                                                     last_move=last_move)
        # 候选点评分: MCTS top5 ∪ policy top5 的落子后 value 评估（当前玩家视角）
        # 暴露 value 对每个候选点的区分度——全高=value 无区分度, 有高有低=搜索/先验问题
        try:
            cand = {}
            for x in info.get('mcts_top', [])[:5]:
                cand[(x['r'], x['c'])] = {'r': x['r'], 'c': x['c'], 'visits': x['visits'],
                                          'q': x['q'], 'policy': None}
            for x in info.get('policy_top', [])[:5]:
                k = (x['r'], x['c'])
                if k in cand:
                    cand[k]['policy'] = x['prob']
                else:
                    cand[k] = {'r': x['r'], 'c': x['c'], 'visits': 0, 'q': 0.0, 'policy': x['prob']}
            for k, v in cand.items():
                b2 = board.copy()
                b2[v['r'], v['c']] = player
                next_player = -player
                c2 = (next_player * b2).astype(np.int8)
                # v18: 候选落子后是 next_player 视角 → ch2 传 next_player
                # ★ plane3 = 刚落的这个候选 (子局面自己的 last_move, 不是父局的) ✓
                _, vv = _agent_predict(agent, c2[np.newaxis], players=next_player,
                                       last_moves=np.array([[v['r'], v['c']]]))
                v['value_after'] = round(-float(np.asarray(vv[0]).reshape(-1)[0]), 3)
            info['candidate_scores'] = sorted(cand.values(), key=lambda x: -x['visits'])
        except Exception as e:
            print(f'[candidate-scores] {e}')
        # Q值异常逆向追踪: 对手有活三/冲四威胁 + 网络却判己方大优(>0.5) → 启动传导链拆解
        if info.get('value') is not None and info['value'] > 0.5:
            opp = -player
            threat_zone = _find_threat_zone(board, opp)
            if threat_zone:
                wf = G['config'].get('black_weight') if player == BLACK else G['config'].get('white_weight')
                if wf:
                    info['value_flow'] = _trace_value_flow(wf, board, player, threat_zone)
        return info
    except Exception as e:
        print(f'[collect-decision] {e}')
        return None


def ai_move(sim_count, greedy, temperature=0.3):
    """执行 AI 走棋，返回 (row, col, prob_or_value, is_fast)"""
    board = G['board'].copy()
    player = G['current']
    agent = G['black_agent'] if player == BLACK else G['white_agent']

    is_fast = (sim_count == 0)
    # ★ 输入 plane3 = 上一手 (2026-09-27 修): 此刻 G['moves'] 里最后一项就是对手刚走的那手 ✓
    _last_mv = _lm_of(G['moves'])
    if is_fast:
        (r, c), val = fast_move(agent, board, player, greedy=greedy, temperature=temperature,
                                last_move=_last_mv)
        info = val
        decision_info = None
    else:
        (r, c), prob = mcts_move(agent, board, player, greedy=greedy, temperature=temperature,
                                 last_move=_last_mv)
        info = prob
        # ⚠️ 必须在 compute_value_map (reset_mcts) 之前采集
        decision_info = collect_decision_info(agent, board, player, r, c, greedy=greedy,
                                              last_move=_last_mv)

    # 执行落子
    action = np.array([r, c])
    obs, reward, terminated, truncated, info_env = G['env'].step(action)
    G['board'] = obs[0]
    G['moves'].append((int(r), int(c)))
    G['current'] = -player

    if terminated or truncated:
        G['over'] = True
        G['winner'] = G['env'].get_winner(obs) if terminated else None

    # 保存决策过程（供 API 返回）
    G['last_decision'] = decision_info

    # 走完棋后计算价值地图 + 记录胜率
    if not G['over']:
        G['value_map'] = compute_value_map()
    else:
        G['value_map'] = None
    record_win_rate()

    return int(r), int(c), info, is_fast


def human_move(r, c):
    """人类走棋验证并执行"""
    board = G['board']
    player = G['current']
    env = G['env']

    if not (0 <= r < 15 and 0 <= c < 15):
        return False, '位置超出棋盘'
    if board[r, c] != 0:
        return False, '该位置已有棋子'
    valid = env.get_valid((board, player))
    if valid[r, c] != 1:
        return False, '无效位置'

    action = np.array([r, c])
    obs, reward, terminated, truncated, info = env.step(action)
    G['board'] = obs[0]
    G['moves'].append((r, c))
    G['current'] = -player

    if terminated or truncated:
        G['over'] = True
        G['winner'] = env.get_winner(obs) if terminated else None
    # 人类走完后也计算价值地图 + 记录胜率
    if not G['over']:
        G['value_map'] = compute_value_map()
    else:
        G['value_map'] = None
    record_win_rate()
    return True, None


def reset_mcts_all():
    """重置所有 agent 的 MCTS 树"""
    for agent in [G['black_agent'], G['white_agent']]:
        if agent:
            agent.reset_mcts()


def compute_value_map():
    """
    对当前局面跑MCTS搜索，返回每个空位的Q值。
    Q = value/count，始终从人类玩家视角（或黑棋视角）的胜率估计。
    返回 15×15 数组，有棋子的位置为 None。
    """
    board = G['board']
    if board is None:
        return None

    player = G['current']
    agent = G['black_agent'] if player == BLACK else G['white_agent']
    if agent is None:
        agent = G['black_agent'] or G['white_agent']
    if agent is None:
        return None

    search_sim = max(1, min(800, G['config'].get('value_sim', 50)))

    import boardgame2
    canonical = player * board
    agent.reset_mcts()
    # ⚠️ prior_noise=True：v2 policy 均匀 + False 时 UCB 全平锁死首点，热力图只剩几个格子
    # v18: 传 player → ch2 正确视角 (否则白视角 value_map 退化成黑视角)
    # ★ plane3 = 上一手 (2026-09-27): value_map 也是模型推理, 根先验同样要 plane3 ✓
    _last_mv_map = _lm_of(G.get('moves') or [])
    # 2026-08-29 论文式: 根噪声一次 + 共用 (替代每 sim 重采样)
    _paper_noise_init(agent, canonical, player=player, last_move=_last_mv_map)
    for _ in range(search_sim):
        _agent_search(agent, canonical, player=player, last_move=_last_mv_map)

    # ⚠️ key 格式必须按版本：v1 用 strfboard(str)，v2/v3 用 tobytes(bytes)
    # 用错格式 q_map=None → 热力图消失（2026-08-02 实踩：切 v2 权重后热力图没了）
    ver = getattr(agent, 'version', 'v1')
    s = boardgame2.strfboard(canonical) if ver == 'v1' else canonical.tobytes()
    q_map = agent.q.get(s)
    count = agent.count.get(s)

    if q_map is None or count is None:
        return None

    # 确定人类玩家视角（如果人类下黑棋就始终从黑棋看，反之亦然）
    cfg = G['config']
    human_player = None
    if cfg.get('black_type') == 'human':
        human_player = BLACK
    elif cfg.get('white_type') == 'human':
        human_player = WHITE

    # 如果不是人类视角，翻转 Q 值
    need_flip = (human_player is not None) and (player != human_player)

    bs = board.shape[0]  # 9×9 (v16) / 15×15
    vmap = np.full((bs, bs), None, dtype=object)
    for idx in range(bs * bs):
        if count.flat[idx] > 0:
            q = float(q_map.flat[idx])
            if need_flip:
                q = -q
            r, c = divmod(idx, bs)
            vmap[r, c] = round(q, 3)
    return vmap.tolist()


def build_state():
    """构建前端所需的完整状态"""
    if G['board'] is None:
        return None
    stones = []
    board = G['board']
    for i, (r, c) in enumerate(G['moves']):
        p = int(board[r, c])
        stones.append({'r': int(r), 'c': int(c), 'p': p, 'n': i + 1})
    return {
        'stones': stones,
        'moves': [{'r': int(r), 'c': int(c)} for r, c in G['moves']],
        'current': G['current'],
        'move_count': len(G['moves']),
        'over': G['over'],
        'winner': int(G['winner']) if G['winner'] is not None else None,
        'value_map': G.get('value_map'),
        'win_history': G.get('win_history', []),
    }


def reset_game():
    """重置全局状态"""
    G['board'] = None
    G['current'] = BLACK
    G['moves'] = []
    G['over'] = False
    G['winner'] = None
    G['env'] = None
    G['black_agent'] = None
    G['white_agent'] = None
    G['config'] = {}
    G['win_history'] = []
    G['board_size'] = 15  # v16 9×9 对弈时由 api_start 覆盖


def _agent_side_wr(agent, board, color, last_move=None):
    """用指定 agent 的 value 头预测【指定颜色】的胜率 (color=1 黑 / -1 白)。

    🎯 2026-10-01 用户: 棋力测评页要"我们自己权重的 value 胜率曲线"
    口径 (与对弈页黑/白胜率同源, 只是视角参数化):
      · 干净前向 —— 只跑 value 头, 不走 MCTS、不加 Dirichlet 噪声
      · 视角 = color: canonical = color*board, players=color → 输出即"该色占优程度"
      · ★ plane3 = 上一手 (不传 → 与训练输入分布不一致, 实测值差可达 0.26)
      · 返回 0~100 的百分点; 权重缺失/异常返回 None
    """
    if agent is None:
        return None
    try:
        import torch
        canonical = color * board
        if hasattr(agent, 'predict'):
            # v18: 固定 color 视角 → ch2 传 color; ★ plane3 = 上一手 (2026-09-27)
            pis, vs = _agent_predict(agent, canonical[np.newaxis], players=color,
                                     last_moves=(np.array([last_move]) if last_move is not None else None))
            raw = float(np.asarray(vs[0]).reshape(-1)[0])
        else:
            with torch.no_grad():
                p, v = agent.net(canonical[np.newaxis])
            raw = float(v.reshape(-1)[0])
        wr = (raw + 1) / 2
        return round(max(0, min(1, wr)) * 100, 1)
    except Exception as e:
        print(f'[win-rate] {e}')
        return None


def _agent_black_wr(agent, board, last_move=None):
    """用指定 agent 的 value 头预测黑棋胜率（黑视角 value → 黑胜率）。"""
    return _agent_side_wr(agent, board, BLACK, last_move=last_move)


def record_win_rate():
    """每步后分别用黑/白 agent 的 value 头独立预测黑方、白方胜率，追加到 win_history。
    两条曲线各自用各自权重评估（黑 agent→黑胜率, 白 agent→白胜率），不强制互补。"""
    if G['board'] is None or G['over']:
        return
    board = G['board']
    ba = G.get('black_agent')
    wa = G.get('white_agent')
    if ba is None and wa is None:
        return
    _lm_cur = _lm_of(G.get('moves') or [])
    black_wr = _agent_black_wr(ba, board, last_move=_lm_cur) if ba is not None else None
    white_wr = _agent_black_wr(wa, board, last_move=_lm_cur) if wa is not None else None
    # 缺一方时用另一方补（如白是人类无 agent，白胜率=100-黑）
    if black_wr is None and white_wr is not None:
        black_wr = round(100 - white_wr, 1)
    if white_wr is None and black_wr is not None:
        white_wr = round(100 - black_wr, 1)
    if black_wr is None and white_wr is None:
        return
    G['win_history'].append({'step': len(G['moves']), 'black_wr': black_wr,
                             'white_wr': white_wr})


# ── API 路由 ──

@app.route('/')
@require_auth
def index():
    resp = make_response(send_from_directory('static', 'index.html'))
    resp.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    resp.headers['Pragma'] = 'no-cache'
    resp.headers['Expires'] = '0'
    return resp


@app.route('/api/weights')
@require_auth
def api_weights():
    meta = load_metadata()
    default_sim = meta.get('config', {}).get('sim_count', 50)
    return jsonify({
        'weights': scan_weights(),
        'default_sim': default_sim,
        # ★ 当前活跃槽 (2026-09-27): 各页面用它决定默认版本, 换槽不用再改页面 ✗
        'active': TRAIN_VERSION,
        'active_compare': COMPARE_VERSION,
        'active_goal': COACH_GOAL_RATE,
    })


def _net_section_stats(path, version):
    """对单个权重文件算三段拆解指标（共享主干/Policy头/Value头 的 L2/std）。
    返回 dict：{'main': {...}, 'policy': {...}, 'value': {...}} 或 None（失败）"""
    try:
        mod = load_mod_for(version)
        net = mod.AlphaZeroNet(conv_channels=128, residual_blocks=8)
        sd = torch.load(path, map_location='cpu', weights_only=False)
        if all(k.startswith('net.') for k in sd):
            sd = {k[4:]: v for k, v in sd.items()}
        net.load_state_dict(sd, strict=False)

        secs = {'main': {'l2': 0.0, 'std_sum': 0.0, 'n': 0},
                'policy': {'l2': 0.0, 'std_sum': 0.0, 'n': 0},
                'value': {'l2': 0.0, 'std_sum': 0.0, 'n': 0}}
        for k, v in sd.items():
            if 'running_' in k or 'num_batches' in k or v.numel() == 1:
                continue
            vf = v.float()
            l2 = float((vf ** 2).sum()) ** 0.5
            std = float(vf.std())
            if 'policy' in k:
                sec = 'policy'
            elif 'value' in k:
                sec = 'value'
            elif 'res_blocks' in k:
                sec = 'main'
            else:
                sec = 'main'
            secs[sec]['l2'] += l2
            secs[sec]['std_sum'] += std
            secs[sec]['n'] += 1
        out = {}
        for sec, d in secs.items():
            out[sec] = {
                'l2': round(d['l2'], 2),
                'std': round(d['std_sum'] / d['n'], 4) if d['n'] else 0,
            }
        return out
    except Exception as e:
        print(f'[net-trend] {path}: {e}')
        return None


def _net_section_movement(prev_sd, cur_sd):
    """计算两个 state_dict 之间的三段移动距离（逐元素差值 L2）。
    返回 {'main':..,'policy':..,'value':..} 或 None（key 不匹配）"""
    try:
        secs = {'main': 0.0, 'policy': 0.0, 'value': 0.0}
        for k in cur_sd:
            if k not in prev_sd:
                continue
            if 'running_' in k or 'num_batches' in k:
                continue
            d = (cur_sd[k].float() - prev_sd[k].float())
            if 'policy' in k:
                sec = 'policy'
            elif 'value' in k:
                sec = 'value'
            else:
                sec = 'main'
            secs[sec] += float((d ** 2).sum())
        out = {}
        for sec, s in secs.items():
            out[sec] = round(s ** 0.5, 4)
        return out
    except Exception as e:
        print(f'[net-movement] {e}')
        return None


def _load_sd(path, version):
    """加载权重 state_dict（去 net. 前缀），失败返回 None"""
    try:
        mod = load_mod_for(version)
        sd = torch.load(path, map_location='cpu', weights_only=False)
        if all(k.startswith('net.') for k in sd):
            sd = {k[4:]: v for k, v in sd.items()}
        return sd
    except Exception as e:
        print(f'[net-load] {path}: {e}')
        return None


@app.route('/api/net-trend', methods=['POST'])
@require_auth
def api_net_trend():
    """按用户选择顺序分析多个权重：每个权重相对前一个的"三段移动距离"（逐元素差值 L2）。
    移动距离才能体现"权重更新了多少"；绝对值 L2 各代几乎不变（实测主干 329.19→329.50），
    画出来是平的——2026-08-02 用户反馈后修复"""
    data = request.json or {}
    files = data.get('files', [])   # [{file, version}, ...] 按用户选择顺序
    if not files:
        return jsonify({'error': '未选择权重'}), 400

    results = []
    prev_sd = None
    for item in files:
        f = item.get('file', '')
        version = item.get('version', 'v2')
        info = next((v for v in VERSIONS if v[0] == version), None)
        if info is None:
            continue
        path = os.path.join(PARENT_DIR, info[1], f)
        if not os.path.exists(path):
            results.append({'file': f, 'version': version, 'error': '文件不存在'})
            continue
        sd = _load_sd(path, version)
        if sd is None:
            results.append({'file': f, 'version': version, 'error': '分析失败'})
            continue
        label = f.replace('.pt', '').replace('weights_it', '代').replace(f'_{version}', '')
        stats = _net_section_stats(path, version)
        if prev_sd is None:
            # 第一个权重：无前驱，movement 为空，用自身 L2 作起点
            results.append({'file': f, 'version': version, 'label': label,
                            'movement': None, 'l2': stats or {}})
        else:
            mov = _net_section_movement(prev_sd, sd)
            results.append({'file': f, 'version': version, 'label': label,
                            'movement': mov, 'l2': stats or {}})
        prev_sd = sd
    return jsonify({'results': results})


@app.route('/api/start', methods=['POST'])
@require_auth
def api_start():
    data = request.json or {}
    black_type = data.get('black_type', 'human')
    black_weight = data.get('black_weight', 'weights.pt')
    white_type = data.get('white_type', 'human')
    white_weight = data.get('white_weight', 'weights.pt')
    sim_count = data.get('sim_count', 50)
    greedy = data.get('greedy', True)
    temperature = float(data.get('temperature', 0.3))  # 对弈采样温度 (0.05=近贪心, 1=按分布, >1更随机)
    value_sim = data.get('value_sim', 50)
    board_size = data.get('board_size', 15)  # v16 9×9 对弈页传 9

    reset_game()
    G['board_size'] = board_size

    if board_size == 9:
        import boardgame2
        env = boardgame2.KInARowEnv(board_shape=9, target_length=5)
    else:
        env = gym.make('Gomuku-v0')
    obs, _ = env.reset()
    G['board'] = obs[0]
    G['env'] = env

    # ── 开局手顺预设 (2026-10-01 用户: "支持前两手指点坐标, 可以第一手, 也可以前两手") ──
    #   语义: 依次落子, 从黑开始黑白交替 (第1手=黑, 第2手=白)
    #     → 与 G['moves'] 的奇偶配色规则天然一致 ✓ (build_state 按 index 奇偶判色)
    #   落完后轮到下一色: 1 手 → 轮白走; 2 手 → 轮黑走 ✓
    #   与棋力测评页 /api/eval/start 的 first_moves 同口径 ✓ (最多 2 手)
    preset = []
    for _i, _m in enumerate(data.get('first_moves') or []):
        try:
            _pr, _pc = int(_m['r']), int(_m['c'])
        except Exception:
            return jsonify({'ok': False, 'error': f'开局手顺第{_i+1}手格式无效 (需 {{r,c}})'})
        if not (0 <= _pr < board_size and 0 <= _pc < board_size):
            return jsonify({'ok': False, 'error': f'开局手顺第{_i+1}手越界 (棋盘 {board_size}×{board_size})'})
        if G['board'][_pr, _pc] != 0:
            return jsonify({'ok': False, 'error': f'开局手顺第{_i+1}手已有子'})
        preset.append((_pr, _pc))
    if len(preset) > 2:
        return jsonify({'ok': False, 'error': '开局手顺最多 2 手 (前两手)'})
    for _i, (_pr, _pc) in enumerate(preset):
        # ⚠️ 必须逐手复查占位: 上面的预校验是在【落子前】做的, 挡不住手顺内部自相重复 ✗
        #    (实测: first_moves=[H8,H8] 会被 env 连落两子 → 同点两子, 棋盘被写坏)
        if G['board'][_pr, _pc] != 0:
            return jsonify({'ok': False, 'error': f'开局手顺第{_i+1}手与前面预设的手重复 (已有子)'})
        obs2, _rw, _term, _trunc, _ = env.step(np.array([_pr, _pc]))
        G['board'] = obs2[0]
        G['moves'].append((_pr, _pc))
        if _term or _trunc:
            G['over'] = True
            return jsonify({'ok': False, 'error': f'开局手顺第{_i+1}手直接成五, 无需对弈'})
    if preset:
        G['current'] = BLACK if len(preset) % 2 == 0 else WHITE
        print('[start] 开局手顺 %d 手 %s → 轮到 %s走'
              % (len(preset), preset, '黑' if G['current'] == BLACK else '白'))
    G['config'] = {
        'black_type': black_type, 'black_weight': black_weight,
        'white_type': white_type, 'white_weight': white_weight,
        'sim_count': sim_count, 'greedy': greedy,
        'temperature': max(0.05, min(5.0, temperature)),
        'value_sim': max(1, min(800, value_sim)),
        'board_size': board_size,
        'first_moves': [{'r': r, 'c': c} for r, c in preset],   # ★ 开局手顺 (2026-10-01)
    }

    # 创建 AI agent（如果启用）
    if black_type == 'ai':
        agent, _ = make_agent(black_weight, max(sim_count, 1), board_size)
        G['black_agent'] = agent
    if white_type == 'ai':
        agent, _ = make_agent(white_weight, max(sim_count, 1), board_size)
        G['white_agent'] = agent

    # 价值地图 + 状态快照 (原逻辑: 自动走子前先算一次) —— 这三行不能删 ✗
    #   (删了会让下面 state 未绑定: 非"黑AI开局"的所有分支直接 UnboundLocalError)
    state = build_state()
    G['value_map'] = compute_value_map()
    state = build_state()

    # 开局轮到 AI 就走一步
    #   ★ 2026-10-01: 原来写死 black_type=='ai' ✗ —— 加了手顺后, 只预设 1 手时
    #     轮白走, 若白是 AI 就没人动 → 改成按"当前该谁走"判定 ✓ (无手顺时行为不变 ✓)
    _cur_is_ai = (G['current'] == BLACK and black_type == 'ai') or \
                 (G['current'] == WHITE and white_type == 'ai')
    if _cur_is_ai and not G['over']:
        r, c, info, is_fast = ai_move(sim_count, greedy, temperature=G['config']['temperature'])
        state = build_state()
        state['last_ai'] = {'r': r, 'c': c, 'info': round(info, 4), 'is_fast': is_fast}
        if G.get('last_decision'):
            state['decision'] = G['last_decision']
            G['last_decision'] = None

    return jsonify({'ok': True, 'state': state})


@app.route('/api/play', methods=['POST'])
@require_auth
def api_play():
    """处理一步走棋（人类点击 或 AI auto）"""
    data = request.json or {}
    r = data.get('row')
    c = data.get('col')

    if G['board'] is None:
        return jsonify({'ok': False, 'error': '游戏未开始', 'state': None})
    if G['over']:
        return jsonify({'ok': False, 'error': '游戏已结束', 'state': build_state()})

    config = G['config']
    player = G['current']
    is_human = (player == BLACK and config['black_type'] == 'human') or \
               (player == WHITE and config['white_type'] == 'human')

    # 安全检查：人类回合但没有坐标 → 拒绝（防止前端 {} 请求误闯）
    if is_human and (r is None or c is None):
        return jsonify({'ok': False, 'error': '人类回合需提供坐标', 'state': build_state()})

    state = None
    last_ai = None

    if is_human:
        # 人类走棋
        ok, err = human_move(r, c)
        if not ok:
            return jsonify({'ok': False, 'error': err, 'state': build_state()})
        state = build_state()

        # 人类走完后，如果轮到 AI，自动走
        if state and not state['over']:
            next_is_ai = (G['current'] == BLACK and config['black_type'] == 'ai') or \
                         (G['current'] == WHITE and config['white_type'] == 'ai')
            if next_is_ai:
                rr, cc, info, is_fast = ai_move(config['sim_count'], config['greedy'],
                                                temperature=config['temperature'])
                state = build_state()
                last_ai = {'r': rr, 'c': cc, 'info': round(info, 4), 'is_fast': is_fast}
    else:
        # AI 走棋（AI vs AI 模式下，前端轮询调用）
        rr, cc, info, is_fast = ai_move(config['sim_count'], config['greedy'],
                                        temperature=config['temperature'])
        state = build_state()
        last_ai = {'r': rr, 'c': cc, 'info': round(info, 4), 'is_fast': is_fast}

    resp = {'ok': True, 'state': state}
    if last_ai:
        resp['last_ai'] = last_ai
    if G.get('last_decision'):
        resp['decision'] = G['last_decision']
        G['last_decision'] = None
    return jsonify(resp)


@app.route('/api/state')
@require_auth
def api_state():
    return jsonify({'state': build_state()})


POSITIONS_FILE = os.path.join(SCRIPT_DIR, 'positions.json')

# ═══════════════════════════════════════════════
# 出题评估测试记录存档 (2026-08-27 用户): 保存每次评估的配置+汇总指标
# ═══════════════════════════════════════════════
QUIZ_RESULTS_FILE = os.path.join(SCRIPT_DIR, 'quiz_results.json')

def _load_quiz_results():
    try:
        if os.path.exists(QUIZ_RESULTS_FILE):
            with open(QUIZ_RESULTS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        print(f'[quiz-results-load] {e}')
    return []

def _save_quiz_results(recs):
    with open(QUIZ_RESULTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(recs, f, ensure_ascii=False, indent=1)

@app.route('/api/quiz/save_result', methods=['POST'])
@require_auth
def api_quiz_save_result():
    """保存一次出题评估记录: weight/mode/sims/summary"""
    try:
        data = request.json or {}
        weight = (data.get('weight') or '').strip()
        mode = data.get('mode', 'policy')
        sims = data.get('sims', [])
        summary = data.get('summary') or {}
        if not weight or not summary:
            return jsonify({'ok': False, 'error': '缺少权重或汇总数据'})
        recs = _load_quiz_results()
        recs.append({
            'time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'weight': weight, 'mode': mode, 'sims': sims, 'summary': summary,
        })
        _save_quiz_results(recs)
        return jsonify({'ok': True, 'total': len(recs)})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)})

@app.route('/api/quiz/results', methods=['GET'])
@require_auth
def api_quiz_results():
    """测试记录列表 (最近在前)"""
    try:
        recs = list(reversed(_load_quiz_results()))
        return jsonify({'ok': True, 'results': recs})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)})

@app.route('/api/quiz/result', methods=['POST'])
@require_auth
def api_quiz_result_delete():
    """按 time 删除单条测试记录"""
    try:
        data = request.json or {}
        t = data.get('time', '')
        recs = _load_quiz_results()
        recs = [r for r in recs if r.get('time') != t]
        _save_quiz_results(recs)
        return jsonify({'ok': True, 'total': len(recs)})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)})

def _load_positions():
    try:
        if os.path.exists(POSITIONS_FILE):
            with open(POSITIONS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        print(f'[positions-load] {e}')
    return []


def _save_positions(positions):
    with open(POSITIONS_FILE, 'w', encoding='utf-8') as f:
        json.dump(positions, f, ensure_ascii=False, indent=1)


@app.route('/api/positions/save', methods=['POST'])
@require_auth
def api_positions_save():
    data = request.json or {}
    name = (data.get('name') or '').strip()
    tag = (data.get('tag') or '').strip()
    stones = data.get('stones')
    # 出题功能 (2026-08-23): answer = 推荐选点列表 [{r,c},...] 顺序=优先级, 作为评估数据集
    answer = data.get('answer')
    # 评分功能 (2026-08-26): score = 当前玩家视角胜率 (-1~1), 用户标注 → 将来测 value 准确率
    score = data.get('score')
    # ★ 2026-09-29 用户要: 对弈保存时自动带 权重/sim/执色/赢家 —— 存进棋谱记录, 事后可追溯 ✓
    meta = data.get('meta')
    if isinstance(meta, dict):
        _keep = ('weight_black', 'weight_white', 'role_black', 'role_white', 'sim_count',
                 'human_color', 'winner', 'winner_text', 'moves', 'version', 'saved_at')
        meta = {k: meta.get(k) for k in _keep if k in meta} or None
        if meta and len(json.dumps(meta, ensure_ascii=False)) > 800:
            meta = None       # 防御: 别让超大 meta 撑爆棋谱文件
    else:
        meta = None
    if not name or not isinstance(stones, list) or not stones:
        return jsonify({'ok': False, 'error': '需要名称和棋子'})
    if answer is not None:
        clean = []
        for a in answer:
            try:
                r, c = int(a['r']), int(a['c'])
                if 0 <= r < 15 and 0 <= c < 15:
                    clean.append({'r': r, 'c': c})
            except Exception:
                pass
        answer = clean if clean else None
    if score is not None:
        try:
            score = float(score)
            if not (-1.0 <= score <= 1.0):
                score = None
        except Exception:
            score = None
    positions = _load_positions()
    # 同名覆盖
    positions = [p for p in positions if p.get('name') != name]
    positions.append({'name': name, 'tag': tag, 'stones': stones, 'answer': answer,
                      'score': score, 'meta': meta,     # ★ 对弈保存自动带的 权重/sim/执色/赢家
                      'created': time.strftime('%Y-%m-%d %H:%M')})
    _save_positions(positions)
    return jsonify({'ok': True, 'positions': positions})


@app.route('/api/quiz/run', methods=['POST'])
@require_auth
def api_quiz_run():
    """出题评估: 遍历 positions 里带 answer 的局面, 批量测指定权重的 policy 准确率
    指标: best_rank(推荐点最低rank) / top1_hit / top3_hit / top5_hit + 汇总"""
    try:
        import numpy as np
        data = request.json or {}
        weight = data.get('weight', '')
        mode = data.get('mode', 'policy')          # 'policy' 静态 | 'mcts' 动态模拟
        player_override = data.get('player')       # None=自动(黑先手) | 1=黑视角 | -1=白视角
        sims = data.get('sims') or [40, 100, 200, 400, 800]   # mcts 档位
        sims = sorted(int(s) for s in sims if 1 <= int(s) <= 2000)
        if not weight:
            return jsonify({'ok': False, 'error': '请选择权重'})
        version, wpath = _resolve_weight(weight)
        mod = load_mod_for(version)
        meta = load_metadata()
        net_kwargs = meta.get('config', {}).get('net_kwargs',
            {'conv_filters': [128], 'residual_filters': [[128, 128]] * 8, 'policy_filters': [128]})
        env = gym.make('Gomuku-v0')
        agent = mod.AlphaZeroAgent(env=env, kwargs=net_kwargs, sim_count=1, batches=1, batch_size=8,
                                   prior_exploration_fraction=0.05)  # 用户定调(08-27): 动态题库 noise 0.05 正确口径, 默认 0.25 会拉低 top5 命中
        sd = torch.load(wpath, map_location=mod.device)
        agent.net.load_state_dict(sd)
        agent.net.to(mod.device)
        agent.net.eval()
        positions = _load_positions()
        # 题目按时间从最近到以前 (2026-08-27 用户): positions 保存序旧→新 → 反转
        quiz = [p for p in reversed(positions) if p.get('answer')]
        results = []
        for p in quiz:
            board = np.zeros((15, 15), dtype=np.int8)
            for s in p['stones']:
                r, c = int(s['r']), int(s['c'])
                if 0 <= r < 15 and 0 <= c < 15:
                    board[r, c] = int(s['v'])
            n_black = int(np.sum(board == 1))
            n_white = int(np.sum(board == -1))
            if player_override is not None:
                player = int(player_override)
            else:
                player = 1 if n_black == n_white else -1   # 黑先手
            canonical = (player * board).astype(np.int8)
            valid = (board == 0)
            _lm_q = _lm_of(p.get('stones') or [])      # ★ plane3 = 题目局面最后一手 (2026-09-27)
            ans_points = [(int(a['r']), int(a['c'])) for a in p['answer']
                          if 0 <= int(a['r']) < 15 and 0 <= int(a['c']) < 15 and valid[int(a['r']), int(a['c'])]]
            if mode == 'mcts':
                # ── MCTS 动态模拟: 一次搜到 max(sims), 在每档记录 count 快照 (树增量扩展, 零额外开销) ──
                agent.reset_mcts()
                snapshots = {}
                max_sim = max(sims)
                _paper_noise_init(agent, canonical, player=player, pef=0.05, last_move=_lm_q)
                for i in range(1, max_sim + 1):
                    _agent_search(agent, canonical, player=player, last_move=_lm_q)
                    if i in sims:
                        s = canonical.tobytes()
                        cnt = agent.count.get(s)
                        if cnt is not None:
                            snapshots[i] = cnt * valid
                q_entry = {'name': p['name'], 'created': p.get('created', ''),
                           'n_ans': len(ans_points), 'sims': {}}
                for i in sims:
                    cntv = snapshots.get(i)
                    if cntv is None or cntv.sum() <= 0:
                        continue
                    prob = cntv / cntv.sum()
                    top5_idx = np.argsort(cntv.ravel())[::-1][:5]
                    top5_hit = any((int(j // 15), int(j % 15)) in ans_points for j in top5_idx)
                    ans_prob = round(float(sum(prob[r, c] for r, c in ans_points)) * 100, 2)
                    t5 = [{'r': int(j // 15), 'c': int(j % 15),
                           'visits': int(cntv.ravel()[j]), 'pct': round(float(prob.ravel()[j]) * 100, 2)}
                          for j in top5_idx]
                    q_entry['sims'][i] = {'top5_hit': top5_hit, 'ans_prob': ans_prob, 'top5': t5}
                results.append(q_entry)
            else:
                # ── 静态 policy ──
                pis, vs = _agent_predict(agent, canonical[np.newaxis], players=player,
                                         last_moves=(np.array([_lm_q]) if _lm_q is not None else None))
                pi = np.exp(np.asarray(pis[0])).reshape(15, 15)
                pm = pi * valid
                if pm.sum() <= 0:
                    pm = valid.astype(np.float64)
                pm = pm / pm.sum()
                ans_ranks = []
                for r, c in ans_points:
                    rank = int((pm.ravel() > pm[r, c]).sum()) + 1
                    ans_ranks.append({'r': r, 'c': c, 'rank': rank,
                                      'pct': round(float(pm[r, c]) * 100, 2)})
                ranks = [x['rank'] for x in ans_ranks if x.get('rank')]
                best_rank = min(ranks) if ranks else 999
                top1_idx = int(np.argmax(pm.ravel()))
                top1_rc = (top1_idx // 15, top1_idx % 15)
                top1_hit = any((r, c) == top1_rc for r, c in ans_points)
                top3_hit = any(x.get('rank') and x['rank'] <= 3 for x in ans_ranks)
                top5_hit = any(x.get('rank') and x['rank'] <= 5 for x in ans_ranks)
                order = np.argsort(pm.ravel())[::-1][:5]
                top5 = [{'r': int(j // 15), 'c': int(j % 15), 'pct': round(float(pm.ravel()[j]) * 100, 2)}
                        for j in order]
                results.append({'name': p['name'], 'created': p.get('created', ''),
                                'n_ans': len(ans_points), 'answer': ans_ranks,
                                'best_rank': best_rank, 'top1_hit': top1_hit,
                                'top3_hit': top3_hit, 'top5_hit': top5_hit, 'top5': top5,
                                'value': round(float(np.asarray(vs[0]).reshape(-1)[0]), 3),
                                'user_score': p.get('score')})
        n = len(results)
        summary = {'total': n, 'weight': weight, 'mode': mode}
        if n:
            if mode == 'mcts':
                summary['sims'] = []
                for i in sims:
                    rates = []
                    for r in results:
                        e = r['sims'].get(i)
                        if e: rates.append(e['top5_hit'])
                    if rates:
                        summary['sims'].append({'sim': i,
                            'top5_rate': round(100 * sum(rates) / len(rates), 1),
                            'avg_ans_prob': round(sum(r['sims'][i]['ans_prob'] for r in results
                                                      if i in r['sims']) / len(rates), 2)})
            else:
                summary['top1_rate'] = round(100 * sum(1 for r in results if r['top1_hit']) / n, 1)
                summary['top3_rate'] = round(100 * sum(1 for r in results if r['top3_hit']) / n, 1)
                summary['top5_rate'] = round(100 * sum(1 for r in results if r['top5_hit']) / n, 1)
                summary['avg_best_rank'] = round(sum(r['best_rank'] for r in results) / n, 1)
                # value 评估 (2026-08-26): 有用户评分标注的题 → 网络 value vs 标注误差
                scored = [r for r in results if r.get('user_score') is not None]
                if scored:
                    errs = [abs(r['value'] - r['user_score']) for r in scored]
                    signs = [1 if (r['value'] >= 0) == (r['user_score'] >= 0) else 0 for r in scored]
                    summary['value_mae'] = round(sum(errs) / len(errs), 3)
                    summary['value_sign_acc'] = round(100 * sum(signs) / len(signs), 1)
                    summary['value_n'] = len(scored)
        return jsonify({'ok': True, 'results': results, 'summary': summary})
    except Exception as e:
        print(f'[quiz] {e}')
        import traceback; traceback.print_exc()
        return jsonify({'ok': False, 'error': str(e)})


@app.route('/api/quiz/trend', methods=['POST'])
@require_auth
def api_quiz_trend():
    """题目跨代趋势: 单题 × 多个权重 → 每个推荐选点的概率/rank 变化
    params: name(题目名), weights([version|file...]), mode('policy'|'mcts'=固定800)"""
    try:
        import numpy as np
        data = request.json or {}
        name = data.get('name', '')
        weights = data.get('weights') or []
        mode = data.get('mode', 'policy')
        if not name or not weights:
            return jsonify({'ok': False, 'error': '需要题目名和权重列表'})
        positions = _load_positions()
        p = next((p for p in positions if p.get('name') == name and p.get('answer')), None)
        if not p:
            return jsonify({'ok': False, 'error': '题目不存在: ' + name})
        board = np.zeros((15, 15), dtype=np.int8)
        for s in p['stones']:
            r, c = int(s['r']), int(s['c'])
            if 0 <= r < 15 and 0 <= c < 15:
                board[r, c] = int(s['v'])
        n_black = int(np.sum(board == 1)); n_white = int(np.sum(board == -1))
        player = 1 if n_black == n_white else -1
        canonical = (player * board).astype(np.int8)
        valid = (board == 0)
        ans_points = [(int(a['r']), int(a['c'])) for a in p['answer']
                      if 0 <= int(a['r']) < 15 and 0 <= int(a['c']) < 15 and valid[int(a['r']), int(a['c'])]]
        meta = load_metadata()
        net_kwargs = meta.get('config', {}).get('net_kwargs',
            {'conv_filters': [128], 'residual_filters': [[128, 128]] * 8, 'policy_filters': [128]})
        results = []
        for weight in weights:
            version, wpath = _resolve_weight(weight)
            mod = load_mod_for(version)
            env = gym.make('Gomuku-v0')
            agent = mod.AlphaZeroAgent(env=env, kwargs=net_kwargs, sim_count=1, batches=1, batch_size=8)
            sd = torch.load(wpath, map_location=mod.device)
            agent.net.load_state_dict(sd)
            agent.net.to(mod.device)
            agent.net.eval()
            _lm_t = _lm_of(p.get('stones') or [])      # ★ plane3 = 题目局面最后一手 (两分支共用)
            if mode == 'mcts':
                agent.reset_mcts()
                _paper_noise_init(agent, canonical, player=player, pef=0.05, last_move=_lm_t)
                for _ in range(800):
                    _agent_search(agent, canonical, player=player, last_move=_lm_t)
                s = canonical.tobytes()
                cnt = agent.count.get(s)
                if cnt is None:
                    continue
                cntv = cnt * valid
                if cntv.sum() > 0:
                    pm = cntv / cntv.sum()
                else:
                    continue
            else:
                pis, vs = _agent_predict(agent, canonical[np.newaxis], players=player,
                                         last_moves=(np.array([_lm_t]) if _lm_t is not None else None))
                pi = np.exp(np.asarray(pis[0])).reshape(15, 15)
                pm = pi * valid
                if pm.sum() > 0:
                    pm = pm / pm.sum()
                else:
                    continue
            ans = []
            for r, c in ans_points:
                rank = int((pm.ravel() > pm[r, c]).sum()) + 1
                ans.append({'r': r, 'c': c, 'pct': round(float(pm[r, c]) * 100, 2), 'rank': rank})
            results.append({'weight': weight, 'ans': ans})
        return jsonify({'ok': True, 'name': name, 'mode': mode, 'results': results})
    except Exception as e:
        print(f'[quiz-trend] {e}')
        import traceback; traceback.print_exc()
        return jsonify({'ok': False, 'error': str(e)})


@app.route('/api/positions/list')
@require_auth
def api_positions_list():
    # 从新到旧: 最近保存的排前面 (用户 2026-08-22 要求, 频繁看最近保存的)
    # 2026-09-12: "题目:xxx" 一律置顶 (用户要求 — 出题/做题时高频访问, 不该被新存的局面挤下去)
    #   实现: 稳定排序 → 组内仍保持"从新到旧", 只是把题目组提到最前
    items = _load_positions()[::-1]
    items.sort(key=lambda p: 0 if str(p.get('name', '')).startswith('题目') else 1)
    return jsonify({'positions': items})


@app.route('/api/positions/delete', methods=['POST'])
@require_auth
def api_positions_delete():
    data = request.json or {}
    name = data.get('name')
    positions = _load_positions()
    positions = [p for p in positions if p.get('name') != name]
    _save_positions(positions)
    return jsonify({'ok': True, 'positions': positions})


def _net_has_fc_head(net):
    """v14 论文式 FC 头 (policy_fc/value_fc1 存在) vs v6/v12 极简头 (conv 直出)"""
    return hasattr(net, 'policy_fc') and hasattr(net, 'value_fc1')


def _agg_2d(t):
    """头中间层 (1,C,15,15) → 225 聚合 (多通道 FC 头取通道均值, 保持 1ch 热力图兼容)"""
    arr = t.detach().cpu().numpy()
    if arr.ndim == 4:
        arr = arr[0]  # (C,15,15)
        if arr.shape[0] > 1:
            arr = arr.mean(axis=0)  # FC 头 2ch → 通道均值
    return [round(float(v), 4) for v in arr.ravel()]


@app.route('/api/network_viz', methods=['POST'])
@require_auth
def api_network_viz():
    """网络前向可视化: 当前局面前向, 返回每层激活(15x15聚合)+统计+policy分布+value"""
    try:
        import torch
        import torch.nn.functional as F
        import numpy as np
        # 支持自定义局面 (摆谱): stones=[{r,c,v}] v=1黑/-1白; weight='version|file'
        data = request.json or {}
        stones_param = data.get('stones')
        weight_param = data.get('weight')
        if stones_param is not None:
            board = np.zeros((15, 15), dtype=np.int8)
            for s in stones_param:
                r, c = int(s['r']), int(s['c'])
                if 0 <= r < 15 and 0 <= c < 15:
                    board[r, c] = int(s['v'])
            n_black = int(np.sum(board == 1))
            n_white = int(np.sum(board == -1))
            player = 1 if n_black == n_white else -1
            wf = weight_param
            if not wf:
                wf = G['config'].get('black_weight') or G['config'].get('white_weight')
        else:
            if G['board'] is None:
                return jsonify({'ok': False, 'error': '游戏未开始'})
            board = G['board'].copy()
            player = G['current']
            wf = G['config'].get('black_weight') if player == BLACK else G['config'].get('white_weight')
            if not wf:
                wf = G['config'].get('black_weight') or G['config'].get('white_weight')
        if not wf:
            return jsonify({'ok': False, 'error': '无可用权重'})
        version, wpath = _resolve_weight(wf)
        mod = load_mod_for(version)
        meta = load_metadata()
        net_kwargs = meta.get('config', {}).get('net_kwargs',
            {'conv_filters': [128], 'residual_filters': [[128, 128]] * 8, 'policy_filters': [128]})
        env = gym.make('Gomuku-v0')
        t_agent = mod.AlphaZeroAgent(env=env, kwargs=net_kwargs, sim_count=1, batches=1, batch_size=8)
        sd = torch.load(wpath, map_location=mod.device)
        t_agent.net.load_state_dict(sd)
        t_agent.net.to(mod.device)
        net = t_agent.net
        net.eval()
        canonical = (player * board).astype(np.float32)
        # v8 4通道输入适配
        # 🐛 修复 (2026-08-23): 原 make_input(canonical) 没传 player → v18 ch2 恒 1(黑视角)
        # 白视角局面(ch0=白) + ch2=1(黑) = 输入矛盾 → value 输出错乱(如白视角局面查成+0.091)
        if hasattr(mod, 'make_input'):
            try:
                x = torch.tensor(mod.make_input(canonical, player=player)).unsqueeze(0)
            except TypeError:  # v12/v14 等旧版 make_input 无 player 参数
                x = torch.tensor(mod.make_input(canonical)).unsqueeze(0)
        else:
            x = torch.tensor(canonical).unsqueeze(0).unsqueeze(0)
        # 共享主干逐层
        acts = {}
        h = F.relu(net.bn_input(net.conv_input(x)))
        acts['conv_input'] = h.detach().squeeze(0)
        for i, blk in enumerate(net.res_blocks):
            h = blk(h)
            acts[f'res{i}'] = h.detach().squeeze(0)
        # policy 头 (逐部件拆解: conv → bn → relu → [fc] → softmax; v14 FC头多 policy_fc 层)
        # v20b (2026-08-27): policy = cat(conv1, conv2) → 450 → policy_fc(450→225) → log_softmax
        p_c = net.policy_conv(h)                     # conv(128→1/2, k3)
        p_b = net.policy_bn(p_c)                     # BN
        p_r = F.relu(p_b)                            # ReLU
        if hasattr(net, 'policy_conv2'):             # v20b: 双通道 + 组合层
            p2 = F.relu(net.policy_bn2(net.policy_conv2(h)))
            p_r = torch.cat([p_r, p2], dim=1)        # (1,2,15,15)
            p_fc = net.policy_fc(p_r.reshape(p_r.shape[0], -1))   # (1,450)→(1,225)
            p_log = F.log_softmax(p_fc, dim=1)
        elif _net_has_fc_head(net):
            p_fc = net.policy_fc(p_r.reshape(p_r.shape[0], -1))   # (1,450)→(1,225)
            p_log = F.log_softmax(p_fc, dim=1)
        else:
            p_log = F.log_softmax(p_r.reshape(p_r.shape[0], -1), dim=1)
        p_prob = torch.exp(p_log).detach().cpu().numpy().reshape(15, 15)
        # value 头 (逐部件拆解: conv → bn → relu → fc1/fc → tanh; v14/v19 双层 FC)
        v_c = net.value_conv(h)                      # conv(128→1/2, k3)
        v_b = net.value_bn(v_c)                      # BN
        v_r = F.relu(v_b)                            # ReLU
        # 🐛 2026-08-26: value 头判断独立于 policy (_net_has_fc_head 要求 policy_fc+value_fc1,
        # v19 有 value_fc1/fc2 但 policy 极简头无 policy_fc → 误判走 value_fc 分支崩)
        if hasattr(net, 'value_fc1'):
            v_h1 = F.relu(net.value_fc1(v_r.reshape(v_r.shape[0], -1)))   # (1,450)→(1,128)
            v_fc = float(net.value_fc2(v_h1).reshape(-1)[0])
        else:
            v_fc = float(net.value_fc(v_r.reshape(v_r.shape[0], -1)).reshape(-1)[0])
        v_val = float(torch.tanh(torch.tensor(v_fc)).reshape(-1)[0])
        # 每层聚合
        layers = []
        for name, a in acts.items():
            arr = a.detach().cpu().numpy()
            layers.append({
                'name': name,
                'shape': list(arr.shape),
                'mean': round(float(arr.mean()), 4),
                'std': round(float(arr.std()), 4),
                'nz': round(float((arr > 0).mean()), 4),
                'agg': [round(float(v), 4) for v in arr.mean(axis=0).ravel()],
                'param': _layer_params(net, name),
            })
        vc_arr = v_c.detach().cpu().numpy()
        # 棋盘棋子 (head 层热力图叠加棋子用)
        stones_list = []
        for r in range(15):
            for c in range(15):
                bv = int(board[r, c])
                if bv != 0:
                    stones_list.append({'r': r, 'c': c, 'v': bv})
        # 头部分解: policy/value 传导链 (每部件 1ch 热力图 或 标量; FC头多通道聚合)
        # 🐛 2026-08-26: value 头判断独立 (v19 有 value_fc1/fc2 但 policy 极简头无 policy_fc)
        fc_head = hasattr(net, 'value_fc1')
        v_chain = [
            {'name': 'value_conv', 'shape': list(v_c.shape), 'mean': round(float(v_c.mean()), 4),
             'std': round(float(v_c.std()), 4), 'nz': round(float((v_c > 0).float().mean()), 4),
             'agg': _agg_2d(v_c), 'stones': stones_list},
            {'name': 'value_bn', 'shape': list(v_b.shape), 'mean': round(float(v_b.mean()), 4),
             'std': round(float(v_b.std()), 4), 'nz': round(float((v_b > 0).float().mean()), 4),
             'agg': _agg_2d(v_b), 'stones': stones_list},
            {'name': 'value_relu', 'shape': list(v_r.shape), 'mean': round(float(v_r.mean()), 4),
             'std': round(float(v_r.std()), 4), 'nz': round(float((v_r > 0).float().mean()), 4),
             'agg': _agg_2d(v_r), 'stones': stones_list},
        ]
        if fc_head:
            v_chain.append({'name': 'value_fc1', 'shape': [1, 128],
                            'bars': [round(float(v), 4) for v in v_h1.detach().cpu().numpy()[0].ravel()]})
            v_chain.append({'name': 'value_fc2', 'shape': [1], 'scalar': round(v_fc, 4)})
        else:
            v_chain.append({'name': 'value_fc', 'shape': [1], 'scalar': round(v_fc, 4)})
        v_chain.append({'name': 'value_tanh', 'shape': [1], 'scalar': round(v_val, 4), 'final': True})
        heads = {
            'policy': [
                {'name': 'policy_conv', 'shape': list(p_c.shape), 'mean': round(float(p_c.mean()), 4),
                 'std': round(float(p_c.std()), 4), 'nz': round(float((p_c > 0).float().mean()), 4),
                 'agg': _agg_2d(p_c), 'stones': stones_list},
                {'name': 'policy_bn', 'shape': list(p_b.shape), 'mean': round(float(p_b.mean()), 4),
                 'std': round(float(p_b.std()), 4), 'nz': round(float((p_b > 0).float().mean()), 4),
                 'agg': _agg_2d(p_b), 'stones': stones_list},
                {'name': 'policy_relu', 'shape': list(p_r.shape), 'mean': round(float(p_r.mean()), 4),
                 'std': round(float(p_r.std()), 4), 'nz': round(float((p_r > 0).float().mean()), 4),
                 'agg': _agg_2d(p_r), 'stones': stones_list},
                {'name': 'policy_softmax', 'shape': [15, 15], 'mean': round(float(p_prob.mean()), 6),
                 'std': round(float(p_prob.std()), 6), 'nz': round(float((p_prob > 0).mean()), 4),
                 'agg': [round(float(v), 6) for v in p_prob.ravel()], 'stones': stones_list, 'final': True},
            ],
            'value': v_chain,
        }
        # 头层参数 (权重/偏置热力图数据)
        def _head_param(mod):
            import numpy as np
            if isinstance(mod, torch.nn.Conv2d):
                w = mod.weight.detach().cpu().numpy()
                b = mod.bias.detach().cpu().numpy() if mod.bias is not None else np.zeros(w.shape[0])
                return {'type': 'conv',
                        'weight': {'mean': round(float(w.mean()), 4), 'std': round(float(w.std()), 4),
                                   'min': round(float(w.min()), 4), 'max': round(float(w.max()), 4),
                                   'ch_mean': [round(float(w[ch].mean()), 4) for ch in range(w.shape[0])]},
                        'bias': [round(float(v), 4) for v in b]}
            if isinstance(mod, torch.nn.BatchNorm2d):
                return {'type': 'bn',
                        'gamma': [round(float(v), 4) for v in mod.weight.detach().cpu().numpy()],
                        'beta': [round(float(v), 4) for v in mod.bias.detach().cpu().numpy()]}
            if isinstance(mod, torch.nn.Linear):
                w = mod.weight.detach().cpu().numpy()
                # ⚠️ v14 FC 头权重巨大 (450×225 / 450×128), 不做 15×15 热力图 (w15 只对 225 尺寸)
                w15 = None
                if w.size == 225:
                    w15 = [round(float(v), 5) for v in w.reshape(-1)]
                return {'type': 'fc', 'w15': w15,
                        'mean': round(float(w.mean()), 5), 'std': round(float(w.std()), 5),
                        'bias': round(float(mod.bias.detach().cpu().numpy().reshape(-1)[0]), 4)}
            return None
        _hp = {
            'policy_conv': _head_param(net.policy_conv), 'policy_bn': _head_param(net.policy_bn),
            'value_conv': _head_param(net.value_conv), 'value_bn': _head_param(net.value_bn),
        }
        if fc_head:
            _hp['policy_fc'] = _head_param(net.policy_fc) if hasattr(net, 'policy_fc') else None
            _hp['value_fc1'] = _head_param(net.value_fc1)
            _hp['value_fc2'] = _head_param(net.value_fc2)
        else:
            _hp['value_fc'] = _head_param(net.value_fc)
        for _chain in heads.values():
            for _ly in _chain:
                _ly['param'] = _hp.get(_ly['name'])

        # policy 概率 (mask 合法点)
        valid = (board == 0).astype(np.float32)
        masked = p_prob * valid
        total = masked.sum()
        if total > 0:
            masked = masked / total
        top_idx = np.argsort(masked.ravel())[::-1][:5]
        top5 = [{'r': int(j) // 15, 'c': int(j) % 15,
                 'pct': round(float(masked.ravel()[j]) * 100, 2)} for j in top_idx]
        # value_map (2026-08-26): 每个空点落子后的 value — 存【对方视角】原值(不取反)
        # 语义 (用户 2026-08-26 修正): 当前玩家落子后轮到对方 → value 是对方视角
        # → 当前玩家倾向选"对方 value 最低"的点 (让对方难受); 最低点=最佳, 最高点=对方想要=最差
        value_map = None
        try:
            empty = np.argwhere((board == 0))
            if len(empty):
                cands = []
                for r, c in empty:
                    b2 = board.copy()
                    b2[r, c] = player
                    cands.append(((-player) * b2).astype(np.int8))
                cb = np.stack(cands) if cands else None
                if cb is not None:
                    # ★ plane3 = 每个候选自己刚落的那一手 (2026-09-27; empty 与 cands 同序 ✓)
                    # ⚠️ empty 是 ndarray → 判空必须用 len(), 不能 if empty (数组真值歧义 ✗ 实踩)
                    _, vs = _agent_predict(t_agent, cb, players=-player,
                                           last_moves=(np.asarray(empty) if len(empty) else None))
                    vals = np.asarray(vs).reshape(-1)
                    vm = np.full((15, 15), np.nan)
                    for (r, c), v in zip(empty, vals):
                        vm[r, c] = round(float(v), 3)   # 对方视角原值 (不再取反)
                    value_map = vm
        except Exception as e:
            print(f'[network-viz value_map] {e}')
        # 🐛 2026-08-26: NaN → None (NaN 不是合法 JSON, jsonify 输出字面 NaN 前端 JSON.parse 崩)
        vm_list = None
        vm_stats = None
        if value_map is not None:
            vm_list = [[None if np.isnan(v) else round(float(v), 3) for v in row] for row in value_map]
            # 边角统计 (2026-08-26 用户): A15单点 + 边角均值 + 全盘方差 — 量化"锁边角/飞棋"风险
            try:
                allv = [float(v) for v in value_map.flatten() if not np.isnan(v)]
                rows = np.arange(15)
                corner_mask = (rows[:, None] <= 1) | (rows[None, :] <= 1) | (rows[:, None] >= 13) | (rows[None, :] >= 13)
                cv = [float(value_map[r, c]) for r, c in zip(*np.where(corner_mask)) if not np.isnan(value_map[r, c])]
                vm_stats = {
                    'a15': round(float(value_map[0, 0]), 3) if not np.isnan(value_map[0, 0]) else None,
                    'corner_mean': round(float(np.mean(cv)), 3) if cv else None,
                    'std': round(float(np.std(allv)), 3) if allv else None,
                    'corner_n': len(cv),
                }
            except Exception as e:
                print(f'[network-viz vm_stats] {e}')
            # policy-value 贴合度 (2026-08-26 用户): policy概率 vs 落子后value取反(当前视角) 应一致
            # 符号一致率: (p>中位) 与 (v_cur>中位) 一致比例; MSE: 归一化后均方误差
            try:
                empty_pts = np.argwhere((board == 0))
                if len(empty_pts) > 2:
                    ps = np.array([masked[r, c] for r, c in empty_pts])
                    vcur = np.array([-value_map[r, c] for r, c in empty_pts])
                    vnorm = (vcur + 1) / 2
                    pm = np.median(ps)
                    vm = np.median(vnorm)
                    sign_agree = float(np.mean((ps > pm) == (vnorm > vm)))
                    mse = float(np.mean((ps - vnorm) ** 2))
                    vm_stats['fit'] = {
                        'sign_agree': round(sign_agree, 3),
                        'mse': round(mse, 4),
                        'n': int(len(empty_pts)),
                    }
            except Exception as e:
                print(f'[network-viz fit] {e}')
        ic = 4 if hasattr(mod, 'make_input') else 1
        structure = _net_structure(net, ic)
        return jsonify({'ok': True, 'value': round(v_val, 3), 'player': int(player),
                        'structure': structure,
                        'layers': layers,
                        'heads': heads,
                        'policy': [round(float(v), 6) for v in masked.ravel()],
                        'top5': top5, 'stones': stones_list,
                        'value_map': vm_list, 'vm_stats': vm_stats})
    except Exception as e:
        print(f'[network-viz] {e}')
        return jsonify({'ok': False, 'error': str(e)})


def _layer_params(net, name):
    """提取层参数(权重统计/偏置数组) 用于权重/偏置热力图展示"""
    try:
        import numpy as np
        def conv_info(mod):
            w = mod.weight.detach().cpu().numpy()
            b = mod.bias.detach().cpu().numpy() if mod.bias is not None else np.zeros(w.shape[0])
            return {'type': 'conv',
                    'weight': {'mean': round(float(w.mean()), 4), 'std': round(float(w.std()), 4),
                               'min': round(float(w.min()), 4), 'max': round(float(w.max()), 4),
                               'ch_mean': [round(float(w[ch].mean()), 4) for ch in range(w.shape[0])]},
                    'bias': [round(float(v), 4) for v in b]}
        def bn_info(mod):
            return {'type': 'bn',
                    'gamma': [round(float(v), 4) for v in mod.weight.detach().cpu().numpy()],
                    'beta': [round(float(v), 4) for v in mod.bias.detach().cpu().numpy()]}
        if name.startswith('res'):
            i = int(name[3:])
            blk = net.res_blocks[i]
            return {'type': 'block', 'parts': [
                {'name': 'conv1', **conv_info(blk.conv1)},
                {'name': 'bn1', **bn_info(blk.bn1)},
                {'name': 'conv2', **conv_info(blk.conv2)},
                {'name': 'bn2', **bn_info(blk.bn2)},
            ]}
        if name == 'conv_input':
            return conv_info(net.conv_input)
        return None
    except Exception:
        return None


def _net_structure(net, input_channels):
    """构建网络计算图结构 (节点 DAG + 每层 in/out shape + 参数数)
    支持 v6/v8 (单层 value_fc) 和 v2 (value_fc1/fc2 + policy_fc)"""
    import numpy as np
    nodes, edges = [], []
    N = 15
    def params_of(w, has_bias):
        p = int(np.prod(w.shape))
        if has_bias:
            p += int(np.prod(np.atleast_1d(w.shape)[:1])) if False else 0
        return p
    def add(nid, name, op, inshape, outshape, params):
        nodes.append({'id': nid, 'name': name, 'op': op,
                      'in_shape': inshape, 'out_shape': outshape, 'params': int(params)})
    # 输入
    add('input', '输入棋盘', 'Input', [1, input_channels, N, N], [1, input_channels, N, N], 0)
    # conv_input
    w = net.conv_input.weight
    cin, cout, k = w.shape[1], w.shape[0], w.shape[2]
    add('conv_input', 'conv_input', f'Conv2d({cin}→{cout},k{k})', [1, input_channels, N, N], [1, cout, N, N], w.numel())
    edges.append(['input', 'conv_input'])
    # bn_input
    add('bn_input', 'bn_input', 'BatchNorm2d', [1, cout, N, N], [1, cout, N, N], net.bn_input.weight.numel() * 2)
    edges.append(['conv_input', 'bn_input'])
    # res 块 (每块折叠为 1 节点)
    prev = 'bn_input'
    for i, blk in enumerate(net.res_blocks):
        w1, w2 = blk.conv1.weight, blk.conv2.weight
        p = w1.numel() + w2.numel() + blk.bn1.weight.numel() * 2 + blk.bn2.weight.numel() * 2
        nid = f'res{i}'
        add(nid, f'res{i}', 'Conv→BN→Conv→BN (残差)', [1, w1.shape[1], N, N], [1, w1.shape[0], N, N], p)
        edges.append([prev, nid])
        prev = nid
    # 分叉: policy 头 (v14 FC头: conv→bn→fc→softmax; 极简头: conv→bn→softmax)
    w = net.policy_conv.weight
    nid = 'policy_conv'
    add(nid, 'policy_conv', f'Conv2d({w.shape[1]}→{w.shape[0]},k{w.shape[2]})', [1, w.shape[1], N, N], [1, w.shape[0], N, N], w.numel())
    edges.append([prev, nid])
    add('policy_bn', 'policy_bn', 'BatchNorm2d', [1, w.shape[0], N, N], [1, w.shape[0], N, N], net.policy_bn.weight.numel() * 2)
    edges.append(['policy_conv', 'policy_bn'])
    if hasattr(net, 'policy_fc'):
        p_in = w.shape[0] * N * N
        add('policy_fc', 'policy_fc', 'Linear', [1, p_in], [1, N * N], net.policy_fc.weight.numel() + net.policy_fc.bias.numel())
        edges.append(['policy_bn', 'policy_fc'])
        add('policy_out', 'policy输出', 'LogSoftmax(225)', [1, N * N], [1, N * N], 0)
        edges.append(['policy_fc', 'policy_out'])
    else:
        add('policy_out', 'policy输出', 'LogSoftmax(225)', [1, w.shape[0], N, N], [1, N * N], 0)
        edges.append(['policy_bn', 'policy_out'])
    # 分叉: value 头
    w = net.value_conv.weight
    add('value_conv', 'value_conv', f'Conv2d({w.shape[1]}→{w.shape[0]},k{w.shape[2]})', [1, w.shape[1], N, N], [1, w.shape[0], N, N], w.numel())
    edges.append([prev, 'value_conv'])
    add('value_bn', 'value_bn', 'BatchNorm2d', [1, w.shape[0], N, N], [1, w.shape[0], N, N], net.value_bn.weight.numel() * 2)
    edges.append(['value_conv', 'value_bn'])
    # value fc (v2 双层 / v6/v8 单层)
    v_in = w.shape[0] * N * N
    if hasattr(net, 'value_fc1'):
        add('value_fc1', 'value_fc1', 'Linear', [1, v_in], [1, net.value_fc1.out_features], net.value_fc1.weight.numel() + net.value_fc1.bias.numel())
        edges.append(['value_bn', 'value_fc1'])
        add('value_fc2', 'value_fc2', 'Linear', [1, net.value_fc1.out_features], [1, 1], net.value_fc2.weight.numel() + net.value_fc2.bias.numel())
        edges.append(['value_fc1', 'value_fc2'])
        add('value_out', 'value输出', 'Tanh', [1, 1], [1], 0)
        edges.append(['value_fc2', 'value_out'])
    else:
        add('value_fc', 'value_fc', 'Linear', [1, v_in], [1, 1], net.value_fc.weight.numel() + net.value_fc.bias.numel())
        edges.append(['value_bn', 'value_fc'])
        add('value_out', 'value输出', 'Tanh', [1, 1], [1], 0)
        edges.append(['value_fc', 'value_out'])
    return {'nodes': nodes, 'edges': edges}


@app.route('/api/weight_viz', methods=['POST'])
@require_auth
def api_weight_viz():
    """权重参数可视化: 每层 weight 统计 + 直方图数据 + conv_input 卷积核"""
    try:
        import numpy as np
        data = request.json or {}
        weight = data.get('weight')
        if not weight:
            return jsonify({'ok': False, 'error': '未指定权重'})
        version, wpath = _resolve_weight(weight)
        mod = load_mod_for(version)
        meta = load_metadata()
        net_kwargs = meta.get('config', {}).get('net_kwargs',
            {'conv_filters': [128], 'residual_filters': [[128, 128]] * 8, 'policy_filters': [128]})
        net = mod.AlphaZeroNet(conv_channels=128, residual_blocks=8, board_size=15)
        sd = torch.load(wpath, map_location='cpu', weights_only=False)
        if all(k.startswith('net.') for k in sd):
            sd = {k[4:]: v for k, v in sd.items()}
        net.load_state_dict(sd, strict=False)
        net.eval()
        layers = []
        for name, p in net.named_parameters():
            if 'running' in name or 'num_batches' in name:
                continue
            w = p.detach().cpu().numpy().flatten()
            hist, edges = np.histogram(w, bins=40)
            layers.append({
                'name': name,
                'shape': list(p.shape),
                'mean': round(float(w.mean()), 5),
                'std': round(float(w.std()), 5),
                'min': round(float(w.min()), 5),
                'max': round(float(w.max()), 5),
                'hist': [int(x) for x in hist],
                'edges': [round(float(x), 4) for x in edges],
            })
        # conv_input 卷积核可视化: (C_in, C_out, k, k) → 前几输入通道的核均值图
        kernels = None
        if 'conv_input.weight' in net.state_dict():
            kw = net.state_dict()['conv_input.weight'].detach().cpu().numpy()  # (out,in,k,k)
            # 每个输入通道的平均核 (out 维均值) → (in, k, k)
            kernels = [round(float(x), 4) for x in kw.mean(axis=0).ravel()]
        return jsonify({'ok': True, 'layers': layers, 'kernels': kernels,
                        'weight': weight})
    except Exception as e:
        print(f'[weight-viz] {e}')
        return jsonify({'ok': False, 'error': str(e)})


def _paper_noise_init(agent, canonical, player=None, pef=None, last_move=None):
    """论文式 MCTS 噪声注入 (2026-08-29): 根先验加【一次】Dirichlet 噪声, 后续 search(prior_noise=False) 共用。
    替代原"每 sim 重采样"实现 (探索放大 800 倍 → value 极端化家族病根源)。
    对任意版本权重生效 (不改原脚本): 首次 search 叶子分支缓存根先验 → 混一次噪声写回 policy[s]。
    返回: 是否成功注入 (失败则后续用无噪声搜索)
    ★ 2026-09-27: last_move 透传 (内部这次 search 同样要 plane3, 否则根先验就是错口径 ✗)
    """
    try:
        import numpy as np
        s = canonical.tobytes()
        if s not in agent.policy:
            # 首次: search 叶子分支 (s 未缓存 → predict + 缓存 policy/valid → return, 不探索)
            _agent_search(agent, canonical, player=player, last_move=last_move)
        if s not in agent.policy:
            return False
        alpha = getattr(agent, 'dirichlet_alpha', 0.03)
        rng = np.random.default_rng()
        nv = rng.gamma(alpha, 1., canonical.shape)
        nv *= agent.valid[s]
        if nv.sum() <= 0:
            return False
        nv /= nv.sum()
        frac = pef if pef is not None else getattr(agent, 'prior_exploration_fraction', 0.05)
        agent.policy[s] = (1. - frac) * agent.policy[s] + frac * nv
        return True
    except Exception as e:
        print(f'[paper-noise] {e}')
        return False


def _paper_search(agent, canonical, player=None, sim=800, pef=None, last_move=None):
    """论文式 MCTS 搜索: 根噪声一次 + sim 次共用 (替代每 sim 重采样循环)"""
    _paper_noise_init(agent, canonical, player=player, pef=pef, last_move=last_move)
    for _ in range(sim):
        _agent_search(agent, canonical, player=player, last_move=last_move)


def _traced_search(agent, board, prior_noise, path, player=None, last_move=None):
    """追踪版 MCTS search（复制 v6 search 逻辑 + 记录递归路径）。
    board = canonical (当前玩家视角). path 记录: node(选点)/leaf(predict v)/terminal(终局)
    player: 当前层绝对颜色 (v18 ch2 需要); None 默认黑 (兼容旧版)
    返回 (v, 是否叶子)"""
    import numpy as np
    s = board.tobytes()
    if s not in agent.winner:
        agent.winner[s] = agent.env.get_winner((board, BLACK))
    if agent.winner[s] is not None:
        w = int(agent.winner[s])
        path.append({'type': 'terminal', 'depth': len(path), 'winner': w})
        return float(w), True
    if s not in agent.policy:
        # ── 叶子: 首次遇到, 网络 predict 出 pis 和 vs, 算出 policy 和 v, 结束本次 search ──
        # v18: 传 players=player → ch2 正确视角; ★ plane3 = last_move (2026-09-27)
        pis, vs = _agent_predict(agent, board[np.newaxis], players=player,
                                 last_moves=(np.array([last_move]) if last_move is not None else None))
        pi = np.exp(np.asarray(pis[0]))  # ⚠️ 修复: net输出log概率, 转概率 (2026-08-10)
        if pi.shape != board.shape:
            pi = pi.reshape(board.shape)
        v = float(np.asarray(vs[0]).reshape(-1)[0])
        valid = (board == 0).astype(np.float32)
        masked = pi * valid
        total_masked = np.sum(masked)
        if total_masked <= 0:
            masked = valid
            total_masked = np.sum(masked)
        agent.policy[s] = masked / total_masked
        agent.valid[s] = valid
        path.append({'type': 'leaf', 'depth': len(path), 'v': round(v, 4)})
        return v, True
    # ── 内部节点: UCB 选点, 递归深入 ──
    count_sum = float(agent.count[s].sum())
    coef = (agent.c_init + np.log1p((1 + count_sum) / agent.c_base)) * \
           (count_sum ** 0.5) / (1. + agent.count[s])
    if prior_noise:
        import numpy.random as npr
        alpha = getattr(agent, 'dirichlet_alpha', 0.03)
        noise = npr.gamma(alpha, 1., board.shape)
        noise *= agent.valid[s]
        noise /= noise.sum()
        prior = (1. - agent.prior_exploration_fraction) * agent.policy[s] + \
                agent.prior_exploration_fraction * noise
    else:
        prior = agent.policy[s]
    ub = np.where(agent.valid[s], agent.q[s] + coef * prior, np.nan)
    location_index = int(np.nanargmax(ub))
    location = np.unravel_index(location_index, board.shape)
    path.append({'type': 'node', 'depth': len(path),
                 'action': [int(location[0]), int(location[1])],
                 'ub': round(float(ub[location]), 4)})
    (next_board, next_player), _, _, _ = agent.env.next_step((board, BLACK), np.array(location))
    next_canonical = next_player * next_board
    # ★ 递归下去时 plane3 = 刚选的这一手 (子局面的真实"上一手") (2026-09-27)
    next_v, _ = _traced_search(agent, next_canonical, prior_noise, path,
                               player=-player if player is not None else None,
                               last_move=(int(location[0]), int(location[1])))
    v = next_player * next_v
    # ── 回溯: 更新挂起局面的 MCTS 参数 q 和 count ──
    agent.count[s][location] += 1
    agent.q[s][location] += (v - agent.q[s][location]) / agent.count[s][location]
    path.append({'type': 'backup', 'depth': len(path) - 1, 'node_depth': len(path) - 2,
                 'action': [int(location[0]), int(location[1])],
                 'v': round(float(v), 4),
                 'count': int(agent.count[s][location]),
                 'q': round(float(agent.q[s][location]), 4)})
    return v, False


@app.route('/api/mcts_viz', methods=['POST'])
@require_auth
def api_mcts_viz():
    """MCTS decide 搜索过程可视化: 追踪每次 search 的递归路径(局面→选点→翻转→叶子predict→回溯更新)
    返回: steps(每次模拟的path+count_sum增长), prob(最终选点概率), top候选"""
    try:
        import numpy as np
        data = request.json or {}
        sim = max(1, min(int(data.get('sim', 50)), 5000))
        # 支持自定义局面 (摆谱): stones=[{r,c,v}] v=1黑/-1白; weight='version|file'
        stones_param = data.get('stones')
        weight_param = data.get('weight')
        board = None
        player = None
        wf = None
        if stones_param is not None:
            board = np.zeros((15, 15), dtype=np.int8)
            for s in stones_param:
                r, c = int(s['r']), int(s['c'])
                if 0 <= r < 15 and 0 <= c < 15:
                    board[r, c] = int(s['v'])
            n_black = int(np.sum(board == 1))
            n_white = int(np.sum(board == -1))
            player = 1 if n_black == n_white else -1   # 黑先手, 黑子==白子 → 轮黑
            wf = weight_param
            if not wf:
                wf = G['config'].get('black_weight') or G['config'].get('white_weight')
        else:
            if G['board'] is None:
                return jsonify({'ok': False, 'error': '游戏未开始'})
            board = G['board'].copy()
            player = G['current']
            wf = G['config'].get('black_weight') if player == BLACK else G['config'].get('white_weight')
            if not wf:
                wf = G['config'].get('black_weight') or G['config'].get('white_weight')
        if not wf:
            return jsonify({'ok': False, 'error': '无可用权重'})
        version, wpath = _resolve_weight(wf)
        mod = load_mod_for(version)
        meta = load_metadata()
        net_kwargs = meta.get('config', {}).get('net_kwargs',
            {'conv_filters': [128], 'residual_filters': [[128, 128]] * 8, 'policy_filters': [128]})
        env = gym.make('Gomuku-v0')
        t_agent = mod.AlphaZeroAgent(env=env, kwargs=net_kwargs, sim_count=sim, batches=1, batch_size=8)
        # 自定义 MCTS 参数 (用户参数面板可调, 默认值=训练配方)
        t_agent.c_init = float(data.get('c_init', 1.25))
        t_agent.c_base = float(data.get('c_base', 19652))
        t_agent.prior_exploration_fraction = float(data.get('prior_exploration_fraction', 0.25))
        t_agent.dirichlet_alpha = float(data.get('dirichlet_alpha', 0.03))
        sd = torch.load(wpath, map_location=mod.device)
        t_agent.net.load_state_dict(sd)
        t_agent.net.to(mod.device)
        t_agent.reset_mcts()
        canonical = (player * board).astype(np.int8)
        s = canonical.tobytes()
        # decide 主循环: while count[s].sum() < sim_count: search
        steps = []
        depth_hist = {}   # 层数(1基) → 次数: policy/value 配合好 → 深度大; 配合差 → 浅层乱撞
        max_depth = 0
        depth_sum = 0
        # ★ plane3 = 上一手 (2026-09-27): 摆谱模式取 stones 末项; 对弈模式取 G['moves'] 末项
        _lm_m = _lm_of(stones_param) if stones_param else _lm_of(G.get('moves') or [])
        # 2026-08-29 论文式: 根先验加一次噪声, 后续共用 (替代每 sim 重采样)
        _paper_noise_init(t_agent, canonical, player=player, last_move=_lm_m)
        # 模拟0 (2026-08-29 用户): 根局面初始叶子 predict value (搜索开始前的当前局面 value)
        init_v = None
        try:
            _, vs0 = _agent_predict(t_agent, canonical[np.newaxis], players=player,
                                    last_moves=(np.array([_lm_m]) if _lm_m is not None else None))
            init_v = round(float(np.asarray(vs0[0]).reshape(-1)[0]), 4)
        except Exception:
            pass
        steps.append({'sim': 0, 'count_sum': 0,
                      'path': [{'type': 'leaf', 'depth': 0, 'v': init_v}] if init_v is not None else []})
        for i in range(sim):
            path = []
            _traced_search(t_agent, canonical, False, path, player=player, last_move=_lm_m)
            # 本次模拟最终深度 = 末尾 leaf/terminal 节点的 depth+1 (1基层数)
            final_depth = 0
            for pnode in reversed(path):
                if pnode.get('type') in ('leaf', 'terminal'):
                    final_depth = pnode['depth'] + 1
                    break
            depth_hist[final_depth] = depth_hist.get(final_depth, 0) + 1
            max_depth = max(max_depth, final_depth)
            depth_sum += final_depth
            steps.append({'sim': i + 1, 'count_sum': int(t_agent.count[s].sum()), 'path': path})
        # 全量返回每次 search 的完整路径 (用户用滑块回放任意一次模拟循环)
        # ── 路线 n-gram 词频 (2026-09-07 用户): 全长度子串聚类 — 滑杆切 1~15 步任意长度 ──
        # 任意一条模拟只要经过某片段(开头/中间/结尾均可)就计入该片段
        from collections import defaultdict
        MAXL = 15
        ngram = {L: defaultdict(int) for L in range(1, MAXL + 1)}   # (moves tuple, L) -> count
        ngram_pos = {L: defaultdict(lambda: [0, 0, 0]) for L in range(1, MAXL + 1)}
        max_seen = 0
        for st in steps:
            if st['sim'] == 0:
                continue
            acts = tuple((p['action'][0], p['action'][1])
                         for p in st['path'] if p.get('type') == 'node')
            if not acts:
                continue
            n = len(acts)
            max_seen = max(max_seen, n)
            for L in range(1, min(MAXL, n) + 1):
                for i in range(0, n - L + 1):
                    sub = acts[i:i + L]
                    ngram[L][sub] += 1
                    rel = i / max(1, n - 1)
                    ngram_pos[L][sub][0 if rel < 1 / 3 else (1 if rel < 2 / 3 else 2)] += 1
        # 按长度分组, 每组按 count 降序取 top (含 count=1: 长片段大多只出现1次, 让滑杆切到 7/9 步也能看到"有哪些"
        route_rank = {}
        for L in range(1, MAXL + 1):
            items = []
            for sub, c in ngram[L].items():
                pos = ngram_pos[L][sub]
                items.append({'moves': [[int(r), int(c)] for r, c in sub],
                              'count': c, 'pos': pos})
            items.sort(key=lambda x: -x['count'])
            route_rank[str(L)] = items[:20]
        route_rank['_max_seen'] = max_seen
        # prob = count[s] / count[s].sum()
        cnt = t_agent.count.get(s)
        valid = t_agent.valid.get(s)
        cntv = None
        if cnt is not None and valid is not None:
            cntv = cnt * valid
            prob = cntv / cntv.sum()
            order = np.argsort(cntv.ravel())[::-1][:10]
            top = [{'r': int(j) // 15, 'c': int(j) % 15,
                    'visits': int(cntv.ravel()[j]),
                    'pct': round(float(prob.ravel()[j]) * 100, 1),
                    'q': round(float(t_agent.q.get(s).ravel()[j]), 3)} for j in order]
            # 最终决策 = argmax
            best_idx = int(np.argmax(cntv.ravel()))
            best = {'r': best_idx // 15, 'c': best_idx % 15,
                    'visits': int(cntv.ravel()[best_idx])}
        else:
            top, best = [], {}
        stones = []
        for r in range(15):
            for c in range(15):
                bv = int(board[r, c])
                if bv != 0:
                    stones.append({'r': r, 'c': c, 'v': bv})
        # ★ 2026-09-27 用户要: 网络 policy 先验输出 + 与 MCTS 实际分布的偏离 (同规格热力图)
        #   ⚠️ 不能读 t_agent.policy[s] —— 它已被 _paper_noise_init 按 prior_exploration_fraction
        #      (0.25) 混入 Dirichlet 噪声 ✗ → 那是"带噪先验", 不是网络原始输出
        #      → 这里单独跑一次干净前向(同一 agent/权重/plane3)拿原始 policy ✓
        pol_flat, pol_dev, prob_norm = None, None, None
        try:
            _lms = np.array([_lm_m]) if _lm_m is not None else None
            _out = _agent_predict(t_agent, canonical[np.newaxis], players=player, last_moves=_lms)
            # ⚠️ 不能盲目 reshape(15,15): 某些版本 predict 返回的元组顺序/形状不同,
            #    取到 value(size 1) 就会 "cannot reshape array of size 1 into (15,15)" ✗
            #    (2026-09-29 实踩: 该异常被外层 except 吞掉 → 响应里没 policy → 前端 drawProb(undefined)
            #     抛错打断整个渲染 → "点模拟没反应" ✗) → 改为【在返回值里找 225 维的那一个】✓
            _pol_arr = None
            for _item in (_out if isinstance(_out, (tuple, list)) else [_out]):
                _a = np.asarray(_item, dtype=np.float64)
                if _a.size == 225:
                    _pol_arr = _a.reshape(15, 15)
                    break
            if _pol_arr is None:
                raise ValueError('predict 返回值里找不到 225 维 policy (形状: %s)'
                                 % [np.asarray(x).shape for x in (_out if isinstance(_out, (tuple, list)) else [_out])])
            _p = np.exp(_pol_arr)          # net 输出 log 概率 → 概率
            _s_root = canonical.tobytes()
            _vf = np.asarray(t_agent.valid.get(_s_root), dtype=bool).reshape(15, 15)
            _p = np.where(_vf, _p, 0.0).ravel()
            _sm = float(_p.sum())
            if _sm > 0:
                _p = _p / _sm
                pol_flat = _p
            if pol_flat is not None and cntv is not None:
                # ⚠️ 注意: 返回给页面的 'prob' 是【原始访问次数】(和=sim, 页面按自身最大值着色)
                #   → 算偏离必须先把次数归一化成概率, 否则是拿"次数"和"概率"比 ✗ (实测 MSE 352 全是噪声)
                _q_raw = np.asarray(cntv.ravel(), dtype=np.float64)
                _qsum = float(_q_raw.sum())
                _q = _q_raw / _qsum if _qsum > 0 else _q_raw
                prob_norm = [round(float(x), 6) for x in _q]
                _d = _q - pol_flat
                pol_dev = {
                    'mse': round(float(np.mean(_d ** 2)), 6),        # 均方差(用户点名要看)
                    'rmse': round(float(np.sqrt(np.mean(_d ** 2))), 6),
                    'mae': round(float(np.mean(np.abs(_d))), 6),
                    # ★ TV(总变差) = 0.5·Σ|Δ| = 搜索"挪走了多少概率质量" —— 比 MSE 直观得多 ✓
                    #   (MSE 被 225 格摊薄, 会把"挪走 41%"这种大重排显示成 1.5e-4 的"小数字" ✗)
                    'tv': round(float(0.5 * np.sum(np.abs(_d))), 4),
                    'max_abs': round(float(np.max(np.abs(_d))), 6),
                    'cos': round(float(np.dot(_q, pol_flat) /
                                       (np.linalg.norm(_q) * np.linalg.norm(pol_flat) + 1e-12)), 4),
                    'kl_pol_to_mcts': round(float(np.sum(np.where(pol_flat > 0,
                                        pol_flat * np.log((pol_flat + 1e-12) / (_q + 1e-12)), 0.0))), 4),
                    'n_valid': int(np.sum(_vf)),
                    'argmax_same': bool(int(np.argmax(_q)) == int(np.argmax(pol_flat))),
                }
        except Exception as _e:
            print(f'[mcts-viz policy] {_e}')
        return jsonify({'ok': True, 'sim_count': sim, 'player': int(player),
                        'steps': steps, 'truncated': False,
                        'depth_hist': [{'depth': k, 'count': v} for k, v in sorted(depth_hist.items(), reverse=True)],
                        'max_depth': max_depth,
                        'avg_depth': round(depth_sum / max(1, sim), 2),
                        'prob': [round(float(x), 6) for x in cntv.ravel()] if cntv is not None else [],
                        'policy': [round(float(x), 6) for x in pol_flat] if pol_flat is not None else [],
                        'prob_norm': prob_norm,      # 归一化后的 MCTS 概率(与 policy 同口径, 供偏离计算)
                        'pol_dev': pol_dev,
                        'route_rank': route_rank,
                        'top': top, 'best': best, 'stones': stones})
    except Exception as e:
        print(f'[mcts-viz] {e}')
        return jsonify({'ok': False, 'error': str(e)})


# ═══════════════════════════════════════════════
# 摆谱研究
# ═══════════════════════════════════════════════

@app.route('/api/study/analyze', methods=['POST'])
@require_auth
def api_study_analyze():
    """分析摆谱局面：价值地图 + 网络推荐 + MCTS推荐"""
    import boardgame2
    data = request.json or {}
    stones = data.get('stones', [])
    weight_file = data.get('weight', 'weights.pt')
    sim_count = max(1, min(800, data.get('sim_count', 50)))

    # 重建棋盘
    board = np.zeros((15, 15), dtype=np.int32)
    for s in stones:
        board[s['r'], s['c']] = s['p']

    # 当前执棋方：棋子数决定（第一步黑，轮流）
    player = BLACK if len(stones) % 2 == 0 else WHITE

    # 加载 agent（支持 v1/v2/v3 多版本权重）
    try:
        version, wpath = _resolve_weight(weight_file)
        mod = load_mod_for(version)
        meta = load_metadata()
        net_kwargs = meta.get('config', {}).get('net_kwargs',
            {'conv_filters': [128], 'residual_filters': [[128, 128]] * 8, 'policy_filters': [128]})
        env = gym.make('Gomuku-v0')
        agent = mod.AlphaZeroAgent(env=env, kwargs=net_kwargs,
                                   sim_count=sim_count, batches=1, batch_size=8)
        if os.path.exists(wpath):
            state_dict = torch.load(wpath, map_location=mod.device)
            agent.net.load_state_dict(state_dict)
            agent.net.to(mod.device)
        else:
            return jsonify({'ok': False, 'error': f'权重文件不存在: {weight_file}'})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)})

    # 运行 MCTS（⚠️ 必须 prior_noise=True：v2 policy 均匀 + prior_noise=False 时 UCB 全平，
    # nanargmax 锁定展平序首点 (0,0)，关键点(堵活三/扩活三)永远不被访问→选点离谱。
    # Dirichlet 噪声打破平局让搜索能探索关键点——2026-08-01 双活三局面实锤）
    canonical = player * board
    agent.reset_mcts()
    # 2026-08-29 论文式: 根噪声一次 + 共用 (替代每 sim 重采样; v2 policy 均匀期靠这一次噪声破平局)
    _lm_s = _lm_of(stones)          # ★ plane3 = 摆谱局面的最后一手 (2026-09-27)
    _paper_search(agent, canonical, player=player, sim=sim_count, last_move=_lm_s)

    # 原始价值网络预测（黑棋胜率）
    pis, vs = _agent_predict(agent, canonical[np.newaxis], players=player,
                             last_moves=(np.array([_lm_s]) if _lm_s is not None else None))
    raw_value = float(np.asarray(vs[0]).reshape(-1)[0])  # 从当前玩家视角
    # 转为黑棋胜率（0~1）
    if player == BLACK:
        black_win_rate = (raw_value + 1) / 2
    else:
        black_win_rate = (1 - raw_value) / 2
    black_win_rate = round(max(0, min(1, black_win_rate)) * 100, 1)

    s = boardgame2.strfboard(canonical) if version == 'v1' else canonical.tobytes()
    q_map = agent.q.get(s)
    count = agent.count.get(s)
    policy = agent.policy.get(s)

    if q_map is None or count is None:
        return jsonify({'ok': False, 'error': 'MCTS 搜索失败'})

    # 价值地图（仅显示访问数 >= MIN_VISITS 的位置——1 次访问的 Q 是单次评估噪声，无统计意义）
    # 阈值按 sim 自适应：sim 越大搜索越充分，可要求更多访问才显示
    MIN_VISITS = max(2, sim_count // 200)
    vmap = [[None]*15 for _ in range(15)]
    for idx in range(225):
        if count.flat[idx] >= MIN_VISITS:
            r, c = divmod(idx, 15)
            vmap[r][c] = round(float(q_map.flat[idx]), 3)

    # 有效位置掩码
    valid = agent.valid.get(s, np.ones((15, 15), dtype=float))

    # 纯网络推荐（policy top-5）
    masked_policy = policy * valid
    policy_flat = masked_policy.flatten()
    top5_policy_idx = np.argsort(policy_flat)[::-1][:5]
    policy_top = []
    for idx in top5_policy_idx:
        if policy_flat[idx] > 0:
            r, c = divmod(idx, 15)
            policy_top.append({'r': int(r), 'c': int(c), 'prob': round(float(policy_flat[idx]), 4)})

    # MCTS 推荐（visit count top-5）
    count_valid = count * valid
    count_flat = count_valid.flatten()
    top5_mcts_idx = np.argsort(count_flat)[::-1][:5]
    mcts_top = []
    for idx in top5_mcts_idx:
        if count_flat[idx] > 0:
            r, c = divmod(idx, 15)
            visits = int(count.flat[idx])
            mcts_top.append({
                'r': int(r), 'c': int(c),
                'visits': visits,
                'q': round(float(q_map.flat[idx]), 3)
            })

    return jsonify({
        'ok': True,
        'value_map': vmap,
        'policy_top': policy_top,
        'mcts_top': mcts_top,
        'current': int(player),
        'black_win_rate': black_win_rate,
        'valid': [[int(valid[r,c]) for c in range(15)] for r in range(15)],
    })


@app.route('/study')
@require_auth
def study_page():
    return send_from_directory(os.path.join(SCRIPT_DIR, 'static'), 'study.html')


@app.route('/game9')
@require_auth
def game9_page():
    """9×9 对弈页（加载 v16 GN 试验版权重）"""
    return send_from_directory(os.path.join(SCRIPT_DIR, 'static'), 'game9.html')

"""eval API 代码块 — 追加到 app.py (Rapfi 棋力测评)
用法: 在 app.py 合适位置插入本文件内容
"""
# ═══════════════════════════════════════════════
# 棋力测评 (Rapfi 外部锚点 PK) — 2026-08-22
# ═══════════════════════════════════════════════
import re as _re
import threading as _threading   # 模块顶层需要 (EVAL_LOCK 初始化在 import threading 之前)

RAPFI_BIN = os.path.expanduser('~/rapfi/run/pbrain-rapfi')
RAPFI_CONFIG = os.path.expanduser('~/rapfi/run/config.toml')
EVAL_LOCK = _threading.Lock()   # 防并发 step (前端轮询+sim大时请求会堆积)
EVAL_STATE = {'active': False, 'board': None, 'current': 1, 'stones': [],
              'agent': None, 'engine': None, 'we_black': True, 'history': [],
              'winner': None, 'moves': 0, 'config': {}, 'wr_history': []}

def _set_rapfi_depth(depth, message_mode='none'):
    """设置 Rapfi 难度 + 输出模式（段感知改写，命中数断言）

    🚨 2026-09-26 用户报"测评页老是无响应"的根因（实测 dump 原始 stdout 确认）:
      本地 config 的 message_mode 被留成 "brief" → mac 持久模式下引擎**不按步回坐标**:
        第1手 20s 内无任何坐标输出; 第2手起坐标**滞后一手**(拿到的是上一手的答案, 连续两手同一坐标)
      → web eval 每步干等 _read_line 30s 超时 / 拿到过期着法 → 页面卡死 ✗
      message_mode="none": 坐标 1.77s 立即返回 ✓
    → 每次开测强制写回 none, 避免被历史污染的 config 再坑
    """
    with open(RAPFI_CONFIG) as f:
        lines = f.read().split('\n')
    cur, out = None, []
    hits = {'max_search_depth': 0, 'message_mode': 0}
    for line in lines:
        st = line.strip()
        if st.startswith('[') and st.endswith(']'):
            cur = st[1:-1]
        elif cur == 'search' and _re.match(r'\s*max_search_depth\s*=', line):
            out.append('max_search_depth = {}'.format(depth))
            hits['max_search_depth'] += 1
            continue
        elif cur == 'general' and _re.match(r'\s*message_mode\s*=', line):
            out.append('message_mode = "{}"'.format(message_mode))
            hits['message_mode'] += 1
            continue
        out.append(line)
    if hits['max_search_depth'] != 1 or hits['message_mode'] != 1:
        raise RuntimeError('rapfi config 改写异常: %s (各应命中 1 次)' % hits)
    with open(RAPFI_CONFIG, 'w') as f:
        f.write('\n'.join(out))

class _RapfiEngine:
    def __init__(self):
        self.p = None
        self.start_new()
    def start_new(self):
        if self.p:
            try: self.p.kill()
            except Exception: pass
        self.p = subprocess.Popen([RAPFI_BIN], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  text=True, bufsize=1, cwd=os.path.dirname(RAPFI_BIN))
        # 启动横幅: 读到静默为止(最多 3s) —— 旧版固定读 2 行, 引擎行数一变就协议错位 ✗
        import select as _sel
        _end = time.time() + 3.0
        while time.time() < _end:
            _r, _, _ = _sel.select([self.p.stdout], [], [], 0.25)
            if not _r:
                break
            if not self.p.stdout.readline():
                break
    def _read_line(self, timeout=30):
        import select as _sel
        end = time.time() + timeout
        while time.time() < end:
            r, _, _ = _sel.select([self.p.stdout], [], [], 0.5)
            if r:
                line = self.p.stdout.readline()
                if not line: return None
                line = line.strip()
                if line and not line.startswith('MESSAGE'):
                    return line
            if self.p.poll() is not None: return None
        return None
    def start(self, size=15):
        self.p.stdin.write('START {}\n'.format(size)); self.p.stdin.flush()
        return self._read_line()
    def move(self, stones):
        lines = ['BOARD']
        for r, c, col in stones:
            lines.append('{},{},{}'.format(c, r, 1 if col == 1 else 2))
        lines.append('DONE')
        _body = '\n'.join(lines) + '\n'
        try:
            self.p.stdin.write(_body); self.p.stdin.flush()
        except (BrokenPipeError, OSError) as _e:
            # ★ 2026-09-30 实踩: Rapfi 引擎进程已死 → 写 stdin 抛 BrokenPipeError →
            #   原来直接冒到 Flask 变 500 (日志一堆 traceback ✗)。这里重启引擎重试一次 ✓
            print('[rapfi] 引擎管道断了 (%s) → 重启引擎重试' % _e)
            try:
                self.start_new()
                self.start(15)          # 新进程需要重新 START
                self.p.stdin.write(_body); self.p.stdin.flush()
            except Exception as _e2:
                print('[rapfi] 重启引擎后仍失败: %s' % _e2)
                return None             # 走 engine_error 提示, 不判胜负 ✓
        resp = self._read_line()
        if resp is None: return None
        parts = resp.split(',')
        if len(parts) >= 2 and parts[0] != '0':
            try:
                mv = (int(parts[1]), int(parts[0]))
            except ValueError:
                return None
            # ★ 空位校验 (2026-09-26): message_mode 被污染时引擎会回"上一手的坐标"(滞后响应),
            #   落在已有子上 → 判引擎异常, 绝不把废着法塞进棋局(否则整局静默走歪)
            if any(mv[0] == r and mv[1] == c for r, c, _ in stones):
                print('[rapfi] 收到已占位坐标 %s (疑似滞后/错位响应) → 判引擎异常' % (mv,))
                return None
            return mv
        return None
    def close(self):
        try:
            if self.p: self.p.kill()
        except Exception: pass


@app.route('/api/eval/start', methods=['POST'])
@require_auth
def api_eval_start():
    """启动棋力测评: 我方权重 vs Rapfi
    params: weight(version|file), sim, depth, we_black(bool), temperature(采样温度, 0.05=贪心)
    """
    try:
        data = request.json or {}
        weight = data.get('weight', '')
        sim = max(1, min(int(data.get('sim', 200)), 2000))
        depth = max(0, min(int(data.get('depth', 10)), 30))
        we_black = bool(data.get('we_black', True))
        temperature = max(0.05, min(5.0, float(data.get('temperature', 0.05))))
        if not weight:
            return jsonify({'ok': False, 'error': '请选择权重'})
        # 清理旧状态
        if EVAL_STATE.get('engine'):
            try: EVAL_STATE['engine'].close()
            except Exception: pass
        # 设置 rapfi 难度
        _set_rapfi_depth(depth)
        # 创建我方 agent
        agent, _ = make_agent(weight, sim)
        # 创建 rapfi 引擎
        engine = _RapfiEngine()
        engine.start(15)
        # 初始化棋盘
        import gym
        env = gym.make('Gomuku-v0')
        obs, _ = env.reset()
        # 🔴 开局手顺预设 (2026-09-27 用户: "黑棋首手 / 白棋首手 都要能预置,
        #    不管那一步本该 Rapfi 走还是我方权重走")
        #   接口: first_moves=[{r,c}, {r,c}, ...] 依次落子(黑先, 黑白交替)
        #   兼容旧参数: first_black={r,c} (= first_moves 只有一手)
        #   应用后 current 自动轮到下一色 → 之后谁走仍按 we_black 判定 ✓
        first_black = data.get('first_black')          # 兼容字段
        first_moves = data.get('first_moves')          # 新: 手顺列表
        stones = []
        moves = 0
        history = []
        preset = []
        if isinstance(first_moves, list) and first_moves:
            for i, it in enumerate(first_moves):
                try:
                    preset.append((int(it['r']), int(it['c'])))
                except Exception:
                    try: engine.close()
                    except Exception: pass
                    return jsonify({'ok': False, 'error': f'first_moves[{i}] 无效(需 {{r,c}})'})
        elif first_black is not None:
            try:
                preset.append((int(first_black['r']), int(first_black['c'])))
            except Exception:
                try: engine.close()
                except Exception: pass
                return jsonify({'ok': False, 'error': 'first_black 参数无效(需 {r,c})'})
        _pr = 1                                        # 预设从黑开始, 逐手交替
        for _i, (pr, pc) in enumerate(preset):
            if not (0 <= pr < 15 and 0 <= pc < 15):
                try: engine.close()
                except Exception: pass
                return jsonify({'ok': False, 'error': f'预设第{_i+1}手 ({pr},{pc}) 越界'})
            if obs[0][pr, pc] != 0:
                try: engine.close()
                except Exception: pass
                return jsonify({'ok': False, 'error': f'预设第{_i+1}手 ({pr},{pc}) 已有子'})
            obs2, reward, terminated, truncated, _ = env.step(np.array([pr, pc]))
            obs = obs2
            stones.append((pr, pc, _pr))
            moves += 1
            history.append({'move': moves, 'who': ('黑' if _pr == 1 else '白') + '(预设)',
                            'r': pr, 'c': pc, 'color': _pr})
            if terminated or truncated:
                try: engine.close()
                except Exception: pass
                return jsonify({'ok': False, 'error': f'预设第{_i+1}手直接成五, 无需对弈'})
            _pr = -_pr
        EVAL_STATE.update({
            'active': True, 'board': obs[0].copy(), 'current': _pr,
            'stones': stones, 'agent': agent, 'engine': engine, 'we_black': we_black,
            'history': history, 'winner': None, 'moves': moves, 'engine_error': False,
            'wr_history': [],          # ★ 我方 value 胜率曲线点 (2026-10-01)
            'config': {'weight': weight, 'sim': sim, 'depth': depth, 'we_black': we_black,
                       'temperature': temperature, 'first_black': bool(preset),
                       'preset_moves': [{'r': r, 'c': c} for r, c in preset]},
            '_env': env,
        })
        _eval_record_wr()          # ★ 起点(开局/预设后)也记一个点 → 曲线从第0手起 (2026-10-01)
        return jsonify({'ok': True, 'state': _eval_state()})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)})


@app.route('/api/eval/step', methods=['POST'])
@require_auth
def api_eval_step():
    """推进一步: 轮到的方走棋 (我方 MCTS 或 Rapfi)
    返回新局面; 结束后返回 winner"""
    if not EVAL_LOCK.acquire(blocking=False):
        return jsonify({'ok': False, 'error': '上一步处理中', 'state': None})
    try:
        if not EVAL_STATE.get('active'):
            return jsonify({'ok': False, 'error': '未开始', 'state': None})
        st = EVAL_STATE
        if st['winner'] is not None:
            return jsonify({'ok': True, 'state': _eval_state(), 'done': True})
        board = st['board'].copy()
        current = st['current']
        stones = list(st['stones'])
        agent = st['agent']; engine = st['engine']
        we_black = st['we_black']
        who = '我方' if (current == 1) == we_black else 'Rapfi'
        # ★ 输入 plane3 = 最后一手 (2026-09-27 修): stones 有序 → 最后一项就是上一手 ✓
        #   不传的话 plane3 恒 0 → 与训练时输入分布不一致 (实测值函数差可达 0.26) ✗
        _last_mv = _lm_of(stones)
        if who == '我方':
            agent.reset_mcts()
            # ✅ 干净搜索 (prior_noise=False) + argmax = 最强着法; 带噪声会稀释堵点先验 → 活三/活四不堵
            # (2026-08-22 用户指出; 2026-08-23 用户拍板 eval 要最强棋力 → 代码与注释统一为 False)
            # v18: search 需传 player (ch2 视角), 否则递归 -player = -None 崩 (2026-08-23 修复)
            for _ in range(st['config']['sim']):
                _agent_search(agent, current * board, player=current, last_move=_last_mv)
            s = (current * board).tobytes()
            cnt = agent.count[s]
            temp = st['config'].get('temperature', 0.05)
            if temp <= 0.05:
                action = tuple(np.unravel_index(int(np.argmax(cnt.ravel())), (15, 15)))
            else:
                # 温度采样: count 分布 ^(1/T) 归一化后采样 (与对弈页同语义)
                pv = cnt.astype(np.float64).ravel()
                pv = pv ** (1.0 / temp)
                pv = pv / pv.sum()
                action = tuple(np.unravel_index(int(np.random.choice(pv.size, p=pv)), (15, 15)))
        else:
            mv = engine.move(stones)
            if mv is None:
                # Rapfi 无回应 — 先检查棋盘是否有真实五连 (2026-08-27: 原 winner=-current 误判——
                # 7手黑4子无五连却判"我方胜"; Rapfi 引擎卡住≠对方真赢)
                try:
                    w = st['_env'].get_winner((board, current))
                except Exception:
                    w = None
                if w is not None:
                    st['winner'] = int(w)   # 真有五连 → 真实胜
                else:
                    st['winner'] = None     # 无五连 = 引擎异常, 不判胜负
                st['engine_error'] = True   # 前端提示"引擎异常"而非胜负
                return jsonify({'ok': True, 'state': _eval_state(), 'done': True})
            action = tuple(mv)
        obs2, reward, terminated, truncated, _ = st['_env'].step(np.array(action))
        st['board'] = obs2[0].copy()
        st['stones'].append((int(action[0]), int(action[1]), current))
        st['moves'] += 1
        st['history'].append({'move': st['moves'], 'who': who, 'r': int(action[0]), 'c': int(action[1]), 'color': current})
        _eval_record_wr()          # ★ 每步后追加我方 value 胜率点 (含终局那一步)
        winner = st['_env'].get_winner(obs2)
        if terminated or truncated:
            st['winner'] = int(winner)
            st['active'] = False
            return jsonify({'ok': True, 'state': _eval_state(), 'done': True})
        st['current'] = -current
        return jsonify({'ok': True, 'state': _eval_state(), 'done': False})
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'ok': False, 'error': str(e)})
    finally:
        EVAL_LOCK.release()


@app.route('/api/eval/state')
@require_auth
def api_eval_state():
    """返回当前测评状态 (保存棋谱用)"""
    return jsonify({'ok': True, 'state': _eval_state()})


@app.route('/api/eval/reset', methods=['POST'])
@require_auth
def api_eval_reset():
    try:
        if EVAL_STATE.get('engine'):
            try: EVAL_STATE['engine'].close()
            except Exception: pass
        EVAL_STATE.update({'active': False, 'board': None, 'current': 1, 'stones': [],
                           'agent': None, 'engine': None, 'winner': None, 'moves': 0,
                           'wr_history': []})
        return jsonify({'ok': True})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)})


def _eval_record_wr():
    """测评每步后追加一个「我方 value 胜率」点 (棋力测评页画曲线用)。

    视角 = 我方执色 (we_black → 黑, 否则白) → 曲线恒为"我方胜率":
    我方 value 头越自信越高 ✓ (终点≈100 = 赢 / ≈0 = 输 → 曲线本身也是 value 准确度的体检)
    move = 当前手数; 0 = 开局(预设手之后, 还没走子); 异常一律吞掉(辅助功能不能拖垮对弈)
    """
    try:
        st = EVAL_STATE
        stones = st.get('stones') or []
        # ⚠️ 空盘不记点 (2026-10-01 实测抓到假点):
        #   空盘在训练里只以"黑先落子前"形态出现 → "空盘 + 我方执白"是分布外输入,
        #   value 头给 90.3% 假值 (执黑空盘 96% 才是真值); 空盘 value 本身也是已知退化点 (09-06 定调)
        #   → 曲线从第1手起 (有预设手顺时 = 预设之后那一手), 避免开局假尖峰
        if not stones:
            return
        color = 1 if st.get('we_black') else -1
        wr = _agent_side_wr(st.get('agent'), st.get('board'), color,
                            last_move=_lm_of(stones))
        hist = st.get('history') or []
        who = hist[-1]['who'] if hist else '开局'
        st.setdefault('wr_history', []).append({
            'move': int(st.get('moves', 0)), 'color': color,
            'our_wr': wr, 'who': who,
            'ours': bool(hist) and who == '我方',
        })
    except Exception as e:
        print(f'[eval-wr] {e}')


def _eval_state():
    st = EVAL_STATE
    return {
        'active': st['active'],
        'current': st['current'],
        'stones': [{'r': r, 'c': c, 'v': col} for r, c, col in st['stones']],
        'history': st['history'],
        'winner': st['winner'],
        'engine_error': st.get('engine_error', False),   # Rapfi 引擎无回应 (非真实胜负)
        'moves': st['moves'],
        'we_black': st['we_black'],
        'wr_history': st.get('wr_history', []),          # ★ 我方 value 胜率曲线 (2026-10-01)
        'our_color': 1 if st.get('we_black') else -1,
        'config': st['config'],
    }


@app.route('/eval')
@require_auth
def eval_page():
    return send_from_directory(os.path.join(SCRIPT_DIR, 'static'), 'eval.html')


@app.route('/api/slot')
@require_auth
def api_slot():
    """当前活跃槽 —— 页面用它决定【默认版本】(单一真相源 active_slot.json)

    2026-09-27: 以前 6 个页面各自硬编码 _pref 默认版本链, 每轮切换要逐页改 ✗
    → 页面改为优先用这里的 slot; 换槽 = `python3 gomoku_web/set_slot.py vNN` (一条命令)
    ⚠️ 本进程的内存常量在启动时定死 → 若文件里的槽与本进程不一致, 返回 restart_needed=True
    """
    try:
        from slot_config import active_cfg
        cfg = active_cfg()
    except Exception:
        cfg = dict(ACTIVE_SLOT)
    return jsonify({
        'ok': True,
        'slot': cfg.get('slot') or TRAIN_VERSION,
        'compare': cfg.get('compare') or COMPARE_VERSION,
        'round': cfg.get('round'),
        'goal': cfg.get('goal'),
        'coach_note': cfg.get('coach_note'),
        'student_note': cfg.get('student_note'),
        'student_name': cfg.get('student_name'),
        'coach_name': cfg.get('coach_name'),
        'updated': cfg.get('updated'),
        'source': cfg.get('source'),
        'running_process_slot': TRAIN_VERSION,       # 本进程实际在用的槽
        'restart_needed': bool(cfg.get('slot') and cfg.get('slot') != TRAIN_VERSION),
    })


@app.route('/api/slot/switch', methods=['POST'])
@require_auth
def api_slot_switch():
    """一键切换【web 监控的活跃槽】(v38 <-> v39) —— 2026-09-29 用户要

    ⚠️ 只切"web 看哪个槽" ✗ —— **服务器换轮 + 血统核验不在这里**:
       服务器侧走 `scripts/round_switch/switch_round.sh <R名>` (两条铁律自动校验 ✓)
       web 侧血统备注用 `python3 gomoku_web/set_slot.py <槽> --round N --coach ... --student ...`
    实现: 写 active_slot.json → touch app.py → Werkzeug reloader(debug=True, 已确认开启)
          自动干净重启子进程 → 新的槽常量生效 ✓ (不 pkill, 不影响别的进程 ✓)
    """
    try:
        data = request.json or {}
        slot = (data.get('slot') or '').strip()
        if slot not in ('v38', 'v39'):
            return jsonify({'ok': False, 'error': 'slot 必须是 v38 / v39'})
        old = TRAIN_VERSION
        if slot == old:
            return jsonify({'ok': True, 'slot': slot, 'old': old, 'restart': 'none',
                            'message': '已经是该槽, 无需切换'})
        from slot_config import write_slot
        write_slot(slot, round_no=data.get('round'),
                   coach_note=data.get('note') or '（web 一键切换：血统备注待补 —— 建议用 set_slot.py 重填）',
                   student_note='')
        # ★ 2026-09-30 改: 不再用 "touch app.py 让 reloader 原地重启" ✗
        #   实踩: reloader 原地重启会抢端口失败 → 新旧进程一起退 → web 整个挂掉 ✗ (用户报"web崩了")
        #   → 改为起一个 detached 小脚本: 等 1.2s(让响应先发出去) → pkill 旧进程 → 重新拉起 ✓
        _log = os.path.join(os.path.dirname(SCRIPT_DIR), 'web_v36.log')
        _helper = (
            'import subprocess, time, sys\n'
            'time.sleep(1.2)\n'
            "subprocess.run(\"pkill -f '[a]pp\\\\.py'\", shell=True)\n"
            'time.sleep(1.5)\n'
            'log = open(%r, "ab")\n'
            'subprocess.Popen([sys.executable, "app.py"], cwd=%r, stdout=log,\n'
            '                 stderr=subprocess.STDOUT, start_new_session=True)\n'
        ) % (_log, SCRIPT_DIR)
        try:
            subprocess.Popen([sys.executable, '-c', _helper], start_new_session=True)
        except Exception as _e:
            print('[slot-switch] respawn 失败:', _e)
            return jsonify({'ok': True, 'slot': slot, 'old': old, 'restart': 'manual',
                            'message': '配置已写, 但自动重启失败 → 请手动重启 web'})
        return jsonify({'ok': True, 'slot': slot, 'old': old, 'restart': 'respawn',
                        'message': '已写配置, web 将在约 3 秒内重启 (旧进程先退出再拉起), 页面稍后自动刷新'})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)})


@app.route('/api/slot.js')
@require_auth
def api_slot_js():
    """把活跃槽以【同步脚本站点】形式暴露: <script src="/api/slot.js"></script>

    为什么不用 fetch: replay.html 的版本表(VER/MAIN_VER)和"记住上次选择"逻辑都在
    页面解析期就要用 VER ✗ 异步 fetch 会先渲染上一轮的槽 → 看着像数据没更新 ✗
    脚本标签是同步执行的, 放在版本块之前即可保证拿到值 ✓
    """
    try:
        from slot_config import active_cfg
        cfg = active_cfg()
    except Exception:
        cfg = dict(ACTIVE_SLOT)
    import json as _json
    payload = {
        'slot': cfg.get('slot') or TRAIN_VERSION,
        'compare': cfg.get('compare') or COMPARE_VERSION,
        'round': cfg.get('round'),
        'goal': cfg.get('goal'),
        'student_name': cfg.get('student_name'),
        'coach_name': cfg.get('coach_name'),
        'coach_note': cfg.get('coach_note'),
        'running_process_slot': TRAIN_VERSION,
    }
    js = 'window.ACTIVE_SLOT = ' + _json.dumps(payload, ensure_ascii=False) + ';'
    return app.response_class(js, mimetype='application/javascript')


# ═══════════════════════════════════════════════
# 本地棋谱胜率重算 (2026-09-09 用户: 对保存的棋谱用所选权重逐步 MCTS 算 qa)
# ═══════════════════════════════════════════════
ANALYSIS_STATE = {'running': False, 'pos_name': None, 'weight': None, 'sim': 0,
                  'total': 0, 'done': 0, 'error': None, 'cancel': False}
ANALYSIS_LOCK = _threading.Lock()
ANALYSIS_FILE = os.path.join(SCRIPT_DIR, 'analysis_records.json')
# 线程内会新起 torch agent → 主进程 fork 安全由 reloader 单进程保证 (debug=False 时无 reloader)


def _load_analysis_records():
    if not os.path.exists(ANALYSIS_FILE):
        return {}
    try:
        return json.load(open(ANALYSIS_FILE, encoding='utf-8'))
    except Exception:
        return {}


def _save_analysis_records(recs):
    tmp = ANALYSIS_FILE + '.tmp'
    json.dump(recs, open(tmp, 'w', encoding='utf-8'), ensure_ascii=False)
    os.replace(tmp, ANALYSIS_FILE)


def _run_analysis(pos_name, stones, weight, sim):
    """后台线程: 逐局面(落子前)用所选权重 MCTS sim 次。
    同时输出两种 qa (2026-09-10 用户: 要最高棋力口径):
      actual_qa: 棋谱实际落子点 Q (评估棋谱那手好坏)
      best_qa:   argmax(访问数) 点 Q = 最高棋力(贪心)会下那点 — 与棋力测评(eval 温度0.05=argmax)同口径,
                 反映"局面本身轮到方在最强续走下几成胜" (80% 预知线应看这个)
    stones: [{r,c,v}...] 完整落子序列(黑先交替)。冷点(落子点零访问): 网络对落子后局面 value 取反补。"""
    import gym
    n = len(stones)
    try:
        st = ANALYSIS_STATE
        agent, env = make_agent(weight, sim)
        # 初始化棋盘
        obs, _ = env.reset()
        board = obs[0].astype(np.int8)
        qas = []
        winner = None
        for k in range(n):
            if st['cancel']:
                st['error'] = '已取消'
                break
            s = stones[k]
            v = int(s['v'])
            # 若轮到方与序列不符(如非完整对局)则停止
            # 棋盘交替: 当前落子前局面轮到谁 = 盘面黑子数>白子数 ? 白 : 黑 (v36 规则: 黑先)
            nb = int((board == 1).sum())
            nw = int((board == -1).sum())
            expect = 1 if nb == nw else -1
            if v != expect:
                st['error'] = f'第{k + 1}手颜色不符(期望{expect}, 棋谱{v}) — 棋谱需从黑1开始完整交替'
                break
            # 搜索前局面 (落子前, 轮到 v) — 与 eval 同配置: 无噪声干净搜索
            agent.reset_mcts()
            canon = (v * board).astype(np.int8)
            # ★ plane3 = 这一步的【上一手】= stones[k-1] (k=0 时 None) (2026-09-27)
            _lm_r = _lm_of(stones[:k])
            for _ in range(sim):
                _agent_search(agent, canon, player=v, last_move=_lm_r)
            sbytes = canon.tobytes()
            cnt = agent.count.get(sbytes)
            qtab = agent.q.get(sbytes)
            actual_qa = None
            best_qa = None
            br = bc = None
            cold = False
            if cnt is not None and qtab is not None:
                # best = argmax(访问数) = eval 贪心最强应对点 (与有子处 count=0 不冲突)
                cntf = np.asarray(cnt, dtype=np.float32)
                best_idx = int(np.argmax(cntf.ravel()))
                br, bc = best_idx // 15, best_idx % 15
                best_qa = round(float(qtab.ravel()[best_idx]), 4)
                # actual = 棋谱实际落子点 (可能 ≠ argmax)
                if cntf[int(s['r']), int(s['c'])] > 0:
                    actual_qa = round(float(qtab[int(s['r']), int(s['c'])]), 4)
            if actual_qa is None:
                # 冷点: 权重搜索从未考虑该落子点 → 直接网络评估"下这手后局面" (对手视角取反 → 当前视角)
                cold = True
                b2 = board.copy()
                b2[int(s['r']), int(s['c'])] = v
                canon2 = ((-v) * b2).astype(np.int8)
                # ★ plane3 = 刚落的这一手 (冷点评估的子局面) (2026-09-27)
                _, vs = _agent_predict(agent, canon2[np.newaxis], players=-v,
                                       last_moves=np.array([[int(s['r']), int(s['c'])]]))
                actual_qa = round(float(-np.asarray(vs).reshape(-1)[0]), 4)
            qas.append({'r': int(s['r']), 'c': int(s['c']), 'v': v,
                        'actual_qa': actual_qa, 'best_qa': best_qa,
                        'best_r': br, 'best_c': bc, 'cold': cold})
            # 落子并检查终局
            obs2, _, terminated, truncated, _ = env.step(np.array([int(s['r']), int(s['c'])]))
            board = obs2[0].astype(np.int8)
            if terminated or truncated:
                try:
                    winner = int(env.get_winner(obs2))
                except TypeError:
                    try:
                        winner = int(env.get_winner((board, 1)))
                    except Exception:
                        winner = None
                except Exception:
                    winner = None
                # 终局手也入 qas 后结束
                if k == n - 1:
                    break
                # 若棋谱后面还有子但已终局 → 停 (棋谱尾部可能多录)
                st['error'] = None
                break
            st['done'] = k + 1
        # 存记录 (按 棋谱名 → [{weight,sim,qa,winner,created}])
        recs = _load_analysis_records()
        entry = {
            'weight': weight, 'sim': sim, 'qas': qas, 'winner': winner,
            'created': time.strftime('%Y-%m-%d %H:%M'),
            'n': len(qas),
        }
        hist = recs.get(pos_name, [])
        # 同 权重+sim 覆盖旧记录
        hist = [h for h in hist if not (h['weight'] == weight and h['sim'] == sim)]
        hist.append(entry)
        recs[pos_name] = hist
        _save_analysis_records(recs)
        st['error'] = None
    except Exception as e:
        import traceback; traceback.print_exc()
        ANALYSIS_STATE['error'] = str(e)
    finally:
        ANALYSIS_STATE['running'] = False
        ANALYSIS_STATE['total'] = n if 'n' in dir() else ANALYSIS_STATE.get('total', 0)
        ANALYSIS_STATE['done'] = ANALYSIS_STATE.get('total', 0)


@app.route('/api/analysis/start', methods=['POST'])
@require_auth
def api_analysis_start():
    """对保存棋谱(positions)做胜率重算: 逐步局面 MCTS 取该步落子点 Q。
    body: {pos_name, weight(version|file), sim}"""
    data = request.json or {}
    pos_name = (data.get('pos_name') or '').strip()
    weight = data.get('weight', '')
    sim = max(1, min(int(data.get('sim', 800)), 2000))
    if not pos_name or not weight:
        return jsonify({'ok': False, 'error': '需要棋谱名和权重'})
    if ANALYSIS_STATE['running']:
        return jsonify({'ok': False, 'error': f"已有分析在跑 ({ANALYSIS_STATE['pos_name']})"})
    pos = next((p for p in _load_positions() if p.get('name') == pos_name), None)
    if not pos or not pos.get('stones'):
        return jsonify({'ok': False, 'error': '棋谱不存在或无棋子: ' + pos_name})
    stones = [{'r': int(s['r']), 'c': int(s['c']), 'v': int(s['v'])} for s in pos['stones']]
    if not stones or stones[0]['v'] != 1:
        return jsonify({'ok': False, 'error': '棋谱需从黑1(v=1)开始完整落子序列'})
    n = len(stones)
    ANALYSIS_STATE.update({'running': True, 'pos_name': pos_name, 'weight': weight, 'sim': sim,
                           'total': n, 'done': 0, 'error': None, 'cancel': False})
    t = _threading.Thread(target=_run_analysis, args=(pos_name, stones, weight, sim), daemon=True)
    t.start()
    return jsonify({'ok': True, 'total': n})


@app.route('/api/analysis/status')
@require_auth
def api_analysis_status():
    st = ANALYSIS_STATE
    return jsonify({'ok': True, 'running': st['running'], 'pos_name': st['pos_name'],
                    'weight': st['weight'], 'sim': st['sim'],
                    'done': st['done'], 'total': st['total'],
                    'pct': round(100 * st['done'] / max(1, st['total']), 1) if st['total'] else 0,
                    'error': st['error'], 'cancel': st['cancel']})


@app.route('/api/analysis/cancel', methods=['POST'])
@require_auth
def api_analysis_cancel():
    ANALYSIS_STATE['cancel'] = True
    return jsonify({'ok': True})


@app.route('/api/analysis/probe', methods=['POST'])
@require_auth
def api_analysis_probe():
    """即时单局面重算: 对给定局面(已落 stones)用所选权重 MCTS sim 次, 返回最高棋力推荐落子+Q。
    用户回放到某步想"最新权重会怎么下/评价多少"时用。body: {stones:[{r,c,v}...当前局面], weight, sim}"""
    data = request.json or {}
    weight = data.get('weight', '')
    sim = max(1, min(int(data.get('sim', 800)), 2000))
    stones = data.get('stones') or []
    if not weight:
        return jsonify({'ok': False, 'error': '需要权重'})
    if not stones or not isinstance(stones, list):
        return jsonify({'ok': False, 'error': '需要当前局面 stones'})
    try:
        import gym
        agent, env = make_agent(weight, sim)
        obs, _ = env.reset()
        board = obs[0].astype(np.int8)
        for s in stones:
            r, c, v = int(s['r']), int(s['c']), int(s['v'])
            if not (0 <= r < 15 and 0 <= c < 15):
                return jsonify({'ok': False, 'error': f'坐标越界 ({r},{c})'})
            obs2, _, terminated, truncated, _ = env.step(np.array([r, c]))
            if terminated or truncated:
                return jsonify({'ok': False, 'error': '当前局面已终局, 无需重算'})
            board = obs2[0].astype(np.int8)
        nb = int((board == 1).sum()); nw = int((board == -1).sum())
        player = 1 if nb == nw else -1
        agent.reset_mcts()
        canon = (player * board).astype(np.int8)
        # ★ plane3 = 当前局面最后一手 (2026-09-27)
        _lm_p = _lm_of(stones)
        for _ in range(sim):
            _agent_search(agent, canon, player=player, last_move=_lm_p)
        sbytes = canon.tobytes()
        cnt = agent.count.get(sbytes); qtab = agent.q.get(sbytes)
        if cnt is None or qtab is None:
            return jsonify({'ok': False, 'error': '搜索未建立'})
        cntf = np.asarray(cnt, dtype=np.float32)
        valid = (board == 0).astype(np.float32)
        mask = cntf * valid
        # top 候选 (访问数掩码排序)
        order = np.argsort(mask.ravel())[::-1][:5]
        top = []
        for j in order:
            if mask.ravel()[j] <= 0:
                continue
            top.append({'r': int(j) // 15, 'c': int(j) % 15,
                        'visits': int(cntf.ravel()[j]),
                        'q': round(float(qtab.ravel()[j]), 4)})
        best = top[0] if top else None
        return jsonify({'ok': True, 'player': int(player), 'sim': sim,
                        'weight': weight, 'best': best, 'top': top,
                        'stones': [{'r': int(s['r']), 'c': int(s['c']), 'v': int(s['v'])} for s in stones]})
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'ok': False, 'error': str(e)})


@app.route('/api/analysis/list')
@require_auth
def api_analysis_list():
    """某棋谱已有分析记录: ?pos_name=xxx → [{weight,sim,qas,winner,created,n}]"""
    pos_name = request.args.get('pos_name', '')
    recs = _load_analysis_records()
    hist = recs.get(pos_name, [])
    # 最近在前
    return jsonify({'ok': True, 'records': hist[::-1], 'pos_name': pos_name})


@app.route('/api/analysis/records')
@require_auth
def api_analysis_records():
    """全部分析记录: 有记录的棋谱名列表(前端历史下拉)"""
    recs = _load_analysis_records()
    out = []
    for name, hist in recs.items():
        out.append({'name': name, 'count': len(hist),
                    'last': hist[-1].get('created', '') if hist else ''})
    out.sort(key=lambda x: -len(x['name']))
    return jsonify({'ok': True, 'names': out})



# ═══════════════════════════════════════════════
# 训练监控驾驶舱（SSH 拉取服务器训练数据）
# ═══════════════════════════════════════════════

import subprocess, re, threading
from datetime import datetime

# ── 监控目标服务器（当前主线 = v18 ch2视角修复版 老机器；历史: v12/v16 新机/v14） ──
# v18 (2026-08-22 起, ch2 视角颜色通道修复 + MCTS player 传递修复, 从零重训): 老机器 15×15
# 根因: v8 起 ch2 恒 1.0 → 网络永远执黑视角 → 白视角防守评估颠倒
# 🚨 安全 (2026-09-10): 连接凭证**不得硬编码入库** — 从本地未跟踪文件 creds.local.json 读取
#   (该文件在 .gitignore 中; 缺失时回退环境变量/占位符, 页面提示未配置)
_CRED_PATH = os.path.join(SCRIPT_DIR, 'creds.local.json')
try:
    with open(_CRED_PATH, encoding='utf-8') as _f:
        _CRED = json.load(_f)
except Exception:
    _CRED = {}
SSH_HOST = os.environ.get('GOMOKU_SSH_HOST') or _CRED.get('host', '未配置(creds.local.json)')
SSH_PORT = int(os.environ.get('GOMOKU_SSH_PORT') or _CRED.get('port', 22))
SSH_USER = os.environ.get('GOMOKU_SSH_USER') or _CRED.get('user', 'root')
SSH_PASS = os.environ.get('GOMOKU_SSH_PASS') or _CRED.get('pass', '')
# ══════════════════════════════════════════════════════════════════
# 🚨 监控目标 = 当前活跃槽 —— 单一真相源: gomoku_web/active_slot.json
#   (2026-09-27 用户: "来回切弄简单点")
#   ★ 双槽交替设计 (见仓库根 轮次切换清单_v38_v39双槽.md):
#     v39 槽 = 学生黑(先手)/教练白;  v38 槽 = 学生白(后手)/教练黑;  两槽轮换复用, 不新增版本
#   ★ 切换 = `python3 gomoku_web/set_slot.py v39` (写 JSON + 重启 web + 打印核验)
#     以前每轮要改 4 处(app.py / replay_api.py / 6个页面 _pref / 扫描脚本默认) ✗ 漏一处就错位
#   优先级: 环境变量 GOMOKU_TRAIN_VERSION > active_slot.json > slot_config.DEFAULT_SLOT
#   ⚠️ 槽目录里的过程数据每轮归档到 <槽>_round<K>/ → 本地也要跟着归档, 否则两轮数据混一个索引 ✗
# ══════════════════════════════════════════════════════════════════
try:
    from slot_config import active_cfg as _active_cfg, SLOT_META as _SLOT_META
except ImportError:      # 非包内导入时的兜底
    import importlib.util as _ilu
    _sp = _ilu.spec_from_file_location('slot_config', os.path.join(SCRIPT_DIR, 'slot_config.py'))
    _sc = _ilu.module_from_spec(_sp)
    _sp.loader.exec_module(_sc)
    _active_cfg, _SLOT_META = _sc.active_cfg, _sc.SLOT_META
ACTIVE_SLOT = _active_cfg()
TRAIN_VERSION = ACTIVE_SLOT['slot']
# ★ 2026-11-15: v24b 起支持「远端根目录可配」——
#   老布局(v36~v42) 项目根固定 /root/pk_v8，权重目录名 = gomoku_checkpoints_<槽号>；
#   v24b 部署在 /root/workspace/run_v24b，而它的 VERSION 常量仍是 "v24"
#   → 目录名不能从槽号派生，必须由 slot_config 显式给 ckpt_basename，否则一条都拉不到。
REMOTE_ROOT = os.environ.get('GOMOKU_REMOTE_ROOT') or ACTIVE_SLOT.get('remote_root') or '/root/pk_v8'
CKPT_BASENAME = ACTIVE_SLOT.get('ckpt_basename') or f'gomoku_checkpoints_{TRAIN_VERSION}'
# loss 曲线 / 胜率的对比线版本 = 另一个槽
COMPARE_VERSION = ACTIVE_SLOT['compare']
SSH_CHECKPOINT_DIR = f'{REMOTE_ROOT}/{CKPT_BASENAME}'
# 🚨 主日志必须在项目目录 (screen 主进程的 stdout 重定向); 早期习惯的 /tmp/train_vNN.log 已停更
#    —— 指错路径会让驾驶舱卡在旧代 + ETA 全空 (2026-09-13 实踩)
#    ⚠️ 同一日志会【跨轮追加】(如 train_v39.log 里还留着上一轮的行) → KPI 解析必须按轮次过滤 ✗
TRAINING_LOG = (os.environ.get('GOMOKU_TRAIN_LOG')
                or (f"{REMOTE_ROOT}/{ACTIVE_SLOT['main_log']}" if ACTIVE_SLOT.get('main_log')
                    else f'{REMOTE_ROOT}/train_{TRAIN_VERSION}.log'))
# 工兵: 20 (20×6=120 超额产 → 主进程只等 100 局)。这里是 SSH 拉取上限，
# 实际显示数量取 metadata.config.worker_count（留余量）
WORKER_NUM = int(ACTIVE_SLOT.get('workers') or 22)   # 各槽工兵数 (v40=20)
_wlfmt = ACTIVE_SLOT.get('worker_log_fmt')
WORKER_LOGS = ([f'{REMOTE_ROOT}/' + _wlfmt.format(i=i) for i in range(WORKER_NUM)] if _wlfmt
               else [f'{REMOTE_ROOT}/worker_{TRAIN_VERSION}_{i}.log' for i in range(WORKER_NUM)])
# 攒批推理服务（独占 GPU 的独立进程，各工兵 MCTS leaf 请求经 Unix socket 汇聚）
# 🚨 日志名不总能从槽号派生 (v38 槽复用 v37 的推理脚本, 日志名带 _v37) → 由 slot_config 给
# v24b: 无攒批推理服务（每工兵本地推理）→ 空串 → 该面板显示 OFF
INFER_LOG = os.environ.get('GOMOKU_INFER_LOG') or ACTIVE_SLOT.get('infer_log') or ''
INFER_SOCK = '/tmp/gomoku_infer.sock'
# 门控赛产物: gate_history.json 在检查点目录, 分片明细在 gate_log/ (项目根)
#   v38/v39 门控都是 OFF (目的改为"学生追固定教练") → 本段照旧采集, 前端按 enabled 显示 OFF
GATE_LOG_DIR = os.environ.get('GOMOKU_GATE_LOG_DIR') or f'{REMOTE_ROOT}/gate_log'
# ══ 教练学习: 冻结教练 (weights_coach.pt) + 只训学生色 ══
COACH_WEIGHTS = f'{SSH_CHECKPOINT_DIR}/weights_coach.pt'      # 冻结教练快照
SPIRAL_FLAG = f'{SSH_CHECKPOINT_DIR}/aborted_spiral.flag'     # 熔断器回滚标记 (存在=已触发过)
LATE_MERGE_LOG = f'{SSH_CHECKPOINT_DIR}/late_merge.log'       # 晚到局结转日志 (LATE_MERGE=1 才有)
REPLAY_POOL_DIR = f'{SSH_CHECKPOINT_DIR}/replay_pool'         # 本轮回放池 (判"本轮从哪代起")
# 熔断器参数 (与 start_vNN.sh 默认一致; 日志里也会印, 以日志为准)
SPIRAL_RISE, SPIRAL_THR, SPIRAL_HEALTHY = 3, 0.25, 0.15
# 各槽的【达标线】与【教练来源】(页面直接用; 换槽自动带出)
VERSION_GOALS = {'v38': 0.55, 'v39': 0.80, 'v40': 0.30, 'v41': 0.30, 'v42': 0.60, 'v24b': 0.30}
VERSION_COACH = {'v38': 'A2 = v39 it39 (第二轮教练: 黑)', 'v39': "A2' = v38 it77 (第三轮教练: 白)",
                 'v40': '无教练 (纯自对弈单通道线)', 'v41': '无教练 (v40+梯度检查点, 单通道)',
                 'v42': '教练法+自动换边 (学生达标即换边, 颜色动态)',
                 'v24b': '无教练 (纯自对弈; 黑方空盘首手走 |value| 最小的均衡手)'}
if ACTIVE_SLOT.get('coach_note'):
    VERSION_COACH[TRAIN_VERSION] = ACTIVE_SLOT['coach_note']
COACH_GOAL_RATE = ACTIVE_SLOT['goal']
COACH_FROM = VERSION_COACH.get(TRAIN_VERSION, '—')

TRAIN_CACHE = {}
TRAIN_LOCK = threading.Lock()


def ssh_cmd(cmd, host=None, port=None, user=None, pwd=None):
    """执行远程命令，返回纯输出（expect 单会话，命令内避免 Tcl 特殊字符）
    默认用当前监控服务器；传 host/port/pwd 可访问其他机器（如 v17 对照机）"""
    host = host or SSH_HOST
    port = port or SSH_PORT
    user = user or SSH_USER
    pwd = pwd or SSH_PASS
    safe_cmd = cmd.replace('"', '\\"')
    script = (
        'set timeout 25\n'
        'spawn ssh -o StrictHostKeyChecking=no -p ' + str(port) + ' '
        + user + '@' + host + ' "' + safe_cmd + '"\n'
        'expect "password:"\n'
        'send "' + pwd + '\\r"\n'
        'expect eof\n'
    )
    try:
        r = subprocess.run(['expect', '-c', script], capture_output=True, text=True, timeout=30)
        out = r.stdout
        idx = out.find('password:')
        if idx >= 0:
            out = out[idx+9:]
        lines = []
        for line in out.split('\n'):
            line = line.strip('\r').strip()
            if line and not line.startswith('spawn') and line not in ['expect', 'eof']:
                lines.append(line)
        return '\n'.join(lines)
    except Exception as e:
        return ''


def _iter_json_objects(text):
    """从任意文本里切出【完整】的 JSON 对象串 (大括号配对 + 字符串/转义感知)。

    用途: health_log.json 是 pretty-printed 数组, 只 tail 了尾部若干字节 → 首尾必然被截断 ✗
    (json.loads 整体会炸) → 逐个切出完整条目才能用 json.loads 解析。
    """
    out, depth, start, in_str, esc = [], 0, None, False, False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == '\\':
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == '{':
            if depth == 0:
                start = i
            depth += 1
        elif ch == '}':
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    out.append(text[start:i + 1])
                    start = None
    return out


def fetch_training_data():
    """采集训练数据（单 SSH 会话多段拉取，避免 AutoDL SSH 频率限制）"""
    data = {}
    data['time'] = datetime.now().strftime('%H:%M:%S')

    # 一次连接拉取全部数据，用 ===TAG=== 分隔（echo 自带换行；PTY 下 \r 可能把上段末尾与 tag 粘成一行，切段时归还前后缀）
    # LOG 拉全量 (training_v12.log, 用于解析全历史 loss 三条线; tail -200 只够本代)
    parts = [
        "echo '===LOG==='; cat '" + TRAINING_LOG + "' 2>/dev/null | tail -4000",
        "echo '===META==='; cat '" + SSH_CHECKPOINT_DIR + "/metadata.json'",
        "echo '===CK==='; ls '" + SSH_CHECKPOINT_DIR + "'/weights_it*.pt 2>/dev/null",
        "echo '===EXP==='; ls '" + SSH_CHECKPOINT_DIR + "'/worker_data/iter*.pkl 2>/dev/null",
        # 攒批推理服务：近 4 条统计行 + socket 是否存在 (1=在, 0=无)
        "echo '===INF==='; tail -4 '" + INFER_LOG + "' 2>/dev/null; echo '---SOCK'; "
        "ls -l '" + INFER_SOCK + "' 2>/dev/null | wc -l",
        # 最近 12 个权重的 mtime(epoch) — 跨进程重启仍连续，用来算每代耗时。
        # 🚨 必须按 mtime 数值排序后取尾（`ls | tail` 是字典序，会取到 it92..it99 这类老代）
        "echo '===CKTS==='; ls -l --time-style=+%s '" + SSH_CHECKPOINT_DIR + "'/weights_it*.pt "
        "2>/dev/null | sort -k6 -n | tail -12",
        # 体检 (health_log.json): v38 招牌指标 = 空盘 value (熔断器输入) + 白1距黑1 中位 + 激活率
        #   🚨 2026-09-25 改法: 原来是 grep 三个 ASCII 键 —— 空盘键是中文键, 文件里是 \uXXXX 转义,
        #      grep 不进 (转义序列要写 \\u7a7a\\u76d8 极难维护 ✗)。改为【拉原始 JSON 尾部, 本地解析】,
        #      一次拿到全部字段 (空盘/活四/判别度/w1/激活率) 且不怕键名转义。
        #      tail -c 40000 ≈ 最近 5~6 代 (实测每代约 7KB / 160 行)
        "echo '===HEALTH==='; tail -c 40000 '" + SSH_CHECKPOINT_DIR + "/health_log.json' 2>/dev/null",
        # 教练 + 熔断器: 教练快照 mtime/size、熔断标记文件内容、每代「教练行剔除」样本量
        "echo '===COACH==='; stat -c '%Y %s' '" + COACH_WEIGHTS + "' 2>/dev/null; "
        "echo '---FLAG'; cat '" + SPIRAL_FLAG + "' 2>/dev/null; "
        "echo '---FILT'; grep -a '教练行剔除' '" + TRAINING_LOG + "' 2>/dev/null | tail -8; "
        # 本轮从哪代起 (回放池文件编号) —— 用于把体检序列/熔断序列裁到本轮, 免得跨轮误读
        "echo '---POOL'; ls '" + REPLAY_POOL_DIR + "'/iter*.pkl 2>/dev/null | sed 's#.*/iter##; s#.pkl##' | sort -n | tr '\\n' ' '; "
        # 晚到局结转 (LATE_MERGE=1 才有): 每代并入多少局 + 池变化
        "echo; echo '---LATE'; tail -6 '" + LATE_MERGE_LOG + "' 2>/dev/null",
        # 门控: 判决历史(gate_history.json) + 汇总表(gate_summary.tsv) + 分片明细 mtime/行数
        #   —— 用于「门控面板」; v38 门控 OFF → 大概率全空 (前端显示 OFF, 不显示"未启门控")
        "echo '===GATE==='; cat '" + SSH_CHECKPOINT_DIR + "/gate_history.json' 2>/dev/null | tr -d '\\n'; "
        "echo; echo '---TSV'; tail -6 '" + GATE_LOG_DIR + "/gate_summary.tsv' 2>/dev/null; "
        "echo '---SHARDS'; ls -l --time-style=+%s '" + GATE_LOG_DIR + "/*_shard*.jsonl' 2>/dev/null "
        "| sort -k6 -n | tail -5",
    ]
    for i in range(WORKER_NUM):
        # ★ 2026-11-15: tail -600 —— 工兵日志很小，全量拿回来才能按代聚合黑白胜
        #   (旧 tail -6 只覆盖最近 2-3 局，白胜率趋势无从谈起)
        # 用于解析 代/局 进度 与 本代黑白胜负（优化版主进程不再打印对局行）
        parts.append("echo '===W" + str(i) + "==='; tail -600 '" + WORKER_LOGS[i] + "' 2>/dev/null")
    raw = ssh_cmd('; '.join(parts))

    # 按 ===TAG=== 切段：re.search 容忍粘连行，标记前后残留（如 } 或 {）分别归还给前/后段
    segs = {}
    cur = None
    for line in raw.split('\n'):
        m = re.search(r'===(\w+)===', line)
        if m:
            prefix = line[:m.start()].strip()
            if prefix and cur is not None:
                segs[cur].append(prefix)
            cur = m.group(1)
            segs[cur] = []
            suffix = line[m.end():].strip()
            if suffix:
                segs[cur].append(suffix)
        elif cur is not None:
            segs[cur].append(line)

    log_lines = [l for l in segs.get('LOG', []) if l.strip()]
    data['log'] = log_lines

    # ── metadata 提前解析（当前代 + 运行配置都要用） ──
    meta_json, meta_cfg = {}, {}
    try:
        meta_json = json.loads('\n'.join(segs.get('META', [])))
        meta_cfg = meta_json.get('config', {}) or {}
    except Exception as e:
        print(f'[meta] {e}')
    # 🚨 2026-11-15: metadata.config 可能为空（bootstrap 写的 random-init 档），
    #   此时回落到 slot_config 的工兵数 / 10局，而不是瞎猜 20/5（v24b 实为 9/10/10）
    data['worker_count'] = meta_cfg.get('worker_count') or WORKER_NUM
    data['worker_games_per_worker'] = meta_cfg.get('worker_games_per_worker') or 10
    data['main_episodes_per_iter'] = meta_cfg.get('train_episodes_per_iteration') or 10

    # ── 每代耗时（攒批推理提速效果的直接证据） ──
    # 优先用权重文件 mtime (跨进程重启仍连续)，日志时间戳作为补充/兜底
    # 日志里的「检查点已保存 (iteration N)」序列另作当前代兜底推断用
    ck_seq = []
    for line in log_lines:
        m = re.search(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*检查点已保存 \(iteration (\d+)\)', line)
        if m:
            ck_seq.append((m.group(1), int(m.group(2))))
    try:
        from datetime import datetime as _dt
        ck_ts = []
        for line in segs.get('CKTS', []):
            parts = line.split()
            if len(parts) >= 7:
                m = re.search(r'weights_it(\d+)', parts[-1])
                if m and parts[5].isdigit():
                    ck_ts.append((int(parts[5]), int(m.group(1))))
        durs = []
        if len(ck_ts) >= 2:
            ck_ts.sort()
            for (t1, _i1), (t2, i2) in zip(ck_ts, ck_ts[1:]):
                durs.append({'iter': i2, 'min': round((t2 - t1) / 60.0, 1),
                             't': _dt.fromtimestamp(t2).strftime('%H:%M')})
        else:
            # 兜底: 日志里相邻两条「检查点已保存 (iteration N)」
            for (ts1, _i1), (ts2, i2) in zip(ck_seq, ck_seq[1:]):
                mins = (_dt.strptime(ts2, '%Y-%m-%d %H:%M:%S')
                        - _dt.strptime(ts1, '%Y-%m-%d %H:%M:%S')).total_seconds() / 60
                durs.append({'iter': i2, 'min': round(mins, 1), 't': ts2[11:16]})
        # 过滤异常值（进程重启/人工干预造成的超长间隔 > 3 倍中位数 或 > 180min）
        if durs:
            vals = sorted(x['min'] for x in durs)
            med = vals[len(vals) // 2]
            durs = [x for x in durs if 0 < x['min'] <= max(180, med * 3)]
        if durs:
            data['iter_durations'] = durs[-12:]
            data['iter_dur_last'] = durs[-1]['min']
            _t3 = [d['min'] for d in durs[-3:]]
            data['iter_dur_avg3'] = round(sum(_t3) / len(_t3), 1)
    except Exception as e:
        print(f'[iter-dur] {e}')

    # ── 当前代 / 本代局进度 ──
    # 🚨 2026-09-13 攒批推理优化版：主进程不再自对弈（train_episodes_per_iteration=0），
    #   主日志改用「等待工兵: N/100局已到」报进度，旧版的「代N 局M/10」行没有了
    cur_game = cur_denom = None
    for line in reversed(log_lines):
        m = re.search(r'等待工兵:\s*(\d+)/(\d+)局已到', line)
        if m:
            cur_game, cur_denom = int(m.group(1)), int(m.group(2))
            break
    if cur_denom is None:
        # 兼容旧版（本地推理时代，主进程自己产局）
        for line in reversed(log_lines):
            m = re.search(r'代(\d+) 局(\d+)/(\d+)', line)
            if m:
                cur_game, cur_denom = int(m.group(2)), int(m.group(3))
                break
    # 当前代: metadata.iteration 就是「正在产局的那一代」(save_checkpoint 里写 iteration+1)；
    # 兜底取 检查点最大代+1 / 工兵日志最新代，三者取最大（各自都可能滞后一拍）
    cand = []
    _mi = meta_json.get('iteration')
    if _mi is not None:
        try:
            cand.append(int(_mi))
        except (TypeError, ValueError):
            pass
    if ck_seq:
        cand.append(max(i for _, i in ck_seq) + 1)
    worker_its_early = []
    for i in range(WORKER_NUM):
        for line in segs.get('W' + str(i), []):
            m = re.search(r'工兵\d+ 代(\d+) 局', line)
            if m:
                worker_its_early.append(int(m.group(1)))
    if worker_its_early:
        cand.append(max(worker_its_early))
    cur_it = max(cand) if cand else None
    data['current_it'] = cur_it
    data['current_game'] = cur_game
    data['current_denom'] = cur_denom

    # ── 第二次连接：服务器当前时间 + 本代产局起点 + 攒批推理服务存活 ──
    # 注意：grep 模式里避免 [ ] 方括号（expect Tcl 命令替换陷阱）
    try:
        raw_t = ssh_cmd("date '+%Y-%m-%d %H:%M:%S'; date '+%s'; "
                        "grep -a '个工兵(v' '" + TRAINING_LOG + "' | tail -1; "
                        "echo '---START2'; grep -a '训练启动' '" + TRAINING_LOG + "' | tail -1; "
                        "echo '---INFAGE'; stat -c %Y '" + INFER_LOG + "' 2>/dev/null; "
                        # ★ 本代真实产局起点（2026-09-26 用户报"时间统计不太对"）：
                        #   metadata.json mtime = 主进程写"本代 iteration"的时刻 ≈ 本代开产时刻
                        #   本代各局文件 mtime  = 逐局落盘时刻（首局比开产晚一局耗时）
                        #   旧锚点「工兵已重启/进程启动」在主进程中途重启时会漂到进程启动 ✗
                        "echo '---GENSTAT'; stat -c '%Y|%n' " + SSH_CHECKPOINT_DIR + "/worker_data/iter*.pkl 2>/dev/null | tail -140; "
                        "echo '---METASTAT'; stat -c %Y " + SSH_CHECKPOINT_DIR + "/metadata.json 2>/dev/null; "
                        # ★ 服务器【实际在跑】哪个槽 (2026-09-27): 配置(active_slot.json) 与实跑不一致
                        #   时前端挂 ⚠ —— 防"整轮用错网络/看错槽"这类地雷 ✗
                        "echo '---SLOTRUN'; pgrep -af 'Gomoku-v0_pt_v' | head -2")
        ts_now = ts_start = ts_restart = ts_prog = None
        now_epoch = inf_mtime = meta_ts = None
        _slot_running = None
        genstat = {}          # 代号 -> [各局文件 mtime(epoch), ...]
        _after = ''
        for l in [x for x in raw_t.split('\n') if x.strip()]:
            if l.strip().startswith('---'):
                _after = l.strip().strip('-')
                continue
            if _after == 'GENSTAT':
                mg = re.match(r'^(\d{10})\|.*iter(\d+)_', l.strip())
                if mg:
                    genstat.setdefault(int(mg.group(2)), []).append(int(mg.group(1)))
                continue
            mnum = re.match(r'^(\d{9,11})$', l.strip())
            if mnum:
                if _after == 'INFAGE':
                    inf_mtime = int(mnum.group(1))
                elif _after == 'METASTAT':
                    meta_ts = int(mnum.group(1))
                elif now_epoch is None:
                    now_epoch = int(mnum.group(1))
                continue
            mts = re.match(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', l)
            if mts:
                if ts_now is None:
                    ts_now = mts.group(1)        # 第一行 = 服务器当前时间
                elif _after == 'START2':
                    ts_prog = mts.group(1)       # 「训练启动」= 进程启动时刻
                elif ts_restart is None:
                    ts_restart = mts.group(1)    # 「N个工兵(v36)已重启」= 旧的产局起点锚
                continue
            if _after == 'SLOTRUN':
                msr = re.search(r'Gomoku-v0_pt_(v\d+)\.py', l)
                if msr:
                    _slot_running = msr.group(1)
        data['slot_running'] = _slot_running
        data['slot_config'] = TRAIN_VERSION
        data['slot_mismatch'] = bool(_slot_running and _slot_running != TRAIN_VERSION)
        from datetime import datetime as _dt, timedelta as _td
        data['iter_now'] = ts_now
        # ★ 本代时间锚（2026-09-26 修）：
        #   首选 = 本代局文件 mtime (真实产局, 重启也不失真)
        #   次选 = metadata.json mtime (≈开产时刻, 比首局落盘早一局)
        #   兜底 = 旧的「工兵已重启」/「进程启动」（主进程中途重启时会漂 → 只作最后兜底）
        gen_now = max(genstat) if genstat else None
        # 🚨 2026-11-15 修: 原写 `if gen_now` —— gen_now==0 (第0代) 是 falsy
        #    → gen_first=None → 下面 _dt.fromtimestamp(None) 抛 TypeError, 整段代次时间锚全丢。
        #    老版本从 it30+ 开始监控, 从没踩到过。
        gen_first = min(genstat[gen_now]) if gen_now is not None else None
        data['gen_now'] = gen_now
        if gen_now is not None:
            data['gen_first_time'] = _dt.fromtimestamp(gen_first).strftime('%Y-%m-%d %H:%M:%S')
        if gen_first and meta_ts:
            # 两者取较早者: 正常 = metadata 早约一局耗时; 若 metadata 被中途重写过则局文件更早
            ts_start_epoch, anchor = (min(gen_first, meta_ts),
                                      'gen_file' if gen_first <= meta_ts else 'gen_meta')
        elif gen_first:
            ts_start_epoch, anchor = gen_first, 'gen_file'
        elif meta_ts:
            ts_start_epoch, anchor = meta_ts, 'gen_meta'
        else:
            _cands = [x for x in (ts_restart, ts_prog) if x]
            ts_start_epoch = None
            anchor = None
            if _cands:
                _ts = max(_cands)
                ts_start_epoch = _dt.strptime(_ts, '%Y-%m-%d %H:%M:%S').timestamp()
                anchor = 'worker_restart' if _ts == ts_restart else 'proc_start'
        data['iter_anchor'] = anchor
        data['iter_start_epoch'] = ts_start_epoch
        if ts_start_epoch and ts_now:
            elapsed_s = max(0, _dt.strptime(ts_now, '%Y-%m-%d %H:%M:%S').timestamp() - ts_start_epoch)
            data['iter_start_time'] = _dt.fromtimestamp(ts_start_epoch).strftime('%Y-%m-%d %H:%M:%S')
            data['iter_elapsed_min'] = round(elapsed_s / 60, 1)
            data['iter_rate'] = round(elapsed_s / max(1, cur_game or 0), 1)
        if now_epoch and inf_mtime:
            age = max(0, now_epoch - inf_mtime)
            data['infer_age_s'] = age
            data['infer_alive'] = age < 180      # 统计行每 30s 一条，180s 未更新 = 停/未启用
    except Exception as e:
        import traceback as _tb
        print(f'[train-time] {e}')
        print(_tb.format_exc())

    # ── 主进程阶段 ──
    last = log_lines[-1] if log_lines else ''
    if cur_it is None:
        data['main_status'] = '启动中'
        data['main_phase'] = 'startup'
    elif '等待工兵' in last:
        data['main_status'] = (f'代{cur_it} 等工兵 {cur_game}/{cur_denom}'
                               if cur_game is not None else f'代{cur_it} 等工兵')
        data['main_phase'] = 'waiting_workers'
    elif '学习完成' in last:
        data['main_status'] = f'代{cur_it} 学习完成 ✅'
        data['main_phase'] = 'learning_done'
    elif '检查点已保存' in last:
        data['main_status'] = f'代{cur_it} 检查点 ✅'
        data['main_phase'] = 'checkpoint'
    elif '健康体检' in last:
        data['main_status'] = f'代{cur_it} 体检完成'
        data['main_phase'] = 'checkpoint'
    elif '工兵' in last and '重启' in last:
        data['main_status'] = f'代{cur_it} 工兵重启中'
        data['main_phase'] = 'waiting_workers'
    elif '自对弈完成' in last:
        # 优化版：主进程不再产局（train_episodes_per_iteration=0），该行只报告「主进程 0 局」
        data['main_status'] = f'代{cur_it} 工兵产局中'
        data['main_phase'] = 'selfplay'
    elif ('收集工兵' in last) or ('经验回放池' in last) or ('训练 ' in last):
        data['main_status'] = f'代{cur_it} 汇总/学习中'
        data['main_phase'] = 'learning'
    elif cur_game and cur_denom and cur_game < cur_denom:
        data['main_status'] = f'代{cur_it} 局{cur_game}/{cur_denom}'
        data['main_phase'] = 'selfplay'
    elif '演示' in last:
        data['main_status'] = f'代{cur_it} 演示中'
        data['main_phase'] = 'demo'
    else:
        data['main_status'] = f'代{cur_it} 就绪'
        data['main_phase'] = 'idle'

    # 本代实际收齐局数（收集工兵行；≥ 目标局数 = 有工兵多产了零头）
    for line in reversed(log_lines):
        m = re.search(r'收集工兵:\s*(\d+)局', line)
        if m:
            data['collected_games'] = int(m.group(1))
            break

    # sim: 优先日志 sim=(实时), 其次 metadata config.sim_count, 兜底 800 (v8 起固定 800)
    sim = None
    for line in reversed(log_lines):
        m = re.search(r'sim=(\d+)', line)
        if m:
            sim = int(m.group(1)); break
    if sim is None:
        try:
            sim = int(meta_cfg.get('sim_count', 800))
        except Exception:
            sim = 800
    data['sim_count'] = sim

    # ── 工兵（数量取 metadata.config.worker_count，SSH 拉取上限 WORKER_NUM） ──
    # 🚨 2026-09-13: 9 → 20 工兵。每工兵产局配额 = 目标局数 / 工兵数
    #   (20×5=100 → 均摊 5 局；主进程收满 100 就推进下一代，所以途中会出现「局N/10 未跑满」)
    quota = round((cur_denom or 100) / max(1, data['worker_count']), 1)
    # ★ 2026-10-11 用户: "监控面板，工兵数量和每工兵目标看看修改下"
    #   每工兵目标优先用 metadata.config.worker_games_per_worker (真值, v42=11);
    #   旧的"目标局数/工兵数"均摊算法不准 (v42 实测 100/10=10 ✗ 但日志是 局N/11 ✓)
    quota = data.get('worker_games_per_worker') or quota
    workers = []
    bw_cur = {'black': 0, 'white': 0}
    stale_workers = 0
    #   生产代的编号比 metadata.iteration 小 1 (口径: metadata 提前一代广播;
    #   工兵日志写的是【数据文件名那代】, 如主进程 it65 时工兵写"代64 局k/6")
    prod_it = (cur_it - 1) if isinstance(cur_it, int) else None
    # ★ 工兵格数量: metadata.worker_count 只是【配置值】—— 实测 v42 有 20+ 个工兵日志/进程
    #   (自动换边会分批起工兵, 进程数在 21~30 间浮动) ⇒ 按实际拉到的工兵段扩到 max_id+1,
    #   否则后一半工兵在面板上根本不存在 ✗ (用户 2026-10-11 指出)
    _seen_wi = [int(k[1:]) for k in segs if k.startswith('W') and k[1:].isdigit()]
    _n_cells = max([data['worker_count']] + ([max(_seen_wi) + 1] if _seen_wi else []))
    data['worker_cells'] = _n_cells
    for i in range(_n_cells):
        w = {'id': i, 'status': '—', 'game': 0, 'iter': None, 'denom': None,
             'quota': quota, 'black': 0, 'white': 0}
        seen = {}   # 该工兵日志尾部出现的各代进度
        for line in segs.get('W' + str(i), []):
            m = re.search(r'工兵\d+ 代(\d+) 局(\d+)(?:/(\d+))?(?::\s*(黑|白)胜)?', line)
            if not m:
                continue
            it, g = int(m.group(1)), int(m.group(2))
            d = int(m.group(3)) if m.group(3) else 0
            st = seen.setdefault(it, {'game': 0, 'denom': 0, 'black': 0, 'white': 0})
            st['game'] = max(st['game'], g)
            if d:
                st['denom'] = d
            if m.group(4):
                st['black' if m.group(4) == '黑' else 'white'] += 1
        if seen:
            w['iter'] = max(seen)
            st = seen[w['iter']]
            w['game'] = st['game']
            w['denom'] = st['denom'] or 10
            w['black'], w['white'] = st['black'], st['white']
            w['status'] = f"代{w['iter']} 局{w['game']}/{w['denom']}"
            # ★ 当前代的黑白计数: 工兵日志的代号 = metadata.iteration 或 (metadata.iteration-1) 都算本轮
            if w['iter'] in (cur_it, prod_it):
                bw_cur['black'] += st['black']
                bw_cur['white'] += st['white']
            elif w['iter'] is not None and prod_it is not None and w['iter'] < prod_it:
                stale_workers += 1
                w['status'] += ' ⚠落后'
        else:
            for line in reversed(segs.get('W' + str(i), [])):
                if '重启衔接' in line:
                    w['status'] = '衔接中'; break
                if '已启用远程攒批推理' in line:
                    w['status'] = '已就绪(远程推理)'; break
        workers.append(w)
    data['workers'] = workers
    data['bw_current'] = {'iter': cur_it, **bw_cur}
    data['workers_stale'] = stale_workers

    # ★ 2026-11-15: 每代黑白胜局聚合（白胜率趋势的数据源）——
    #   v24/v24b 家族主进程不打印「本代白方胜率」，工兵日志才是每局真值的唯一来源。
    #   每行形如「工兵0 代0 局1/10: 黑胜 37条」；重启后工兵会重进同代但只补缺失局
    #   （已存在的打「已存在，跳过」，不带胜负），所以按 代×wid×局号 去重后不会重复计数。
    bw_gen = {}
    for _k in segs:
        if not (_k.startswith('W') and _k[1:].isdigit()):
            continue
        _wid = int(_k[1:])
        for _ln in segs[_k]:
            _m = re.search(r'工兵\d+ 代(\d+) 局(\d+)(?:/(\d+))?:\s*(黑|白)胜', _ln)
            if not _m:
                continue
            _it, _g, _w = int(_m.group(1)), int(_m.group(2)), _m.group(4)
            _slot = bw_gen.setdefault(_it, {'games': {}})
            _slot['games'][(_wid, _g)] = _w
    # ★ 2026-11-15: 逐代累计落到本地磁盘 —— 服务器主日志/工兵日志被覆盖时(重启用 > 而非 >>)
    #   历史代就全丢了，面板会突然"没数"。本地存一份合并结果，重启/覆盖都不影响。
    _BW_STORE = os.path.join(PARENT_DIR, f'.bw_store_{TRAIN_VERSION}.json')
    _bw_persist = {}
    try:
        if os.path.exists(_BW_STORE):
            _bw_persist = json.load(open(_BW_STORE))
    except Exception:
        _bw_persist = {}
    for _it, _v in bw_gen.items():
        _gk = _v['games']
        _b = sum(1 for x in _gk.values() if x == '黑')
        _w2 = sum(1 for x in _gk.values() if x == '白')
        # 只增不减：同一代取较大的局数（日志窗口可能只覆盖到部分局）
        _prev = _bw_persist.get(str(_it)) or {}
        if (_b + _w2) >= (int(_prev.get('black', 0)) + int(_prev.get('white', 0))):
            _bw_persist[str(_it)] = {'black': _b, 'white': _w2}
    try:
        json.dump(_bw_persist, open(_BW_STORE, 'w'))
        bw_gen = {int(k): {'games': {}} for k in _bw_persist}
        for k, v in _bw_persist.items():
            bw_gen[int(k)]['games'] = v      # 直接存 {'black','white'} 形式
    except Exception as _e:
        print(f'[bw-store] {_e}')

    if bw_gen:
        _gits, _gb, _gw, _gt, _gwr = [], [], [], [], []
        for _it in sorted(bw_gen):
            _gk = bw_gen[_it]['games']
            if 'black' in _gk:                       # 来自本地持久化
                _b, _w2 = int(_gk['black']), int(_gk['white'])
            else:                                    # 来自日志解析
                _b = sum(1 for v in _gk.values() if v == '黑')
                _w2 = sum(1 for v in _gk.values() if v == '白')
            _n = _b + _w2
            _gits.append(_it); _gb.append(_b); _gw.append(_w2); _gt.append(_n)
            _gwr.append(round(100.0 * _w2 / _n, 1) if _n else None)
        data['bw_per_gen'] = {'iterations': _gits, 'black': _gb, 'white': _gw,
                              'total': _gt, 'white_rate': _gwr}
        # 白胜率趋势（跨代）—— 页面「白胜率」曲线直接吃这个
        data['white_rate_trend'] = {'iterations': _gits, 'rate': _gwr, 'total': _gt}

    # ★ 2026-11-15: 主进程也产局的版本（v24/v24b 家族）本代总局数 =
    #     工兵数 × 每工兵局数 + 主进程局数，而不是主日志里那个"每工兵 10 局"的分母。
    #   旧版(v36~v42) main_episodes_per_iter=0 → 公式退化成 工兵数×局数，与原口径一致。
    if cur_it is not None:
        _wg = sum(w.get('game') or 0 for w in workers
                  if w.get('iter') == prod_it) or sum(w.get('game') or 0 for w in workers)
        _total_target = (data['worker_count'] * data['worker_games_per_worker']
                         + (data.get('main_episodes_per_iter') or 0))
        _main_done = 0
        for line in reversed(log_lines):
            m = re.search(r'代%d\s+局(\d+)/(\d+)' % cur_it, line)
            if m:
                _main_done = int(m.group(1))
                break
        if _total_target > 0:
            data['current_denom'] = _total_target
            data['denom_includes_main'] = True   # 前端不要再 + 主进程局数
        if data.get('main_episodes_per_iter'):
            data['current_game'] = min(_wg + _main_done, _total_target)
        data['main_games_done'] = _main_done

    # ── 攒批推理服务统计（独立进程独占 GPU；统计行每 30s 一条） ──
    infer = {'alive': bool(data.get('infer_alive')), 'age_s': data.get('infer_age_s')}
    inf_lines = segs.get('INF', [])
    for line in reversed(inf_lines):
        m = re.search(r'统计:\s*请求\s*(\d+)\s*\|\s*forward\s*(\d+)\s*\|\s*平均batch\s*([\d.]+)'
                      r'\s*\|\s*(\d+)\s*req/s\s*\|\s*forward均耗时\s*([\d.]+)ms', line)
        if m:
            infer.update({'requests': int(m.group(1)), 'forwards': int(m.group(2)),
                          'avg_batch': float(m.group(3)), 'qps': int(m.group(4)),
                          'forward_ms': float(m.group(5))})
            break
    _sock = [l.strip() for l in inf_lines if l.strip().isdigit()]
    infer['sock'] = bool(_sock and _sock[-1] == '1')
    data['infer'] = infer

    # ── v37 门控赛 (AGZ 风格: 每 10 代 候选 vs best 配对赛, 胜率≥55% 晋升, 否则回滚) ──
    #   gate_history.json = 每次判决一行(紧凑); gate_summary.tsv = 含置信区间/黑白分色的汇总;
    #   分片 jsonl 的 mtime/行数 → 判断"正在跑"与进度
    gate = {'enabled': None, 'history': [], 'summary': [], 'running': False}
    g_lines = segs.get('GATE', [])
    try:
        _gj = [l for l in g_lines if l.strip().startswith('[')]
        if _gj:
            raw_hist = json.loads(_gj[-1])
            # 归一化: gate_history.json 有三种条目格式 (旧/作废/手动重跑) —— 键名不同, 统一成一套
            #   ⚠️ 2026-09-25 实踩: 手动重跑条目用 games/sim/elapsed_min/best_promoted, 且
            #      cand_as_black 是 "50/50 (100%)" 这种带百分比的字符串; 直接用 h['n_games'] 会 KeyError/漏显示
            for h in raw_hist:
                it = h.get('iteration', h.get('iter'))
                gate['history'].append({
                    'iteration': it,
                    'time': h.get('time', ''),
                    'verdict': (h.get('verdict') or '').upper(),
                    'rate': h.get('rate'),
                    'games': h.get('n_games', h.get('games')),
                    'cand_wins': h.get('cand_wins'),
                    'best_wins': h.get('best_wins'),
                    'draws': h.get('draws'),
                    'as_black': h.get('cand_as_black'),
                    'as_white': h.get('cand_as_white'),
                    'rolled_back': bool(h.get('rolled_back')),
                    'no_promote': bool(h.get('no_promote')),
                    'manual_rerun': bool(h.get('manual_rerun')),
                    'best_promoted': h.get('best_promoted'),
                    'sim': h.get('sim'),
                    'elapsed_min': h.get('elapsed_min'),
                    'note': h.get('note', ''),
                })
    except Exception as e:
        print(f'[gate-hist] {e}')
    for line in g_lines:                      # tsv: tag n_games best_wins cand_wins draws rate ci_lo ci_hi 黑白 先手 verdict
        p = line.split('\t')
        if len(p) >= 12 and p[0] not in ('tag',):
            try:
                gate['summary'].append({'tag': p[0], 'n': int(p[1]),
                                        'best': int(p[2]), 'cand': int(p[3]), 'draws': int(p[4]),
                                        'rate': float(p[5]), 'ci_lo': float(p[6]), 'ci_hi': float(p[7]),
                                        'as_black': p[8], 'as_white': p[9],
                                        'black_adv': float(p[10]), 'verdict': p[11].strip()})
            except Exception:
                pass
    # 门控配置行 (主日志): v37 门控: ON | 每 10 代 | 100 局配对 | 阈值 55% | 回滚 ON(方案B)
    for line in reversed(log_lines):
        m = re.search(r'门控:\s*(ON|OFF)\s*\|\s*每\s*(\d+)\s*代\s*\|\s*(\d+)\s*局配对\s*\|\s*阈值\s*([\d.]+)%', line)
        if m:
            gate['enabled'] = (m.group(1) == 'ON')
            gate['every'] = int(m.group(2))
            gate['games'] = int(m.group(3))
            gate['thresh'] = float(m.group(4)) / 100.0
            break
    # 门控开跑行: 【门控】代10: 候选 vs best, 100 局配对 (4 分片, sim=800) ...
    #   —— 实测值以这行为准 (脚本里的 GATE_SIM 可能被运行环境覆盖, 别按脚本假设)
    for line in reversed(log_lines):
        m = re.search(r'【门控】代(\d+):\s*候选 vs best,\s*(\d+)\s*局配对\s*\((\d+)\s*分片,\s*sim=(\d+)\)', line)
        if m:
            gate['iter_gate'] = int(m.group(1))
            gate['games'] = int(m.group(2))
            gate['shards'] = int(m.group(3))
            gate['sim'] = int(m.group(4))
            break
    # 最近一条【门控】行 (进行中/结果/回滚) —— 给前端当状态文案
    for line in reversed(log_lines):
        if '【门控】' in line:
            gate['last_line'] = re.sub(r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+ INFO ', '', line).strip()[:180]
            break
    # 分片 mtime → 是否正在跑 + 当前 tag 的已完成局数
    ck_now = None
    _now_s = data.get('iter_now')
    if _now_s:
        try:
            from datetime import datetime as _dt
            ck_now = _dt.strptime(str(_now_s), '%Y-%m-%d %H:%M:%S').timestamp()
        except Exception:
            ck_now = None
    shard_ts = []
    for line in g_lines:
        p = line.split()
        if len(p) >= 7 and p[-1].endswith('.jsonl'):
            try:
                shard_ts.append((int(p[5]), p[-1]))
            except Exception:
                pass
    if shard_ts and ck_now:
        newest_t, newest_f = max(shard_ts)
        gate['age_s'] = int(max(0, ck_now - newest_t))
        gate['running'] = gate['age_s'] < 900          # 15 分钟内分片还在写 = 进行中
        gate['tag'] = newest_f.split('/')[-1].rsplit('_shard', 1)[0]
        # 当前 tag 各分片累计局数 = 门控进度 (jsonl 每行一局)
        # ⚠️ v38 门控 OFF → 不再发这条 SSH 查询 (省一次往返; 旧分片残留不该被当"进行中")
        if gate['running'] and gate.get('enabled') is not False:
            try:
                _cnt = ssh_cmd("wc -l '" + GATE_LOG_DIR + "'/" + gate['tag'] + "_shard*.jsonl 2>/dev/null | "
                               "grep -c . ; wc -l '" + GATE_LOG_DIR + "'/" + gate['tag'] + "_shard*.jsonl "
                               "2>/dev/null | awk '{s+=$1} END {print s}'")
                _nums = [int(x) for x in _cnt.split() if x.strip().isdigit()]
                if _nums:
                    gate['progress'] = _nums[-1]
            except Exception as e:
                print(f'[gate-prog] {e}')
    # 下一次门控 = 下一个 every 的整数倍
    if gate.get('enabled') and gate.get('every') and cur_it:
        nxt = ((cur_it // gate['every']) + 1) * gate['every']
        gate['next_iter'] = nxt
        gate['to_next'] = nxt - cur_it
    data['gate'] = gate
    # ── 体检趋势 (health_log.json): 空盘 value = v38 熔断器输入 (最重要的新指标) ──
    #   还带: 白1距黑1 中位 (基线14/目标≤4) + 激活率 + 判别度(=黑活四 − 空盘)
    #   数据源改为【原始 JSON 尾部 + 本地解析】—— 空盘是中文键(\\uXXXX 转义), grep 拿不到 ✗
    health = []
    try:
        raw_h = '\n'.join(segs.get('HEALTH', []))
        # 🚨 tail -c 必然把头部切在任意字节 → 可能切进字符串/数字中间, 括号配对随之失准 (实测偶发解析为空 ✗)
        #   解法: 从第一个"行首 { "处开始 (health_log 每条目以 ' {' 起行, 见文件格式) → 保证从条目边界起解析
        _k = raw_h.find('\n {')
        if _k < 0:
            _k = raw_h.find('\n{')
        if _k >= 0:
            raw_h = raw_h[_k:]
        for obj in _iter_json_objects(raw_h):
            try:
                e = json.loads(obj)
            except Exception:
                continue
            if 'iteration' not in e:
                continue
            _empty = e.get('value_\u7a7a\u76d8')                 # 空盘 value
            _live4 = e.get('value_\u9ed1\u6d3b\u56db')            # 黑活四 (高判别度锚点)
            _disc = None
            if isinstance(_empty, (int, float)) and isinstance(_live4, (int, float)):
                _disc = round(_live4 - _empty, 4)               # 判别度 <0.15 = 红线
            health.append({'iter': e.get('iteration'), 'act': e.get('act_mean'),
                           'w1': e.get('w1_dist_med'), 'w1all': e.get('w1_dist_all'),
                           'top1': e.get('w1_top1_med'), 'dead': e.get('bn_dead'),
                           'bntot': e.get('bn_total'), 'empty': _empty, 'live4': _live4,
                           'disc': _disc})
    except Exception as e:
        print(f'[health] {e}')
    data['health'] = health[-6:]

    # ── v38 教练学习 (冻结黑方教练 + 只训白方) ──
    #   教练色从日志取 (教练行剔除行的「(教练色=1)」/启动行「冻结黑方教练」), 不假设
    coach = {'color': None, 'color_name': '黑', 'goal': COACH_GOAL_RATE, 'filt': [], 'wsize': None,
             'from': COACH_FROM}
    try:
        # ① 首选 = 工兵日志「工兵0 教练已加载: weights_coach.pt | 执白 | 本地推理(无remote)」← 最权威
        #    🚨 v39 实踩: 主日志的启动横幅是 v38 复制来的, 仍写「冻结黑方教练」(文案残留) ✗
        #       → 若信横幅会把 v39 的教练色判反 (镜像版), 进而把学生胜率算成 100−x ✗✗
        for _wi in range(WORKER_NUM):
            for _l in segs.get('W' + str(_wi), []):
                m = re.search(r'教练已加载.*?\|\s*执(黑|白)', _l)
                if m:
                    coach['color'] = 1 if m.group(1) == '黑' else -1
                    break
            if coach['color'] is not None:
                break
        # ② 其次 = 主日志「(教练色=-1)」 ③ 兜底 = 横幅「冻结X方教练」(可能是残留文案, 仅当①②都没有)
        if coach['color'] is None:
            for line in reversed(log_lines):
                m = re.search(r'教练色=(-?\d+)', line)
                if m:
                    coach['color'] = int(m.group(1))
                    break
        if coach['color'] is None:
            for line in reversed(log_lines):
                m = re.search(r'冻结([黑白])方教练', line)
                if m:
                    coach['color'] = 1 if m.group(1) == '黑' else -1
                    break
        coach['color_name'] = '黑' if coach['color'] == 1 else ('白' if coach['color'] == -1 else '?')
        # 学生 = 非教练色 (v38 教练执黑→学生练白; v39 教练执白→学生练黑)
        # 🚨 2026-11-15: 无教练的槽(v40/v41/v24b) coach['color'] 恒 None →
        #   旧写法把它判成"教练执白 → 学生黑"，主指标变成黑方胜率 ✗
        #   对无教练线，主指标应由 slot_config 指定要观察的执色（v24b = 白）。
        if coach['color'] is None:
            coach['student_name'] = ACTIVE_SLOT.get('student_name') or '白'
            coach['coachless'] = True
        else:
            coach['student_name'] = '白' if coach['color'] == 1 else '黑'
            coach['coachless'] = False
        coach['coach_white'] = (coach['color'] == -1)
        # 教练快照 (stat -c '%Y %s' → mtime size)
        c_lines = segs.get('COACH', [])
        # 教练快照 (stat -c '%Y %s' → 段首的 "mtime size" 一行)
        for _ln in c_lines:
            _p = _ln.strip().split()
            if len(_p) == 2 and _p[0].isdigit() and _p[1].isdigit() and not _ln.strip().startswith('---'):
                coach['wmtime'], coach['wsize'] = int(_p[0]), int(_p[1])
                break
        # 按 ---TAG 切块 (FLAG=熔断标记 / FILT=教练行剔除 / POOL=本轮回放池编号 / LATE=晚到局日志)
        _blocks, _cur = {}, None
        for _ln in c_lines:
            _m = re.match(r'^---(\w+)\s*(.*)$', _ln.strip())
            if _m:
                _cur = _m.group(1)
                _blocks[_cur] = [_m.group(2).strip()] if _m.group(2).strip() else []
            elif _cur:
                _blocks.setdefault(_cur, []).append(_ln.strip())
        coach['flag'] = ' '.join(_blocks.get('FLAG', []))[:400] or None
        coach['pool_iters'] = sorted(int(x) for x in re.findall(r'\d+', ' '.join(_blocks.get('POOL', []))))
        # ★ 本轮起点 = 【worker_data 的最小代】(本轮实际产出的第一代), 优先于池编号
        #   为什么: 池的编号会被"补种"污染 —— v38 第二轮把第一轮的 iter60~63 补进了池(跳过 warmup),
        #   于是池最小代 = 60 ✗ 但第一轮的 health_log 里也正好有 60~63 这些编号 →
        #   按池编号裁体检序列会把【上一轮的读数】混进本轮 ✗✗ (实测: 代60~63 的空盘其实是 07:07~08:22 写的)
        _exp_iters = []
        for _l in segs.get('EXP', []):
            _m3 = re.search(r'iter(\d+)_', _l)
            if _m3:
                _exp_iters.append(int(_m3.group(1)))
        coach['worker_min'] = min(_exp_iters) if _exp_iters else None
        coach['round_start'] = coach['worker_min'] or (coach['pool_iters'][0] if coach['pool_iters'] else None)
        for line in c_lines + log_lines:
            m = re.search(r'教练行剔除:\s*(\d+)\s*→\s*(\d+)\s*条样本', line)
            if m:
                t, k = int(m.group(1)), int(m.group(2))
                if not coach['filt'] or coach['filt'][-1]['kept'] != k:
                    coach['filt'].append({'total': t, 'kept': k,
                                          'drop': round((1 - k / t) * 100, 1) if t else None})
        coach['filt'] = coach['filt'][-6:]
        if coach['filt']:
            coach['drop_now'] = coach['filt'][-1]['drop']
            coach['kept_now'] = coach['filt'][-1]['kept']
        # 晚到局结转 (GOMOKU_LATE_MERGE=1): 每代「itN: X 局未进池 → MODE=on: 已并入 (池 A→B局)」
        #   为什么值得上卡: 关闭时每代会【静默丢弃 ~16% 的长局】✗ 而白方这一轮学的正是中后盘
        _late = [l for l in _blocks.get('LATE', []) if l]
        if _late:
            _m2 = re.search(r'it(\d+)\s*\+?(\d*)局?.*?池\s*(\d+)局', _late[-1])
            coach['late'] = {'last': _late[-1][:170], 'lines': len(_late),
                             'iter': int(_m2.group(1)) if _m2 else None,
                             'merged': int(_m2.group(2)) if (_m2 and _m2.group(2)) else None,
                             'pool': int(_m2.group(3)) if _m2 else None}
        else:
            coach['late'] = None
    except Exception as e:
        print(f'[coach] {e}')
    data['coach'] = coach

    # ── v38 空盘value 熔断器状态 (设计文档 §①) ──
    #   判据: 连续 3 代上升 且 空盘 > 0.25 → 回滚到 <0.15 的健康点; 30 代内 2 次 → 停训
    #   ⚠️ 单代尖峰会自愈 (v37 it14 +0.612→it15 +0.074), 所以看【连续上升】而非单代超阈
    spiral = {'empty': None, 'rise': 0, 'series': [], 'trips': 0, 'guard_err': 0,
              'flag': coach.get('flag'), 'thr': SPIRAL_THR, 'healthy': SPIRAL_HEALTHY,
              'need_rise': SPIRAL_RISE, 'status': 'ok', 'disc': None}
    try:
        # ★ 先把体检/熔断序列裁到【本轮】(回放池最小代起) —— 槽目录的 health_log.json 跨轮累积,
        #   不裁就会把上一轮的代混进"连续上升"判断和空盘走势里 ✗ (2026-09-26 双槽复用后必做)
        _rs = coach.get('round_start')
        if _rs:
            spiral['round_start'] = _rs
            health = [h for h in health if (h.get('iter') or 0) >= _rs]
            data['health'] = health[-6:]
        # 熔断模式: 第二轮起 start_v38.sh 默认 warn = 【只告警】不回滚/不停机
        spiral['mode'] = 'warn' if any('GOMOKU_SPIRAL_MODE=warn' in l for l in log_lines) else 'full'
        # ⚠️ 熔断标记文件是【跨轮累积】的 (v38 槽里还躺着第一轮 09-25 那条) → 只有 flag 里的
        #    iter >= 本轮起点才算"本轮触发", 否则会让卡片永久显示"已触发回滚" ✗
        _mfi = re.search(r'iter=(\d+)', spiral['flag'] or '')
        spiral['flag_iter'] = int(_mfi.group(1)) if _mfi else None
        spiral['flag_current'] = bool(_rs and spiral['flag_iter'] is not None and spiral['flag_iter'] >= _rs)
        _es = [(h['iter'], h['empty']) for h in health if isinstance(h.get('empty'), (int, float))]
        spiral['series'] = [{'iter': i, 'empty': round(v, 4)} for i, v in _es]
        if _es:
            spiral['empty'] = round(_es[-1][1], 4)
            _r = 0
            for k in range(len(_es) - 1, 0, -1):
                if _es[k][1] > _es[k - 1][1]:
                    _r += 1
                else:
                    break
            spiral['rise'] = _r
        spiral['disc'] = next((h['disc'] for h in reversed(health) if h.get('disc') is not None), None)
        # ⚠️ 只统计【本次主进程启动之后】的熔断行 —— 否则历史里一次"守卫报错"会让卡片永久显示告警 ✗
        #    (实测: 18:33 gen31 那次 `_spiral_hist` 未初始化异常, 18:37 重启后已修, 但日志留痕)
        _boot = 0
        for _i, _l in enumerate(log_lines):
            if 'v37 + 教练学习' in _l:
                _boot = _i
        _cur_lines = log_lines[_boot:]
        spiral['trips'] = sum(1 for l in _cur_lines if '【熔断】' in l and '检查异常' not in l)
        spiral['guard_err'] = sum(1 for l in _cur_lines if '【熔断】检查异常' in l)
        spiral['boot_line'] = _boot
        if not spiral['series']:
            spiral['status'] = 'nodata'                         # health 没采到 → 别显示"正常"骗人 ✗
        elif spiral['flag_current']:
            spiral['status'] = 'triggered'                      # 本轮真的回滚过 (标记文件存在)
        elif spiral['guard_err']:
            spiral['status'] = 'guard_err'                      # 守卫自身报错 → 该代没检查 (需人看)
        elif spiral['mode'] == 'warn' and spiral['rise'] >= spiral['need_rise'] and (spiral['empty'] or 0) > spiral['thr']:
            spiral['status'] = 'would_trip'                     # 条件成立, 但 warn 模式只告警 ✓
        elif spiral['rise'] >= spiral['need_rise'] and (spiral['empty'] or 0) > spiral['thr']:
            spiral['status'] = 'trip'                           # 条件已成立 (下一次检查应触发)
        elif spiral['rise'] >= spiral['need_rise'] - 1:
            spiral['status'] = 'warn'                           # 再升 1 代即触发
        elif spiral['empty'] is not None and spiral['empty'] > spiral['thr']:
            spiral['status'] = 'warn'                           # 绝对值已超, 只等连续上升
    except Exception as e:
        print(f'[spiral] {e}')
    data['spiral'] = spiral

    # ── v38 主指标: 每代「对固定教练的白胜率」 ──
    #   ① 优先用【主日志的实时行】(2026-09-25 晚 用户自己加的; 不依赖本地同步)——
    #         v38 本代白方胜率: 24.0% (24/100) | 历史 27% 16% 26% 18% 23% 24%
    #         v38 白方胜率历史(重启重建, 7代): 17% 19% 24% 19% 27% 16% 26%
    #      之前只读本地同步数据 → 卡上永远"落后服务器 N 代" ✗ (要点: 主指标已可直接从日志取)
    #   ② 中位手数 / 局数 从【本地已同步的自对弈数据】按代数对齐 (日志这两项不给)
    kpi = {'target': COACH_GOAL_RATE, 'gens': [], 'latest': None, 'lag_gens': [], 'source': None,
           'student_name': coach.get('student_name', '白'), 'coach_white': bool(coach.get('coach_white'))}
    try:
        # 🚨 口径转换 (镜像版必需): 数据/日志里的「白方胜率」
        #   v38 (教练执黑, 学生白) → 白方胜率 = 学生胜率 ✓ 直接用
        #   v39 (教练执白, 学生黑) → 白方胜率 = 教练胜率 ✗ 学生(黑)胜率 = 100 − 它
        #   (主日志那行文案 v39 仍写「本代白方胜率」= 复制残留, 数字确实是白方/教练的 ✓ 实盘核对过)
        _coach_white = bool(coach.get('coach_white'))
        _stu = (lambda r: round(100.0 - float(r), 1)) if _coach_white else (lambda r: round(float(r), 1))
        # (a) 本地索引: iter -> {n, rate, steps_med}
        from replay_api import load_index as _load_idx
        _local = {}
        _idx = _load_idx(TRAIN_VERSION)
        if _idx and _idx.get('iters'):
            for it in sorted(_idx['iters'], key=int):
                _games = _idx['iters'][it]['files']
                if not _games:
                    continue
                _w = sum(1 for g in _games if g['winner'] == -1)
                _st = sorted(g['steps'] for g in _games if g.get('steps'))
                _local[int(it)] = {'n': len(_games),
                                   'rate': _stu(round(_w / len(_games) * 100, 1)),
                                   'raw_white': round(_w / len(_games) * 100, 1),
                                   'steps_med': _st[len(_st) // 2] if _st else 0, 'src': 'local'}
        # (b) 主日志: 学生胜率 + 历史序列
        #   ⚠️ 两种标签都要吃 (2026-09-26 训练脚本改成"动态取学生执色"后):
        #     ① 新标签「本代学生(黑)胜率: 44.7% (45/101) | 历史 ...」← 数字【已经是学生胜率】→ 不能再换算 ✗
        #     ② 旧标签「本代白方胜率: X%」← 白方口径, 教练执白时要 100−X
        # ★ 按轮次过滤日志 (2026-09-27): 同一日志【跨轮追加】(train_v39.log 里还留着上一轮
        #   "本代学生(黑)胜率 82%" 那批行) → 不过滤会把上一轮的成绩当本轮 ✗
        #   本轮起点 = 最后一次「训练启动 / 已从检查点恢复」行的时间戳
        #   ⚠️ 两个坑:
        #     ① 「N个工兵(vNN)已重启」不是主进程启动(主进程会周期性重启落后工兵) → 不能当初点 ✗
        #        (实测: 09:47 的工兵重启把本轮窗口截断成 1 小时 → KPI 读不到本轮历史 ✗)
        #     ② 「已从检查点恢复(上次训练到 iteration N)」若 N >= 本轮起始代 = 轮中重启 → 忽略,
        #        只认 N < 本轮起始代 的那次(= 真正从本轮起点起跑)
        try:
            _rstart_gen = int((coach or {}).get('worker_min'))
        except Exception:
            _rstart_gen = None
        _boot_ts = None
        for line in log_lines:
            _mts0 = re.match(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
            if not _mts0:
                continue
            if '训练启动' in line:
                _boot_ts = _mts0.group(1)
                continue
            _mr = re.search(r'已从检查点恢复.*?iteration\s*(\d+)', line)
            if _mr:
                _n = int(_mr.group(1))
                if _rstart_gen is None or _n < _rstart_gen:
                    _boot_ts = _mts0.group(1)
        _lines_round = log_lines
        # ★ 窗口口径 (2026-09-27 二次修): 不用时间戳过滤 ✗
        #   实测坑: 主进程 09:47 又打了一次"训练启动"(轮中重启) → 按时间戳切会把本轮历史截成 1 小时 ✗
        #   改法: 本轮"每代胜率行"的条数 = 已收齐代数 = (当前在产代 − 本轮起始代 + 1)
        #         从尾部只取这么多行(+2 容错 0/0 退化行); 本轮一代都没收齐 → 不取任何旧行
        try:
            _prod = int(cur_it) - 1
        except Exception:
            _prod = None
        _n_round = None
        if _prod is not None and _rstart_gen is not None:
            _n_round = _prod - _rstart_gen + 1
        if _n_round is not None:
            _rate_idx = [i for i, l in enumerate(log_lines)
                         if re.search(r'本代(?:学生\([黑白]\)|白方|黑方)胜率[:：]', l)]
            # ★ 时间下界 (2026-09-27 实踩): 服务器切轮时若【没归档上一轮的 worker_data】,
            #   worker_min 仍是旧代 → _n_round 虚高 → 会把上一轮的胜率行当本轮显示 ✗
            #   (实况: 第五轮刚开局, KPI 显示了第三轮尾巴的 80.4%) → 用最后一次"训练启动"兜住
            if _boot_ts:
                _keep = []
                for _i in _rate_idx:
                    _m0 = re.match(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', log_lines[_i])
                    if _m0 and _m0.group(1) >= _boot_ts:
                        _keep.append(_i)
                _rate_idx = _keep
            if _n_round <= 0 or not _rate_idx:
                _lines_round = []          # 本轮还没收齐任何一代(或没有本轮的行) → 一条旧行都不吃 ✗
            elif len(_rate_idx) > _n_round + 2:
                _lines_round = log_lines[_rate_idx[-(_n_round + 2)]:]
            else:
                _lines_round = log_lines[_rate_idx[0]:]
        kpi['round_boot_ts'] = _boot_ts
        kpi['round_start_gen'] = _rstart_gen
        kpi['round_gens_expected'] = _n_round
        kpi['round_log_lines'] = len(_lines_round)
        _log_rate, _log_hist = None, []
        _rate_is_student, _hist_is_student = False, False
        for line in reversed(_lines_round):
            m = re.search(r'本代学生\(([黑白])\)胜率[:：]\s*([\d.]+)%\s*\((\d+)/(\d+)\)', line)
            if m and int(m.group(4)) > 0:      # ⚠️ n=0 是"空收集"退化行(0.0% (0/0)) → 不能当读数 ✗
                _log_rate = {'rate': float(m.group(2)), 'wins': int(m.group(3)), 'n': int(m.group(4)),
                             'side': m.group(1)}
                _rate_is_student = True
                break
            m = re.search(r'本代白方胜率[:：]\s*([\d.]+)%\s*\((\d+)/(\d+)\)', line)
            if m and int(m.group(3)) > 0:
                _log_rate = {'rate': float(m.group(1)), 'wins': int(m.group(2)), 'n': int(m.group(3)),
                             'side': '白'}
                break
        if _log_rate is None and coach.get('color') is None:   # 兼容旧写法:「白方胜率 24.0% (24/100)」
            for line in reversed(_lines_round):
                m = re.search(r'白方胜率[:：]\s*([\d.]+)%\s*\((\d+)/(\d+)\)', line)
                if m:
                    _log_rate = {'rate': float(m.group(1)), 'wins': int(m.group(2)), 'n': int(m.group(3)),
                                 'side': '白'}
                    break
        for line in reversed(_lines_round):
            # ① 新标签: 「学生(黑)胜率历史(重启重建, 3代): 51% 45% 42%」(已是学生口径 ✓)
            m = re.search(r'学生\(([黑白])\)胜率历史(?:\(重启重建[^)]*\))?[:：]?\s*([\d%\. ]+%[\d%\. ]*)', line)
            if m:
                _log_hist = [float(x) for x in re.findall(r'(\d+(?:\.\d+)?)%', m.group(2))]
                _hist_is_student = True
                break
            # ② 旧标签: 「本代白方胜率: 24.0% (24/100) | 历史 27% 16% ...」(每代行的历史只有一个竖线)
            m = re.search(r'(?:白方胜率历史(?:\(重启重建[^)]*\))?|\|\s*历史)[:：]?\s*([\d%\. ]+%[\d%\. ]*)', line)
            if m:
                _log_hist = [float(x) for x in re.findall(r'(\d+(?:\.\d+)?)%', m.group(1))]
                break
        if _log_rate:
            kpi['raw_white'] = _log_rate['rate'] if _log_rate.get('side') == '白' else None
            kpi['rate_side'] = _log_rate.get('side')
            if not _rate_is_student:
                _log_rate['raw_white'] = _log_rate['rate']
                _log_rate['rate'] = _stu(_log_rate['rate'])  # ★ 就地换算, 否则后面合并又用回原始白胜率 ✗
        kpi['log_rate'] = _log_rate['rate'] if _log_rate else None
        kpi['log_rate_is_student'] = _rate_is_student
        if not _hist_is_student:
            kpi['raw_white_hist'] = list(_log_hist)      # 旧标签才需要留档原始白胜率
            _log_hist = [_stu(v) for v in _log_hist]     # 换算成"学生方胜率"再合并
        # (c) 合并: 【本地数据为权威】(有局数/中位手数, 逐代真实统计), 日志只补"本地还没有的最新几代"
        #   ⚠️ 2026-09-25 实踩: 曾按"历史末项=上一代, 逐个往前对"做位置映射 → 一旦历史里漏了某代
        #      (gen37 空转 0 局 → 日志历史里没有它), 后面全体错位, 还把本地正确值覆盖成错的值 ✗
        #   正确做法: 只对"本地缺的那几代"取日志历史的**尾部**对应长度, 且不覆盖本地已有的值
        _merged = {it: dict(g, iter=it) for it, g in _local.items()}
        _last_it = None
        if cur_it:
            try:
                _last_it = int(cur_it) - 1
            except Exception:
                _last_it = None
        if _last_it is not None and _log_hist:
            _mx_local = max(_local) if _local else None
            if _mx_local is not None:
                # ★ 有本地数据时: 只可能补【最新那一代】(它的数字来自当代行, 无歧义)
                #   ⚠️ 不要拿历史序列去推更早的缺代 —— v39 的种子文件是 v38 的 iter58, 文件号比代号大,
                #      历史是按【文件号】排序的 → 尾部错位, 会凭空造出"幻影代"且把种子值安到未完成的代上 ✗
                _miss = [_last_it] if _last_it > _mx_local else []
                _vals = _log_hist[-1:]
            else:
                _start = max(1, _last_it - len(_log_hist) + 1)
                _miss = [it for it in range(_start, _last_it + 1)]
                _vals = _log_hist[-len(_miss):] if _miss else []
            _off = len(_miss) - len(_vals)        # 历史不够长时, 只补能对上的尾部
            for _i, _it in enumerate(_miss[_off:] if _off > 0 else _miss):
                _v = _vals[_i] if _i < len(_vals) else None
                if _v is None:
                    continue
                _merged.setdefault(_it, {'iter': _it, 'n': None, 'steps_med': None})
                if _merged[_it].get('src') != 'local':
                    _merged[_it]['rate'] = _v
                    _merged[_it]['src'] = 'log'
        if _log_rate and _last_it is not None and _last_it not in _local:
            _merged.setdefault(_last_it, {'iter': _last_it, 'steps_med': None})
            _merged[_last_it].update({'rate': _log_rate['rate'], 'n': _log_rate['n'],
                                      'wins': _log_rate['wins'], 'src': 'log'})
        _gens = [_merged[k] for k in sorted(_merged) if _merged[k].get('rate') is not None][-8:]
        kpi['gens_all'] = _gens          # 全量(含在产代, 仅调试用)
        kpi['source'] = 'log' if any(g.get('src') == 'log' for g in _gens) else 'local'
        # 本地缺哪些代 (replay 页看不到 → 提示同步); 只报最近 6 代内的
        if _local:
            _mx = max(_local)
            kpi['lag_gens'] = [g['iter'] for g in _gens if g.get('src') == 'log' and g['iter'] > _mx]
        if _gens:
            # ★ 主指标只用【完整代】: 在产代的胜率是半局噪声 (实测: 13 局 0 胜 → 卡片显示 0% ✗)
            #   日志来源的代 = 主进程收集时的完整一代 (n 可能没写) → 视为完整;
            #   本地来源的代按 n>=90 判完整 (目标 100 局)
            def _complete(g):
                #    ★ 本轮一代 120 局, 写死 90 会把"还在产的当代"(如 91局) 误判成完整代 ✗
                #      → 用"当前在产代" prod_it 精确排除 (日志已出胜率行的必为已收齐) ✓
                if isinstance(prod_it, int) and (g.get('iter') or 0) >= prod_it:
                    return g.get('src') == 'log'
                return g.get('src') == 'log' or (g.get('n') or 0) >= 90
            kpi['pending'] = next((g for g in reversed(_gens) if not _complete(g)), None)
            _done = [g for g in _gens if _complete(g)]
            kpi['partial_gen'] = False
            if not _done:
                # 本轮还没有任何"收齐"的代 (刚开轮 / 刚清理过脏批次) → 退用局数最多且 >=40 的那代,
                # 并打上"未收齐"标记 —— 总比卡片显示空更好, 但口径必须诚实标注 ✓
                _big = [g for g in _gens if (g.get('n') or 0) >= 40]
                if _big:
                    _done = [max(_big, key=lambda g: g['iter'])]
                    kpi['partial_gen'] = True
            if _done:
                kpi['latest'] = max(_done, key=lambda g: g['iter'])
                # ★ 曲线/趋势只吃【完整代】(2026-09-27): 在产代(如刚开代 2 局凑巧全胜)进曲线
                #   会画出 100% 的假尖峰 ✗ → 在产代单独走 pending
                kpi['gens'] = _done
                kpi['gap_pp'] = round(COACH_GOAL_RATE * 100 - kpi['latest']['rate'], 1)
                _tr = [g['rate'] for g in _done][-5:]
                kpi['trend'] = _tr
                if len(_tr) >= 3:
                    _d = [(_tr[k] - _tr[k - 1]) for k in range(1, len(_tr))]
                    _avg = sum(_d) / len(_d)
                    kpi['slope_pp'] = round(_avg, 2)
                    if _avg > 0.2 and kpi['gap_pp'] > 0:
                        kpi['eta_gens'] = int(kpi['gap_pp'] / _avg + 0.999)
        kpi.setdefault('gens', [])       # 无完整代时也要是空列表(前端直接遍历) ✓
        # 本地索引里最新的代 (replay 页最新可看代)
        kpi['local_max'] = max(_local) if _local else None
    except Exception as e:
        print(f'[kpi] {e}')
    # ★ 2026-11-15: 兜底 KPI —— v24/v24b 没有教练、也没有本地同步的自对弈索引，
    #   但工兵日志能逐代聚合出白胜率（bw_per_gen）→ 直接喂给主指标卡，
    #   否则卡片永远写"本地还没有该版本的自对弈数据"。
    if not kpi.get('gens') and data.get('bw_per_gen', {}).get('iterations'):
        _bp = data['bw_per_gen']
        _stu_white = (kpi.get('student_name') == '白')
        kpi['gens'] = [{
            'iter': it,
            'rate': (_bp['white_rate'][i] if _stu_white
                     else (None if _bp['white_rate'][i] is None
                           else round(100.0 - _bp['white_rate'][i], 1))),
            'n': _bp['total'][i],
        } for i, it in enumerate(_bp['iterations'])]
        _valid = [g for g in kpi['gens'] if g['rate'] is not None]
        # 前端 kpi.latest 要的是对象 {iter, rate, n}（不是标量）
        kpi['latest'] = _valid[-1] if _valid else None
        kpi['source'] = 'worker_log'
    data['kpi'] = kpi

    data['train_version'] = TRAIN_VERSION
    data['compare_version'] = COMPARE_VERSION

    # 🚨 门控代标记 (必须放在 gate 解析之后): 门控赛期间主进程阻塞 → 那一代的"耗时"含门控时间
    #   (实测 v37 门控 at 代10 → 代11 显示 120 min, 其中 82 min 是门控) → 标出来,
    #   并把「近3代均」只按非门控代算, 否则波动/提速判断被污染
    try:
        _gated = set()
        for h in (gate.get('history') or []):
            if h.get('iteration') is not None:
                _gated.add(int(h['iteration']) + 1)          # 门控在代 N 结束时跑 → 拖慢代 N+1
        if _gated and data.get('iter_durations'):
            for x in data['iter_durations']:
                x['gated'] = x['iter'] in _gated
            _clean = [x['min'] for x in data['iter_durations'] if not x.get('gated')][-3:]
            if _clean:
                data['iter_dur_avg3'] = round(sum(_clean) / len(_clean), 1)
                data['iter_dur_avg3_n'] = len(_clean)
    except Exception as e:
        print(f'[iter-dur-gate] {e}')

    # 主进程阶段修正: 门控赛进行中时主进程在等门控 (日志被 【门控】 行占据) —— 覆盖成门控态
    if gate.get('running'):
        _gi = gate.get('iter_gate') or cur_it
        _prog = f"{gate['progress']}/{gate.get('games', 100)} 局" if gate.get('progress') else f"{gate.get('games', 100)} 局配对"
        data['main_status'] = f"⏳ 门控赛进行中 代{_gi} ({_prog})"
        data['main_phase'] = 'gating'

    # ★ 近期产局速率（2026-09-26 用户报"时间统计不太对"）：
    #   从主日志「等待工兵: N/100局已到」序列算最近一段的真实吞吐（秒/局）
    #   比"已持续 ÷ 已到局数"的平均速率更贴当前状态（工兵快慢会变、重启后从已有局数续算）
    try:
        from datetime import datetime as _dt2
        _ps = []
        for line in log_lines:
            m = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*?等待工兵: (\d+)/(\d+)局已到', line)
            if m:
                _ps.append((m.group(1), int(m.group(2)), int(m.group(3))))
        if len(_ps) >= 5:
            _ps = _ps[-400:]           # 只看最近约 1 小时(日志 10s 一条)
            data['iter_target'] = _ps[-1][2]
            _cut = 0                       # 若本代重置(已到局数回落), 只取重置之后那段
            for _i in range(len(_ps) - 1, 0, -1):
                if _ps[_i][1] < _ps[_i - 1][1]:
                    _cut = _i
                    break
            _seg = _ps[_cut:]
            _sec = (_dt2.strptime(_seg[-1][0], '%Y-%m-%d %H:%M:%S')
                    - _dt2.strptime(_seg[0][0], '%Y-%m-%d %H:%M:%S')).total_seconds()
            _dg = _seg[-1][1] - _seg[0][1]
            if _sec >= 120 and _dg > 0:
                data['iter_rate_recent'] = round(_sec / _dg, 1)      # 秒/局(近期)
                data['iter_recent_span_min'] = round(_sec / 60, 1)
                data['iter_recent_games'] = _dg
    except Exception as e:
        print(f'[train-rate-recent] {e}')

    # 预计结束时间：剩余局数 × 速率 + 当前时间（基准必须是"现在"而非"开始时间"，
    # 否则已耗时段被漏掉，ETA 偏早——2026-08-01 用户指出修正）
    # 🚨 2026-09-13 攒批推理优化版口径：主进程不再产局，本代进度就是「主进程收到的工兵局数」
    #   (等待工兵 N/100)，剩余 = 目标 - 已到。旧版按「最慢工兵还差多少」算会把 ETA 推到下一倍
    #   （20 工兵均摊 5 局，单个工兵分母仍是 10，min(game) 天然偏小 → ETA 虚高）
    # 🚨 2026-09-26 速率优先用「近期」(日志序列) —— 平均速率含重启前的空窗会偏大
    try:
        _rate = data.get('iter_rate_recent') or data.get('iter_rate')
        _now = data.get('iter_now')
        data['iter_rate_used'] = ('recent' if data.get('iter_rate_recent') else
                                  'avg' if data.get('iter_rate') else None)
        # ★ 样本不足就不给速率/ETA (2026-09-27): 刚换代时"近期速率"可能只有 2~3 局样本
        #   (实测: 本代 3 局 → 近期 86.7 s/局 → ETA 偏 2.5 小时 ✗)
        #   口径: 本代已到 >= 12 局, 或近期窗口样本 >= 12 局, 才出预测
        if (cur_game or 0) < 12 and (data.get('iter_recent_games') or 0) < 12:
            data['iter_rate_used'] = 'insufficient'
            data['iter_rate_note'] = '样本不足(本代 %d 局, 满 12 局才给预测)' % (cur_game or 0)
            data['iter_rate'] = None
            data['iter_rate_recent'] = None
            data['iter_eta'] = None
            data['iter_remain_min'] = None
            _rate = None
        if _rate and _now:
            _denom = data.get('iter_target') or cur_denom or 100
            remain_games = max(0, _denom - (cur_game or 0))
            remain_s = remain_games * float(_rate)
            from datetime import datetime as _dt, timedelta as _td
            eta_dt = _dt.strptime(str(_now), '%Y-%m-%d %H:%M:%S') + _td(seconds=remain_s)
            data['iter_eta'] = eta_dt.strftime('%Y-%m-%d %H:%M:%S')
            data['iter_remain_min'] = round(remain_s / 60, 1)
    except Exception as e:
        print(f'[train-eta] {e}')

    # loss 历史: 优先从 training_v12.log 解析"学习完成"行 (全量拉取后完整), metadata losses 补充
    losses = {}
    try:
        meta = json.loads('\n'.join(segs.get('META', [])))
        losses = meta.get('config', {}).get('losses', {})
    except:
        pass
    log_loss = {}
    for line in log_lines:
        m = re.search(r'训练\s+(\d+):\s+学习完成\s+policy_loss=([\d.]+)\s+value_loss=([\d.]+)\s+total=([\d.]+)', line)
        if m:
            it = int(m.group(1))
            log_loss[it] = (float(m.group(2)), float(m.group(3)), float(m.group(4)))
    # 合并: log 优先, metadata 补充缺失代
    for k, v in losses.items():
        try:
            it = int(k)
        except:
            continue
        if it not in log_loss:
            log_loss[it] = (v.get('policy_loss', 0), v.get('value_loss', 0), v.get('total_loss', 0))
    # 本地累积备份合并 (防服务器 log 被覆盖/重启丢失 — 2026-08-11 误用 > 覆盖教训)
    # v29 (2026-09-03): c_init 1.25→3.5 单变量(v28全同基线) → 独立编号(it0起)
    # → loss 曲线 = v29 自己的 (备份 v29 目录 + log 合并)
    LOSS_BACKUP = os.path.join(PARENT_DIR, CKPT_BASENAME, 'loss_history.json')
    backup = {}
    if os.path.exists(LOSS_BACKUP):
        try:
            b = json.load(open(LOSS_BACKUP))
            for it, pl, vl, tl in zip(b.get('iterations', []), b.get('policy', []),
                                     b.get('value', []), b.get('total', [])):
                backup[int(it)] = (float(pl), float(vl), float(tl))
        except Exception:
            pass
    for it in log_loss:
        backup[it] = log_loss[it]
    its, pls, vls, tls = [], [], [], []
    for it in sorted(backup.keys()):
        pl, vl, tl = backup[it]
        its.append(it); pls.append(round(pl, 4)); vls.append(round(vl, 4)); tls.append(round(tl, 4))
    data['loss'] = {'iterations': its, 'policy': pls, 'value': vls, 'total': tls}
    data['loss_source'] = 'log' if log_loss else ('metadata' if losses else 'none')
    data['loss_backup'] = len(backup)
    try:
        # 🚨 2026-11-15: 目录不存在会静默写失败（新仓库/新槽首次运行必踩）
        os.makedirs(os.path.dirname(LOSS_BACKUP), exist_ok=True)
        json.dump({'iterations': its, 'policy': pls, 'value': vls, 'total': tls},
                  open(LOSS_BACKUP, 'w'))
    except Exception as e:
        print(f'[loss-backup] 写入失败: {e}')
    # 对比线: 主线版本的对照版 = COMPARE_VERSION
    #   v37 (2026-09-25): 对照 v36 —— v37 = 从零训练 + 开局完全自由(RANDOM_OPEN_MOVES 3→0) + 权重门控;
    #   脚本保留 GOMOKU_RANDOM_OPEN_MOVES=3 可复现 v36 行为 → 同 it 轴同期对照最有意义
    #   (规则: 切主线 ≠ 重置对比线; 用户另行指定时改这里的常量与 dashboard label)
    try:
        def _load_loss(dirn):
            p = os.path.join(PARENT_DIR, f'gomoku_checkpoints_{dirn}', 'loss_history.json')
            if not os.path.exists(p):
                return {}
            b = json.load(open(p))
            return {int(it): (float(pl), float(vl), float(tl))
                    for it, pl, vl, tl in zip(b.get('iterations', []), b.get('policy', []),
                                              b.get('value', []), b.get('total', []))}
        cmp = _load_loss(COMPARE_VERSION)   # v37 → v36 (同窗对照; 无数据则前端不画对比线)
        if cmp:
            cits, cpls, cvls, ctls = [], [], [], []
            for it in sorted(cmp):
                pl, vl, tl = cmp[it]
                cits.append(it); cpls.append(pl); cvls.append(vl); ctls.append(tl)
            data['loss_compare'] = {'iterations': cits, 'policy': cpls, 'value': cvls, 'total': ctls}
    except Exception:
        pass

    # 黑白胜率趋势 (2026-08-29 用户): 解析 train log "代X 局Y/10: 黑胜/白胜" → 每代黑白胜局数
    # ⚠️ 2026-09-13 攒批推理优化版起主进程不再产局 → 主日志已无对局行，本段对 v36 主线恒为空。
    #    当前代的黑白拆解改从工兵日志聚合（见上方 data['bw_current']）；此段保留兼容旧版本日志。
    try:
        bw_iters, bw_black, bw_white, bw_total = [], [], [], []
        bw_stat = {}
        for line in log_lines:
            m = re.search(r'代(\d+)\s+局\d+/\d+:\s*(黑|白)胜', line)
            if m:
                it = int(m.group(1)); win = m.group(2)
                st = bw_stat.setdefault(it, [0, 0])
                if win == '黑':
                    st[0] += 1
                else:
                    st[1] += 1
        for it in sorted(bw_stat):
            b, w = bw_stat[it]
            bw_iters.append(it)
            bw_black.append(b)
            bw_white.append(w)
            bw_total.append(b + w)
        data['black_white_rate'] = {'iterations': bw_iters, 'black': bw_black,
                                    'white': bw_white, 'total': bw_total}
    except Exception as e:
        print(f'[bw-rate] {e}')

    # 检查点（服务器 v2 文件名无 _v2 后缀，本地解析排序避免 sort -n 对非纯数字失效）
    cks = []
    for line in segs.get('CK', []):
        m = re.search(r'weights_it(\d+)', line)
        if m:
            cks.append(int(m.group(1)))
    data['checkpoints'] = sorted(cks)

    # 经验存档（v2 worker_data 文件 iter{N}_{wid}_game_{M}.pkl）
    exps = []
    for line in segs.get('EXP', []):
        m = re.search(r'iter(\d+)_', line)
        if m:
            it = int(m.group(1))
            if it not in exps:
                exps.append(it)
    data['experiences'] = sorted(exps)

    # P1 2026-09-10: SSH 通道慢/限流时 ssh_cmd 返回空 → 旧逻辑无条件 clear 会让驾驶舱整页空白且无日志
    # 现在: 仅在拿到有效数据时替换缓存; 失败则保留上次成功数据 + 标记 ssh_ok=False 供前端提示
    got_valid = bool(data.get('current_it') is not None or data.get('log')
                     or (data.get('loss') or {}).get('iterations'))
    with TRAIN_LOCK:
        if got_valid:
            TRAIN_CACHE.clear()
            TRAIN_CACHE.update(data)
            TRAIN_CACHE['ssh_ok'] = True
            TRAIN_CACHE['stale'] = False
            TRAIN_CACHE['last_ok'] = data.get('time')
        else:
            if TRAIN_CACHE:
                TRAIN_CACHE['ssh_ok'] = False
                TRAIN_CACHE['stale'] = True
            else:
                TRAIN_CACHE.update({'ssh_ok': False, 'stale': True, 'time': data.get('time'),
                                    'main_status': 'SSH 连接失败(通道慢/限流)',
                                    'workers': [], 'log': [], 'checkpoints': [], 'experiences': [],
                                    'loss': {}, 'black_white_rate': {}})
            print('[train] SSH 拉取无效(通道慢/限流?), 保留上次数据 last_ok=' + str(TRAIN_CACHE.get('last_ok')))


def background_poll():
    while True:
        try:
            fetch_training_data()
        except Exception as e:
            print(f'[train-pool] {e}')
        time.sleep(8)


# ── 权重自动同步（2026-08-02 新增：本地 web 权重趋势需要最新服务器权重） ──
SYNC_LOCK = threading.Lock()
SYNC_STATE = {'last': None, 'status': 'idle', 'msg': '', 'pulled': 0}


def _server_weight_list():
    """SSH 列出服务器 v2 权重文件名（weights_itN.pt）"""
    out = ssh_cmd(f"ls {SSH_CHECKPOINT_DIR}/weights_it*.pt 2>/dev/null")
    names = []
    for line in out.split('\n'):
        line = line.strip()
        if 'weights_it' in line:
            names.append(line.split('/')[-1])
    return names


def sync_weights_once():
    """把服务器缺失的最新权重同步到本地 gomoku_checkpoints_v29/（服务器权重已带 _v29 后缀，同名拷贝）。
    v29 训练用服务器独立目录 gomoku_checkpoints_v29（权重自带 _v29 后缀），本地同名。
    返回 (pulled_count, msg)。若已有同步在进行，返回 (None, '同步进行中')"""
    if not SYNC_LOCK.acquire(blocking=False):
        return None, '同步进行中'
    try:
        names = _server_weight_list()
        if not names:
            SYNC_STATE.update({'last': datetime.now().strftime('%H:%M:%S'), 'status': 'error',
                               'msg': '服务器连接失败或无权重(SSH 超时/限流)', 'pulled': 0})
            return 0, '服务器连接失败或无权重(SSH 超时/限流)'
        local_dir = os.path.join(PARENT_DIR, CKPT_BASENAME)
        os.makedirs(local_dir, exist_ok=True)
        local_names = set(os.listdir(local_dir))
        pulled = 0
        failed = 0
        # ★ 2026-11-15: 权重文件后缀 ≠ 槽号。v24b 槽训练脚本里 VERSION 仍是 "v24"
        #   → 文件叫 weights_it0_v24.pt；旧写法按 _{TRAIN_VERSION}.pt 过滤会一个都匹配不上，
        #   于是 /api/weights 永远返回空 → 网络前向/MCTS/摆谱/出题 四个页面都没有权重可选。
        _wsuf = ACTIVE_SLOT.get('weight_suffix') or TRAIN_VERSION
        for n in sorted(names):
            if not n.endswith(f'_{_wsuf}.pt') and n != 'weights_best.pt':
                continue                     # 只拉主线版本的权重 (+ 门控 best)
            local_n = n  # 服务器权重已带版本后缀，直接同名
            if local_n in local_names:
                # ⚠ 2026-09-10: 旧逻辑只按文件名判"已存在" → 截断文件被永久跳过(还显示"已最新")
                # 现在: 已存在的也校验完整性, 截断/损坏的删掉重拉
                _lp = os.path.join(local_dir, local_n)
                try:
                    import zipfile as _zf
                    _ok = os.path.getsize(_lp) >= 6_000_000 and _zf.is_zipfile(_lp)
                except Exception:
                    _ok = False
                if _ok:
                    continue
                try:
                    os.remove(_lp)
                    print(f'[sync] {local_n} 本地已有但截断/损坏, 删除重拉')
                except Exception:
                    pass
            # 用 scp 拉取（expect 会话支持 scp 同密码）
            remote = f"{SSH_USER}@{SSH_HOST}:{SSH_CHECKPOINT_DIR}/{n}"
            # ★ 2026-11-15: 三个致命问题一起修
            #   ① set timeout 300 / subprocess 330s —— 这条链路实测只有 ~20KB/s，
            #      10MB 权重需要 ~500s ⇒ 每次都在 330s 被掐断，留下截断文件后再被删掉，
            #      权重永远同步不下来（/api/weights 恒空 → 网络前向/MCTS/摆谱/出题 全没得选）
            #   ② 直接写目标文件名 → 下载过程中页面会读到半截文件
            #   ③ 失败重试无退避
            _part = os.path.join(local_dir, local_n + '.part')
            script = (
                'set timeout 1200\n'
                'spawn scp -o StrictHostKeyChecking=no -o ServerAliveInterval=15 -o ServerAliveCountMax=40 '
                '-o Compression=no -P ' + str(SSH_PORT) + ' '
                + remote + ' ' + _part + '\n'
                'expect "password:"\n'
                'send "' + SSH_PASS + '\\r"\n'
                'expect eof\n'
            )
            try:
                r = subprocess.run(['expect', '-c', script], capture_output=True, text=True, timeout=1250)
                lp = _part
                if os.path.exists(lp):
                    # 🛡 完整性校验 (2026-08-27): scp 中断留半截文件 (it253 5.4MB vs 正常 10MB)
                    # 或字节数凑巧对但 zip 结构坏 (it246/247 10MB 但 torch 读崩) → 双检查
                    ok_size = os.path.getsize(lp) >= 6_000_000
                    ok_zip = False
                    try:
                        import zipfile
                        ok_zip = zipfile.is_zipfile(lp)
                    except Exception:
                        pass
                    if ok_size and ok_zip:
                        os.replace(lp, os.path.join(local_dir, local_n))   # 校验通过才转正
                        pulled += 1
                    else:
                        failed += 1
                        try:
                            _sz = os.path.getsize(lp)
                            os.remove(lp)
                            print(f'[sync] {local_n} 截断/损坏({_sz} 字节, 大小或zip校验失败)已删除, 下次重拉')
                        except Exception:
                            pass
                else:
                    failed += 1
                    print(f'[sync] {local_n} scp 未产出文件(超时/连接失败), 稍后重试')
            except Exception as _e:
                failed += 1
                print(f'[sync] {local_n} scp 异常: {_e}')
        if pulled and not failed:
            msg = f'同步完成，新增 {pulled} 个权重'
        elif failed and pulled:
            msg = f'新增 {pulled} 个，{failed} 个拉取失败(通道慢/超时)，可稍后重试'
        elif failed:
            msg = f'拉取失败 {failed} 个(通道慢/超时截断)，未新增 — 可稍后重试'
        else:
            msg = '无新权重（已最新）'
        SYNC_STATE.update({'last': datetime.now().strftime('%H:%M:%S'), 'status': 'ok', 'msg': msg, 'pulled': pulled})
        return pulled, msg
    finally:
        SYNC_LOCK.release()


def background_sync():
    """每 20 分钟自动同步一次权重"""
    while True:
        try:
            sync_weights_once()
        except Exception as e:
            print(f'[sync] {e}')
        time.sleep(1200)  # 20 分钟


@app.route('/api/sync-weights', methods=['POST'])
@require_auth
def api_sync_weights():
    """手动触发立即同步"""
    try:
        pulled, msg = sync_weights_once()
        if pulled is None:
            return jsonify({'ok': False, 'busy': True, 'msg': msg, 'state': SYNC_STATE}), 409
        return jsonify({'ok': True, 'pulled': pulled, 'msg': msg, 'state': SYNC_STATE})
    except Exception as e:
        return jsonify({'ok': False, 'msg': str(e)}), 500


@app.route('/api/sync-state')
@require_auth
def api_sync_state():
    # 不拿锁（SYNC_STATE 是简单 dict，读快照即可，避免与后台同步互相阻塞）
    return jsonify(dict(SYNC_STATE))


@app.route('/api/training')
@require_auth
def api_training():
    # 手动刷新模式: 每次请求实时 SSH 拉取 (fetch_training_data 无返回值, 内部持锁更新 TRAIN_CACHE)
    try:
        fetch_training_data()
    except Exception as e:
        print(f'[api/training] 拉取失败: {e}')
    return jsonify(dict(TRAIN_CACHE))


@app.route('/dashboard')
@require_auth
def dashboard_page():
    return send_from_directory(os.path.join(SCRIPT_DIR, 'static'), 'dashboard.html')


@app.route('/network')
@require_auth
def network_page():
    return send_from_directory(os.path.join(SCRIPT_DIR, 'static'), 'network.html')


@app.route('/mcts')
@require_auth
def mcts_page():
    return send_from_directory(os.path.join(SCRIPT_DIR, 'static'), 'mcts.html')


@app.route('/quiz')
@require_auth
def quiz_page():
    """出题评估: 用户棋理标注局面 × 任意权重的 policy 准确率"""
    return send_from_directory(os.path.join(SCRIPT_DIR, 'static'), 'quiz.html')


# ═══════════════════════════════════════════════
# 权重开局分布演化
# ═══════════════════════════════════════════════

@app.route('/evolution')
@require_auth
def evolution_page():
    return send_from_directory(os.path.join(SCRIPT_DIR, 'static'), 'evolution.html')


@app.route('/replay')
@require_auth
def replay_page():
    return send_from_directory(os.path.join(SCRIPT_DIR, 'static'), 'replay.html')


# 自对弈数据观察 (replay_pool)
from replay_api import replay_bp
app.register_blueprint(replay_bp)


@app.route('/api/opening_dist', methods=['POST'])
@require_auth
def api_opening_dist():
    """权重开局分布: 对每个权重跑空盘前向, 返回黑第一手 policy 分布(225) + 熵 + top5。
    权重按自然序(it0,1,2,...49)排序(非字典序!), 结果缓存到 opening_dist_cache.json"""
    try:
        import torch, glob, re, os
        import numpy as np
        data = request.json or {}
        version = data.get('version', 'v8')
        ver = next((v for v in VERSIONS if v[0] == version), None)
        if not ver:
            return jsonify({'ok': False, 'error': f'未知版本 {version}'})
        _, wdir, mod_path, _ = ver
        wdir_full = os.path.join(PARENT_DIR, wdir)
        files = glob.glob(os.path.join(wdir_full, f'weights_it*_{version}.pt'))
        its = []
        for f in files:
            # ^ 锚定防双重后缀污染 (weights_it0_v12_v12.pt / weights_it0_v14_v12.pt 不匹配)
            m = re.search(r'^weights_it(\d+)_' + version + r'\.pt$', os.path.basename(f))
            if m:
                its.append((int(m.group(1)), f))
        its.sort()  # 自然序 (数值), 避开字典序 it10<it2 的坑
        if not its:
            return jsonify({'ok': False, 'error': f'{wdir} 无 {version} 权重'})
        cache_file = os.path.join(SCRIPT_DIR, 'opening_dist_cache.json')
        cache = {}
        if os.path.exists(cache_file):
            try:
                cache = json.load(open(cache_file))
            except Exception:
                pass
        cache_key = version + ':' + ','.join(str(i) for i, _ in its)
        if cache.get('key') == cache_key and cache.get('ok_its') and \
           len(cache.get('items', [])) == len(cache.get('ok_its', [])):
            return jsonify({'ok': True, 'version': version, 'cached': True, 'items': cache['items']})
        mod = load_mod_for(version)
        meta = load_metadata()
        net_kwargs = meta.get('config', {}).get('net_kwargs',
            {'conv_filters': [128], 'residual_filters': [[128, 128]] * 8, 'policy_filters': [128]})
        # v16 是 9×9 试验版 (GN): 环境/棋盘/坐标全按 9×9; 其余版本 15×15
        is9 = (version == 'v16')
        if is9:
            import boardgame2
            env = boardgame2.KInARowEnv(board_shape=9, target_length=5)
            board = np.zeros((9, 9), dtype=np.float32)
        else:
            env = gym.make('Gomuku-v0')
            board = np.zeros((15, 15), dtype=np.float32)
        bsize = 9 if is9 else 15
        items = []
        skipped = []
        for it, f in its:
            try:
                t_agent = mod.AlphaZeroAgent(env=env, kwargs=net_kwargs, sim_count=1, batches=1, batch_size=8)
                sd = torch.load(f, map_location=mod.device)
                t_agent.net.load_state_dict(sd)
                t_agent.net.to(mod.device)
                net = t_agent.net
                net.eval()
                if hasattr(mod, 'make_input'):
                    x = torch.tensor(mod.make_input(board)).unsqueeze(0)
                else:
                    x = torch.tensor(board).unsqueeze(0).unsqueeze(0)
                with torch.no_grad():
                    p_log, v = net(x)
                p = torch.exp(p_log).cpu().numpy().reshape(-1)
                p = p / p.sum()
                ent = float(-np.sum(p * np.log(p + 1e-12)))
                top_idx = np.argsort(p)[::-1][:5]
                top = [{'r': int(j) // bsize, 'c': int(j) % bsize, 'pct': round(float(p[j]) * 100, 2)} for j in top_idx]
                vv = float(v.reshape(-1)[0])
                items.append({'it': it, 'entropy': round(ent, 3), 'top': top,
                              'value': round(vv, 3),
                              'winrate': round((vv + 1) / 2 * 100, 1),
                              'dist': [round(float(vv), 5) for vv in p]})
            except Exception as e:
                skipped.append({'it': it, 'error': str(e)[:60]})
                print(f'[opening-dist] 跳过损坏权重 it{it}: {e}')
        try:
            json.dump({'key': cache_key, 'ok_its': [x['it'] for x in items], 'items': items}, open(cache_file, 'w'))
        except Exception as e:
            print(f'[opening-dist] 缓存写入失败: {e}')
        return jsonify({'ok': True, 'version': version, 'cached': False,
                        'items': items, 'count': len(items), 'skipped': skipped})
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'ok': False, 'error': str(e)})


if __name__ == '__main__':
    print('🚀 五子棋 Web 对战启动: http://127.0.0.1:8765')
    print('📊 训练监控:        http://127.0.0.1:8765/dashboard')
    print('🧠 网络前向:        http://127.0.0.1:8765/network')
    try:
        # 手动刷新模式: 启动时不自动拉训练数据, 也不启动 background_poll (8秒SSH轮询 → 限流)
        pass
    except Exception as e:
        print(f'[init] 训练监控不可用: {e}')
    # 权重自动同步 (20分钟一次, 启动时立即同步一次) — 保留, 低频无碍
    # ⚠ 2026-09-10: debug=True 时 Werkzeug reloader 会 fork 父进程(监控者)+ 服务子进程,
    # 而 __main__ 块在两个进程都会执行 → 旧写法让两进程各起一个 background_sync 并发拉权重,
    # 抢带宽 + 双倍 SSH 连接触发 AutoDL 限流, 每次都截断失败。只在实际服务进程启动同步线程。
    if os.environ.get('WERKZEUG_RUN_MAIN') == 'true':
        try:
            threading.Thread(target=background_sync, daemon=True).start()
        except Exception as e:
            print(f'[init] 权重同步不可用: {e}')
    # 端口/主机可用环境变量覆盖（默认沿用历史 127.0.0.1:8765）
    app.run(host=os.environ.get('GOMOKU_WEB_HOST', '127.0.0.1'),
            port=int(os.environ.get('GOMOKU_WEB_PORT', '8765')),
            debug=os.environ.get('GOMOKU_WEB_DEBUG', '0') == '1')
