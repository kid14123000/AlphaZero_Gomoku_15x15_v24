"""
五子棋 AlphaZero 训练 v24 工兵脚本 (与 Gomoku-v0_pt_v24.py 配套)
与 Gomoku-v0_pt_v24.py 配套，使用相同 self_play（含 temperature 调度 + 开局约束）
v24 (2026-08-30): 从零训练 + ch2 恒定 1 (撤销执色编码, 灭空盘捷径), 无冻结
"""
import sys, os, time, json, glob, pickle
import logging
import importlib.util

logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

import numpy as np
import torch
import gym
from boardgame2 import BLACK, WHITE

# ── 路径 ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(SCRIPT_DIR, 'gomoku_checkpoints_v24')
METADATA_PATH = os.path.join(CHECKPOINT_DIR, 'metadata.json')
WEIGHTS_PATH = os.path.join(CHECKPOINT_DIR, 'weights.pt')
WORKER_DIR = os.path.join(CHECKPOINT_DIR, 'worker_data')

GAMES_PER_ITER = 10  # 每个工兵每代10局

torch.backends.cudnn.allow_tf32 = True
torch.set_num_threads(1)

# 从 v2 训练脚本导入
spec = importlib.util.spec_from_file_location('gomoku_v24b',
    os.path.join(SCRIPT_DIR, 'Gomoku-v0_pt_v24b.py'))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
mod.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
AlphaZeroNet = mod.AlphaZeroNet
AlphaZeroAgent = mod.AlphaZeroAgent
self_play_fn = mod.self_play

env = gym.make('Gomuku-v0')


def get_meta():
    if not os.path.exists(METADATA_PATH):
        return None
    with open(METADATA_PATH) as f:
        return json.load(f)


def generate_and_save(iteration, wid):
    """用当前 weights.pt 自对弈 GAMES_PER_ITER 局并保存"""
    # 加载最新权重
    try:
        state_dict = torch.load(WEIGHTS_PATH, map_location=mod.device)
    except Exception as e:
        logging.error('工兵{} 加载权重失败: {}'.format(wid, e))
        return False
    kwargs = {'conv_filters': [128], 'residual_filters': [[128, 128]] * 8,
              'policy_filters': [128]}
    agent = AlphaZeroAgent(env=env, kwargs=kwargs, sim_count=800,
                           batches=1, batch_size=8, disable_compile=True)
    if hasattr(agent, 'raw_net') and agent.raw_net is not None:
        agent.raw_net.load_state_dict(state_dict, strict=False)
        agent.raw_net.to(mod.device)
        agent.net = agent.raw_net
    else:
        agent.net.load_state_dict(state_dict, strict=False)
        agent.net.to(mod.device)

    # 攒批推理: 通知 server 热加载这批权重
    if getattr(agent, '_remote', None) is not None:
        try:
            agent._remote.update_weights(WEIGHTS_PATH)
        except Exception as e:
            logging.warning('工兵{} 通知 server 换权重失败: {}'.format(wid, e))

    # v2: 模拟量固定 400（与主脚本一致，全程不变）
    agent.sim_count = 800
    # 开局约束：不强制（v2 不启用）
    force_center = False

    for gid in range(GAMES_PER_ITER):
        path = os.path.join(WORKER_DIR, f'iter{iteration}_{wid}_game_{gid}.pkl')
        # 跳过已存在的局（重启衔接：代内续跑，不覆盖不重算）
        if os.path.exists(path):
            logging.info('工兵{} 代{} 局{} 已存在，跳过'.format(wid, iteration, gid))
            continue
        try:
            df = self_play_fn(env, agent, return_trajectory=True, verbose=False,
                              force_center=force_center)
        except Exception as e:
            logging.error('工兵{} 代{} 局{} 自对弈失败: {}'.format(wid, iteration, gid, e))
            time.sleep(5)  # 失败重试前稍等
            continue  # 不退出，跳过这一局继续
        with open(path, 'wb') as f:
            pickle.dump(df, f)
        w = df['winner'].iloc[0]
        ws = '黑胜' if w == 1 else '白胜' if w == -1 else '平局'
        logging.info('工兵{} 代{} 局{}/{}: {} {}条'.format(
            wid, iteration, gid + 1, GAMES_PER_ITER, ws, len(df)))
    return True


def main(wid):
    os.makedirs(WORKER_DIR, exist_ok=True)
    last_iteration = -1
    # 重启衔接: 扫描 worker_data 里自己(工兵wid)已产出的最大代。
    # ⚠️ 关键: last_iteration 初始化为 max-1（不是max!），否则若当前代只产出一部分
    # (如重启前产了19/90局) 工兵会整代跳过，主进程永远等不齐90局卡死。
    # generate_and_save 内部按"局"跳过已存在文件，补产缺失局，天然衔接。
    try:
        max_seen = -1
        for fn in glob.glob(os.path.join(WORKER_DIR, f'iter*_{wid}_game_*.pkl')):
            it = int(os.path.basename(fn).split('_')[0].replace('iter', ''))
            max_seen = max(max_seen, it)
        last_iteration = max_seen - 1  # 允许重新进入当前代，局级跳过已存在
        if max_seen >= 0:
            logging.info('工兵{} 重启衔接: 已产出到代{}, 重新进入该代补产缺失局'.format(wid, max_seen))
    except Exception as e:
        logging.warning('工兵{} 扫描历史产出失败(忽略): {}'.format(wid, e))
    while True:
        time.sleep(3)
        try:
            meta = get_meta()
        except Exception as e:
            logging.error('工兵{} 读metadata失败: {}'.format(wid, e))
            time.sleep(10)
            continue
        if meta is None:
            continue
        current_iter = meta.get('iteration', 0)
        if current_iter > last_iteration:
            # 确保权重已就绪
            if os.path.exists(WEIGHTS_PATH):
                ok = generate_and_save(current_iter, wid)
                # 无论成功与否都推进 last_iteration，避免死循环重试同一代
                last_iteration = current_iter
            else:
                logging.info('工兵{} 等待权重文件...'.format(wid))


if __name__ == '__main__':
    wid = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    main(wid)
