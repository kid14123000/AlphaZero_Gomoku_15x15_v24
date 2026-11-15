#!/usr/bin/env python3
"""v24b 验证（真·v24 代码 + 真·v24 权重 it73）
① 空盘 / 探针 value 是否合理（ch2 恒 1 的正确语义下）
② 均衡首手能否区分（|V| 有梯度）且落在中心区
③ 与 v24 默认（搜索 200 sims）首手对比
"""
import importlib.util
import os
import sys

import numpy as np
import torch

MOD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Gomoku-v0_pt_v24b.py")
CKPT = os.environ.get("V24_CKPT") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "gomoku_checkpoints_v24", "weights_it73_v24.pt")
spec = importlib.util.spec_from_file_location("gomoku_v24b", MOD)
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)

import gym                                            # noqa: E402
import boardgame2                                     # noqa: E402,F401

env = gym.make('Gomuku-v0')
agent = M.AlphaZeroAgent(env, sim_count=200, disable_compile=True)
sd = torch.load(CKPT, map_location=M.device)
agent.raw_net.load_state_dict(sd)
agent.net = agent.raw_net
agent.raw_net.eval()
print("权重已载入 it73_v24（126 键）")

obs, _ = env.reset()
board, player = obs

# ① 空盘 value
_, v = agent.predict_inference((player * board)[None], [None], None)
print("① 空盘 value = %+.4f  （黑先手，应明显为正）" % float(np.asarray(v).reshape(-1)[0]))

# ② 均衡首手（开关打开）
agent.balanced_first = 1
print("② 均衡首手（6 次抽样，中心 5×5 / 前 3 随机）：")
for _ in range(6):
    loc, prob = agent.decide(obs, return_prob=True, move_no=0, last_move=None, noise=False)
    cb = (player * board).copy()
    cb[loc] = 1
    _, vv = agent.predict_inference(cb[None], [loc], None)
    print("   %s  |value|=%+.4f" % (str(tuple(int(x) for x in loc)),
                                    abs(float(np.asarray(vv).reshape(-1)[0]))))

# ③ v24 默认（搜索）首手
agent.balanced_first = 0
loc, prob = agent.decide(obs, return_prob=True, move_no=0, last_move=None, noise=False)
cb = (player * board).copy()
cb[loc] = 1
_, vv = agent.predict_inference(cb[None], [loc], None)
print("③ v24 默认（200 sims）首手 = %s  |value|=%+.4f"
      % (str(tuple(int(x) for x in loc)), abs(float(np.asarray(vv).reshape(-1)[0]))))
