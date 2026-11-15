#!/usr/bin/env python3
"""v24b 快速验证驱动：从 it73 权重出发，跑 N 代自对弈+训练，只比较首手策略。
用法：BALANCED_FIRST=1 TARGET_ITERS=75 python run_quick.py
"""
import importlib.util
import os

MOD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Gomoku-v0_pt_v24b.py")
spec = importlib.util.spec_from_file_location("gomoku_v24b", MOD)
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)

cfg = {
    "train_iterations": int(os.environ.get("TARGET_ITERS", "75")),
    "train_episodes_per_iteration": int(os.environ.get("EPISODES", "10")),
    "batches": 32,
    "batch_size": 512,
    "sim_count": int(os.environ.get("SIMS", "800")),
    "net_kwargs": {
        "conv_filters": [128],
        "residual_filters": [[128, 128]] * 8,
        "policy_filters": [128],
    },
    "lr": 0.001,
    "worker_count": int(os.environ.get("WORKERS", "9")),
    "worker_games_per_worker": int(os.environ.get("GAMES_PER_WORKER", "10")),
    "freeze_policy_iters": 0,
    "freeze_value_iters": 0,
}
print("config:", {k: cfg[k] for k in ("train_iterations", "sim_count", "worker_count",
                                      "worker_games_per_worker")},
      "BALANCED_FIRST=", os.environ.get("GOMOKU_BALANCED_FIRST", "0"))
M.train(cfg)
