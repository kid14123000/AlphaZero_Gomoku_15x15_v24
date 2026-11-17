#!/usr/bin/env python3
"""
v36 远程推理客户端 (工兵/主进程侧)
====================================
RemotePredictor: MCTS leaf 推理客户端 — Unix socket → infer_server_v36

设计: 每个 worker 内部保持串行单 leaf 推理, 9 工兵 + 1 主的请求并发到达 server
      → server 攒批 (≤9 或 3ms) GPU forward。行为零改变
      (eval 模式 BN 用 running stats → batch 大小无关)。

来源: legacy/tools/Gomoku-remote-predict.py 的 RemotePredictor 部分
  B1. socket 协议不变 (4 字节大端长度前缀 + pickle)
  B2. predict() 增加 players 透传参数 (server 侧忽略, 仅为保持 v36 接口一致)
  B3. 不含 SearchAgent (v36 已有自己的 AlphaZeroAgent MCTS, 不再复刻第二份)

v36 接入方式 (scripts/../Gomoku-v0_pt_v36.py 内的钩子):
  环境变量 GOMOKU_INFER_SOCK 存在 → AlphaZeroAgent.predict_inference 走本客户端

用法:
    from remote_predict_v36 import RemotePredictor
    predictor = RemotePredictor('/tmp/gomoku_infer.sock')
    p, v = predictor.predict(board_batch, last_moves, players)
    predictor.update_weights('/path/to/weights_it310_v36.pt')   # 热加载
    predictor.close()
"""
import os
import pickle
import socket
import struct
import time

import numpy as np


class RemotePredictor:
    def __init__(self, sock_path='/tmp/gomoku_infer.sock',
                 connect_retries=60, timeout=120.0):
        self.sock_path = sock_path
        self.sock = None
        self.timeout = timeout
        self._connect(connect_retries)

    def _connect(self, retries):
        last_err = None
        for _ in range(retries):
            try:
                self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                self.sock.settimeout(self.timeout)
                self.sock.connect(self.sock_path)
                return
            except Exception as e:
                last_err = e
                if self.sock is not None:
                    try:
                        self.sock.close()
                    except Exception:
                        pass
                    self.sock = None
                time.sleep(0.5)
        raise ConnectionError('无法连接 inference server {}: {}'.format(self.sock_path, last_err))

    def _recv_exact(self, n):
        buf = b''
        while len(buf) < n:
            chunk = self.sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError('server 连接断开')
            buf += chunk
        return buf

    def predict(self, board_batch, last_moves=None, players=None):
        """单 leaf 推理 (N=1), 与 v36 predict_inference 返回形状一致: (p (1,225) log概率, v (1,1))
        board_batch: (1,15,15) canonical numpy"""
        b = np.asarray(board_batch)
        lm = None
        if last_moves is not None and len(last_moves) > 0:
            lm = last_moves[0]
        # players 透传 (server 侧忽略; 旧协议只带 board/last_move → 保持兼容)
        msg = pickle.dumps((b[0], lm))
        self.sock.sendall(struct.pack('>I', len(msg)) + msg)
        hdr = self._recv_exact(4)
        (n,) = struct.unpack('>I', hdr)
        payload = self._recv_exact(n)
        p, v = pickle.loads(payload)
        return p[None], v[None]

    def update_weights(self, weights_path):
        """通知 server 热加载权重 (主进程每代保存新权重后调用)"""
        msg = pickle.dumps(('UPDATE', weights_path))
        self.sock.sendall(struct.pack('>I', len(msg)) + msg)
        hdr = self._recv_exact(4)
        (n,) = struct.unpack('>I', hdr)
        _ = self._recv_exact(n)
        return True

    def close(self):
        try:
            if self.sock is not None:
                self.sock.close()
        except Exception:
            pass
        self.sock = None
