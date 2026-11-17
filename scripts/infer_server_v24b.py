#!/usr/bin/env python3
"""
v36 攒批推理服务 (跨 worker 集中推理, 2026-09-13 适配 v36)
============================================================
独立进程独占 GPU, 监听 Unix socket, 接收各工兵/主进程的 MCTS leaf predict 请求,
攒批 (≤max_batch 或 max_wait 超时) 一次 forward, 逐个回复。

来源: legacy/tools/Gomoku-inference-server.py (v12 时代原版) — 本版仅做 v36 适配:
  A1. 网络构造 → v36 的 AlphaZeroNet(128, 8, 15)
  A2. 输入组装 → 与 v36 AlphaZeroAgent.predict_inference 逐行一致 (ch2 恒定 1)
  A3. 权重加载 → 兼容 v36 state_dict 格式 + strict=False
  A4. 保留: 攒批 / UPDATE 热加载 / 长度前缀协议
  A5. 新增: 运行统计 (QPS / 平均 batch / forward 耗时), 便于性能对照

协议 (与旧版完全相同, 便于对照):
  4 字节大端长度前缀 + pickle
  请求: pickle((board, last_move))          board (15,15) canonical, last_move (2,) 或 None
  响应: pickle((p_logprob, v))              p (225,) log概率, v (1,)
  命令: pickle(('UPDATE', weights_path))    → 热加载权重, 回复 4 字节 0 ack

行为保证: eval 模式 BN 用 running stats → batch 大小不影响单样本结果
          (batch=1 等价性已由旧版验证; 本版需按 E1/E2 重新验证)

环境变量:
  GOMOKU_INFER_SOCK  socket 路径 (默认 /tmp/gomoku_infer.sock)
  GOMOKU_MAX_BATCH   攒批上限 (默认 9)
  GOMOKU_MAX_WAIT    攒批等待 (默认 0.003 秒)
  GOMOKU_WEIGHTS     初始权重路径 (默认 <ckpt>/weights.pt)
  GOMOKU_CKPT_DIR    检查点目录 (默认 <repo>/gomoku_checkpoints_v24)

用法: python3 scripts/infer_server_v24b.py
"""
import importlib.util
import logging
import os
import pickle
import queue
import socket
import struct
import sys
import threading
import time

import numpy as np

logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOCK_PATH = os.environ.get('GOMOKU_INFER_SOCK', '/tmp/gomoku_infer.sock')
INPUT_CHANNELS = 4


def _recv_exact(conn, n):
    buf = b''
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


class InferenceServer:
    def __init__(self, net_builder, weights_path, device,
                 max_batch=9, max_wait=0.003, sock_path=SOCK_PATH):
        self.sock_path = sock_path
        self.max_batch = max_batch
        self.max_wait = max_wait
        self.device = device
        self.weights_path = weights_path
        self.net = net_builder().to(device)
        self.net.eval()
        if weights_path and os.path.exists(weights_path):
            self.load_weights(weights_path)
            logging.info('server 已加载权重 %s', weights_path)
        else:
            logging.warning('server 无权重文件 (从随机初始化开始): %s', weights_path)
        self.req_q = queue.Queue()
        self.net_lock = threading.Lock()
        self.running = True
        # 统计
        self.stat_requests = 0
        self.stat_forwards = 0
        self.stat_batch_sum = 0
        self.stat_forward_time = 0.0
        self._stat_last = time.time()

    def load_weights(self, path):
        import torch
        state = torch.load(path, map_location=self.device, weights_only=False)
        if isinstance(state, dict) and 'state_dict' in state:
            state = state['state_dict']
        self.net.load_state_dict(state, strict=False)
        self.net.eval()
        logging.info('server 权重已更新: %s', os.path.basename(path))

    def handle_conn(self, conn):
        try:
            while True:
                hdr = _recv_exact(conn, 4)
                if hdr is None:
                    break
                (n,) = struct.unpack('>I', hdr)
                payload = _recv_exact(conn, n)
                if payload is None:
                    break
                msg = pickle.loads(payload)
                if (isinstance(msg, tuple) and len(msg) == 2
                        and isinstance(msg[0], str) and msg[0] == 'UPDATE'):
                    with self.net_lock:
                        self.load_weights(msg[1])
                    conn.sendall(struct.pack('>I', 4) + struct.pack('>I', 0))
                    continue
                # v36 客户端可能发 (board, last_move, player) → 兼容取前两项
                if isinstance(msg, tuple) and len(msg) >= 2:
                    board, last_move = msg[0], msg[1]
                else:
                    logging.warning('非法请求格式: %r', type(msg))
                    continue
                self.req_q.put((conn, board, last_move))
        except Exception as e:
            logging.warning('连接处理异常: %s', e)
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def batch_loop(self):
        while self.running:
            batch = []
            deadline = time.time() + self.max_wait
            while len(batch) < self.max_batch:
                remaining = deadline - time.time()
                if remaining <= 0:
                    break
                try:
                    batch.append(self.req_q.get(timeout=remaining))
                except queue.Empty:
                    break
            if not batch:
                time.sleep(0.0005)
                continue
            conns, boards, lms = zip(*batch)
            try:
                t0 = time.time()
                with self.net_lock:
                    ps, vs = self._forward(np.stack(boards), list(lms))
                dt = time.time() - t0
                self.stat_requests += len(batch)
                self.stat_forwards += 1
                self.stat_batch_sum += len(batch)
                self.stat_forward_time += dt
                for conn, p, v in zip(conns, ps, vs):
                    payload = pickle.dumps((p, v))
                    try:
                        conn.sendall(struct.pack('>I', len(payload)) + payload)
                    except Exception:
                        pass
                now = time.time()
                if now - self._stat_last >= 30.0:
                    avg_b = self.stat_batch_sum / max(self.stat_forwards, 1)
                    qps = self.stat_requests / (now - self._stat_last)
                    logging.info('统计: 请求 %d | forward %d | 平均batch %.2f | %.0f req/s | forward均耗时 %.1fms',
                                 self.stat_requests, self.stat_forwards, avg_b, qps,
                                 self.stat_forward_time / max(self.stat_forwards, 1) * 1000)
                    self.stat_requests = 0
                    self.stat_forwards = 0
                    self.stat_batch_sum = 0
                    self.stat_forward_time = 0.0
                    self._stat_last = now
            except Exception as e:
                logging.error('forward 失败: %s', e)

    def _forward(self, x, last_moves):
        """与 v36 AlphaZeroAgent.predict_inference 的输入组装逐行一致"""
        import torch
        inp = np.zeros((x.shape[0], INPUT_CHANNELS) + x.shape[1:], dtype=np.float32)
        inp[:, 0] = x > 0
        inp[:, 1] = x < 0
        inp[:, 2] = 1.0          # v24: ch2 恒定 1, 无执色信息
        for i, lm in enumerate(last_moves):
            if lm is not None:
                inp[i, 3, int(lm[0]), int(lm[1])] = 1.0
        t = torch.as_tensor(inp, device=self.device)
        with torch.inference_mode(), torch.cuda.amp.autocast():
            p, v = self.net(t)
        return (p.float().detach().cpu().numpy(),
                v.float().detach().cpu().numpy())

    def run(self):
        if os.path.exists(self.sock_path):
            os.unlink(self.sock_path)
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.bind(self.sock_path)
        sock.listen(64)
        logging.info('server 监听 %s (max_batch=%d, max_wait=%.3fs)',
                     self.sock_path, self.max_batch, self.max_wait)
        threading.Thread(target=self.batch_loop, daemon=True).start()
        try:
            while self.running:
                conn, _ = sock.accept()
                threading.Thread(target=self.handle_conn, args=(conn,), daemon=True).start()
        except KeyboardInterrupt:
            pass
        finally:
            self.running = False
            sock.close()
            try:
                os.unlink(self.sock_path)
            except Exception:
                pass


def main():
    import torch
    spec = importlib.util.spec_from_file_location('gomoku_v24b',
                                                  os.path.join(BASE, 'Gomoku-v0_pt_v24b.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mod.device = device
    logging.info('使用设备: %s', device)

    max_batch = int(os.environ.get('GOMOKU_MAX_BATCH', '9'))
    max_wait = float(os.environ.get('GOMOKU_MAX_WAIT', '0.003'))
    ckpt_dir = os.environ.get('GOMOKU_CKPT_DIR',
                              os.path.join(BASE, 'gomoku_checkpoints_v24'))
    weights_path = (os.environ.get('GOMOKU_WEIGHTS')
                    or os.path.join(ckpt_dir, 'weights.pt'))

    server = InferenceServer(
        net_builder=lambda: mod.AlphaZeroNet(
            conv_channels=128, residual_blocks=8, board_size=15),
        weights_path=weights_path,
        device=device,
        max_batch=max_batch,
        max_wait=max_wait,
    )
    server.run()


if __name__ == '__main__':
    main()
