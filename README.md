# AlphaZero Gomoku 15×15 — v24b

15×15 五子棋（无禁手）的 AlphaZero 自对弈实现。这一支是 **v24 原版 + 一处行为改动**：
黑方在空盘第一手不按搜索走，而是走「落子后价值最接近均衡」的一手。

上一支（v4 线）在 [kid14123000/AlphaZero_Gomoku_15x15](https://github.com/kid14123000/AlphaZero_Gomoku_15x15)，
两支的网络结构和输入语义不同，权重不通用。

---

## 网络

`AlphaZeroNet`，**2,501,226** 参数。

| 项 | 值 |
|---|---|
| 主干 | 输入 conv 3×3 (4→128) + 8 个残差块（每块 128→128→128）+ BN |
| policy 头 | conv 3×3 (128→1) → BN → log_softmax，直接输出 225 个点的 log 概率 |
| value 头 | conv 3×3 (128→1) → BN → FC(225→1) 单层 |
| 输入平面 | 4 通道 |

输入平面的语义（`make_input`）：

```
ch0  当前玩家棋子
ch1  对手棋子
ch2  恒 1.0            ← v24 的关键：撤销了执色编码
ch3  上一步落子位置
```

`ch2` 恒 1 是有意的。早期版本用 ch2 编码「我执黑还是执白」，网络很快学会走捷径——
空盘状态下就能凭「我是黑→赢 / 我是白→输」给出极端 value，不读棋盘。
把执色信息彻底拿掉之后，黑白视角的输入完全对称，这条捷径在物理上不存在。

代价是网络分不清自己在执黑还是执白，所以 value 的正负只表示「输入里标记为 +1 的一方」
的局势高低，判断均衡只能比绝对值。v24b 的均衡首手正是这么用的。

---

## v24b 改了什么

只有一处行为。空盘（`move_no == 0`）时黑方不走搜索：

```
对中心 5×5 的每个空点：
    把本方子放上去 → 网络前向 → 取 |value|
取 |value| 最小的前 3 手里随机挑一个
```

- 只替换落子位置；返回的 `prob` 是单手 one-hot，首手不参与 policy 监督，不影响后续训练
- 开关：`GOMOKU_BALANCED_FIRST`，默认 `1`（开）；`BALANCED_FIRST=0 ./run_v24b.sh` 得到纯 v24 基线

两个细节不这么做会翻车：

| 细节 | 原因 |
|---|---|
| 候选限定中心 5×5 | 不限的话冷网络会在边角点上「恰好估值≈0」，那是估值错误不是真均衡 |
| 在 \|value\| 最小的前 3 手里随机 | 纯 argmin 会让每局首手完全一样（实测确认过） |

实测（权重 `weights_it73_v24.pt`，CPU）：

```
空盘 value（黑先手）           = +0.6976      ← 与历史记录 +0.698 吻合，说明通道语义对
均衡首手 6 次抽样  (5,5)|V|=0.176 / (5,9) 0.245 / (9,5) 0.091   ← 有梯度、落在中心
v24 默认（200 sims）首手  (6,9)|V|=0.765                        ← 明显更极端
```

---

## 跑起来

```bash
pip install -r requirements.txt

./run_v24b.sh                        # 均衡首手 ON
BALANCED_FIRST=0 ./run_v24b.sh       # 纯 v24 基线（对照组）
SIMS=800 WORKERS=8 TARGET_ITERS=75 ./run_v24b.sh

REGION=5 TOPK=3 BALANCED_FIRST=1 ./run_v24b.sh   # 候选区 / 随机池
PY=/path/to/python ./run_v24b.sh                 # 指定解释器
```

驱动是 `run_quick.py`（读环境变量拼训练配置），主脚本 `Gomoku-v0_pt_v24b.py`，
工兵 `Gomoku-worker_v24b.py`。

验证首手行为（需要一份权重，`V24_CKPT` 指路径）：

```bash
V24_CKPT=./gomoku_checkpoints_v24/weights_it73_v24.pt python validate_v24b.py
```

`web/` 是训练监控看板（Flask），`python web/app.py` 起在 `127.0.0.1:8765`，
读 `logs/metrics.jsonl` 画曲线。它原本对接一台远程训练机（`web/slot_config.py` 里的
槽位配置、`web/app.py` 里的 scp 拉取逻辑），单机自对弈下这些远程功能会报「未配置」，
不影响本地看板。

---

## 文件

```
Gomoku-v0_pt_v24.py       v24 原版（未改动，788 行单文件：网络 + MCTS + 训练循环）
Gomoku-v0_pt_v24b.py      v24b = v24 + 均衡首手
Gomoku-worker_v24.py      v24 工兵
Gomoku-worker_v24b.py     v24b 工兵（只改了 import 的模块名）
run_quick.py              训练驱动
run_v24b.sh               启动脚本
validate_v24b.py          首手行为验证
requirements.txt
web/                      训练监控看板
```

---

## 不含什么

**权重**。`gomoku_checkpoints_v24/` 在 `.gitignore` 里。这一支训练到 `it73`，
但那个检查点没有可引用的外部实力锚点（没跑过 Rapfi 之类的外部引擎），
单独放出来容易被误读成「这个棋力」。要复现就从 `it0` 自己练。

**多机分发启动器**。这一支只保留单机自对弈的驱动。

---

## 与 v4 线的关系

v4 线（另一个仓）是 junxiaosong 派生的并行训练工程，网络输入也是 4 平面但 ch2 语义不同，
训练策略走的是「根先验噪声 + 分层采样」。这一支是 v24 的极简单文件实现，
token 不通用，训练超参也不通用，只能当两个独立实验看。
