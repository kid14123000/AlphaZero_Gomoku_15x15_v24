# 🔬 五子棋 AlphaZero Web 驾驶舱

AlphaZero 五子棋项目的**可视化 + 评估 + 运维**一条龙工具链。训练在远程服务器上跑,这个 Web 应用负责把整条链路打通:

```
对弈试玩 → 局面存档 → 拆解(看网络在想什么) → 评估(棋力测评/PK) → 运维(训练监控/权重同步)
```

**10 个页面 + 31 个 API**,共用同一套权重与局面数据。项目从 v1 一路训到 **v37**,这个工具链全程跟着迭代——它的价值不在于"能下棋",而在于**让你看得见网络在想什么、并能用多种口径验证它到底强了多少**。

> 🔄 **监控目标可切换**:驾驶舱默认跟主线 **v37**(`app.py` 的 `TRAIN_VERSION`),含 v37 专属的 **🏛 权重门控面板**;
> 「自对弈数据观察」页有**数据源下拉**(v37 主线 / v36 上一版),各自独立索引。
> 看旧版本:`GOMOKU_TRAIN_VERSION=v36 GOMOKU_TRAIN_LOG=/root/pk_v8/train_v36.log python3 app.py`

> 📖 配套文章:仓库 `blog_articles/CSDN_v36实战_*.md`(五篇实战系列),其中第五篇《评估工具》讲的就是这里大部分页面的设计思路。

---

## 🗺️ 页面导航(10 个)

### 对弈与试玩

| 页面 | 地址 | 用途 |
|---|---|---|
| ⚫ **对弈** | `/` | 人机 / AI 自战 / 人机对战;AI 思考过程可视化 + 胜率走势;实战局面一键存档 |
| 🎯 **9×9 对弈** | `/game9.html` | 9×9 小棋盘对弈(早期 v16 归一化对照实验遗留,保留备用) |

### 研究与可视化(看网络在想什么)

| 页面 | 地址 | 用途 |
|---|---|---|
| 🔬 **摆谱研究** | `/study` | 自由摆棋 → 深度思考(网络推荐 + MCTS 推荐 + 胜率) |
| 🧠 **网络前向** | `/network` | 逐层热力图 + policy/value 双头传导链;policy Top5 直接标注到摆谱棋盘(金色★) |
| 🔍 **MCTS 搜索** | `/mcts` | 搜索过程可视化:候选分布、模拟区间回放、**n-gram 词频分析**、最佳决策点 ★ 标注、参数面板 |
| 🗂 **自对弈数据观察** | `/replay` | 翻看训练对局(含主进程局);白胜率趋势;④ **棋谱胜率分析**(双口径 + 80% 预知线 + 即时重算) |
| 🧬 **开局分布演化** | `/evolution` | 各代开局落点分布演化(天元率 / 中心率 / 边角率),看"布局学没学会" |

### 评估(Rapfi / 出题)

| 页面 | 地址 | 用途 |
|---|---|---|
| ⚔️ **棋力测评 vs Rapfi** | `/eval` | 接入开源引擎 Rapfi 做**绝对棋力锚点**(depth 阶梯 1/2/3/5/8/10);支持指定黑 1 位置、棋谱自动命名 |
| 📝 **出题评估** | `/quiz` | 固定题库命中率(奇偶题集 / 混合题集),做题趋势曲线 |

### 训练运维

| 页面 | 地址 | 用途 |
|---|---|---|
| 📊 **驾驶舱** | `/dashboard` | 远程训练状态(SSH 实时拉取):loss 趋势、工兵进程、权重列表、**一键同步权重** |

---

## 🚀 快速开始

### 依赖

```bash
pip install flask numpy gym boardgame2 torch
```

> `boardgame2` 提供五子棋环境,需从项目源码安装(见仓库根目录说明)。

### 启动

```bash
cd gomoku_web
python3 app.py
```

- 访问 **http://127.0.0.1:8765**
- **无密码**（默认不校验登录）。要启用口令：`GOMOKU_PASSWORD=xxx python3 app.py`
- 端口/主机可覆盖：`GOMOKU_WEB_PORT=8785 GOMOKU_WEB_HOST=127.0.0.1 python3 app.py`
- 长跑前建议 `caffeinate -i python3 app.py`(macOS 防止系统挂起进程导致端口在监听但请求全超时)

### 远程服务器连接(可选,只有驾驶舱/同步功能需要)

凭证**不入库**,放在 `gomoku_web/creds.local.json`(已在 `.gitignore` 中):

```json
{ "host": "你的服务器", "port": 22, "user": "root", "pass": "你的密码" }
```

也支持环境变量覆盖:`GOMOKU_SSH_HOST` / `GOMOKU_SSH_PORT` / `GOMOKU_SSH_USER` / `GOMOKU_SSH_PASS`。

> 🚨 **安全约定**:凭证只放 `creds.local.json`,永不写进代码/DOC/提交。提交前先跑一遍凭证扫描(见文末"安全")。

---

## 🔗 核心链路:对弈 → 存档 → 研究

```
对弈页: 人机 / AI 自战 → 💾 保存棋谱 → positions.json(同名覆盖,跨页共享)
                                              ↓
        网络前向 ←── 加载局面 ──→ 看 policy / value / 双头传导链
        MCTS 搜索 ←─ 加载局面 ──→ 看搜索树 / 候选点 / n-gram 词频 / 参数对比
        摆谱研究 ←── 加载局面 ──→ 深度思考分析
```

三个研究页面共用 `gomoku_web/positions.json`。对弈中遇到的局面(比如"AI 这步又飞了")可以一键存档,切到研究页拆解——**这是排查"它为什么不堵"的标准姿势**。

---

## ⚙️ 各页面功能详解

### 🧠 网络前向(`/network`)

看**结果**优先,不对再往下看过程。

- **Policy / Value 输出** —— 直接给落点概率和胜率
- **① 双头传导链**:policy(`conv→BN→ReLU→softmax`)与 value(`conv→BN→ReLU→fc→tanh`)逐部件热力图 + 标量,和主干 8 个残差块拼成完整链条
- **Policy Top5 标注到摆谱棋盘**(金色★ = top1,排名圈深→浅)
- 棋盘**数字版(实际视角 ±1)vs canonical(当前玩家视角)** 并列对照,直观理解视角切换
- ④ 权重参数分布(每层直方图,查死神经元/塌缩)⑤ 网络计算图

> ⚠️ 探针注意:输入必须走正规的 4 通道构造(`ch0` 当前方 / `ch1` 对手 / `ch2` 恒 1 / `ch3` 上一步)。直接喂全零张量会得到假的"病态"结论。

### 🔍 MCTS 搜索(`/mcts`)

- 摆谱 → 跑模拟 → **🎯 最终候选点**(访问热力图 + top 列表)
- **模拟循环区间回放**:左右滑块选区间(如"模拟 5~10"),逐条展示区间内每次搜索的完整递归路径
- **🏁 n-gram 词频分析**:统计搜索路径里 1~15 步连续子串的出现次数(任意位置),带**步数滑杆**(快捷 [3/5/7/9])和位置分布(开/中/尾)——用来回答"搜索到底在反复走哪几条线"
- **最佳决策点标注**:热力图 / 轨迹回放 / 摆谱大棋盘都标 **best 金色★ + TOP6 排名圈**
- **⚙️ 参数面板**:`c_init` / `c_base` / `噪声占比` / `Dirichlet α`(默认值 = 训练配方,`?` 悬停看含义);改参数重跑看候选变化;配置可命名保存(localStorage)

### 🗂 自对弈数据观察(`/replay`)

- 翻看训练对局轨迹(**同时扫 `worker_data/` 和 `game_data/`** —— 主进程那 10 局/代不能漏)
- **白胜率趋势 + 对比线**(目标带 45-55%)
- **④ 棋谱胜率分析**(本地棋谱逐手重算):
  - 双口径:**`best_qa`**(该局面最优应手,看"局面本身胜率")与 **`actual_qa`**(棋谱实际落子,看"那手好不好"),前端下拉即时切换
  - **80% 预知线** + 破线步数标注
  - 📌 回放每步自动显示"下一步"对比 + 🔍 **即时重算此局面**(只跑你需要的那一步)
- 典型用法:加载 40 手棋谱 → 第 13 手权重推荐 F8(+0.88)而实际下 H9 → **实锤失误**

### ⚔️ 棋力测评(`/eval`)—— vs Rapfi

**绝对棋力锚点**。Rapfi 在项目里的身份是"期末考卷",绝不当训练对手(项目铁律:纯自举训练,不注入外部知识)。

- depth 阶梯:1 / 2 / 3 / 5 / 8 / 10
- **可指定黑 1 位置**(留空 = 自动),用于复现特定开局
- 棋谱自动命名带后缀(`_权重_sim_d{depth}_执色`),方便归档比对

> ⚠️ 读战绩前必做两件事:① **确认对局独立性**(看步数分布,别是同一盘复制 N 次)② **拆黑白看**(ring 换先后执黑仍占 55-75%)。

### 🧬 开局分布演化(`/evolution`)

各代开局落点分布(天元率 / 中心率 / 边角率),用来回答"policy 到底学没学会占中心"。

**观察线:中心率 > 20-30% 才算学会开局。** 早期版本全程天元率 0.2-2.4%(≈没有)。

### 📊 驾驶舱(`/dashboard`)

- 训练进度 / loss 趋势 / 工兵进程状态(**每次请求实时 SSH 拉取**,不是缓存)
- **权重同步**:
  - 内置后台自动同步线程 + 页面"同步权重"按钮
  - 逐文件比对服务器权重列表,**已存在的文件也做完整性校验**(≥6MB + zip 结构),**截断/损坏的自动删除重拉**
  - 曾经吃过"只按文件名判存在 → 截断文件被永久跳过,还显示已最新"的亏,所以有了这道校验

---

## 💾 数据存储

| 数据 | 位置 | 说明 |
|---|---|---|
| 局面棋谱 | `gomoku_web/positions.json` | 跨页共享,同名覆盖;API `/api/positions/{save,list,delete}` |
| 棋谱分析记录 | `gomoku_web/analysis_records.json` | 分析进度与结果(不入库) |
| 做题结果 | `gomoku_web/quiz_results.json` | 不入库 |
| MCTS 参数配置 | 浏览器 `localStorage`(key `mcts_cfgs`) | 本地保存 |
| 权重 | 项目根 `gomoku_checkpoints_v1 ~ v36/` | **版本后缀命名防混淆**(`weights_it{N}_vNN.pt`) |
| 发布权重 | `weights_release/` | 精选权重(唯一入库的权重) |
| 服务器凭证 | `gomoku_web/creds.local.json` | **不入库** |

---

## 📁 文件结构

```
gomoku_web/
├── app.py                 # Flask 后端: 对弈 / 前向 / MCTS / 评估 / 出题 / 棋谱分析 / SSH 监控
├── replay_api.py          # 自对弈数据索引与读取(双目录扫描)
├── train_monitor.py       # 训练监控模块
├── creds.local.json       # 服务器凭证(不入库)
└── static/
    ├── index.html         # 对弈
    ├── game9.html         # 9×9 对弈
    ├── study.html         # 摆谱研究
    ├── network.html       # 网络前向
    ├── mcts.html          # MCTS 搜索
    ├── replay.html        # 自对弈数据观察
    ├── evolution.html     # 开局分布演化
    ├── eval.html          # 棋力测评 vs Rapfi
    ├── quiz.html          # 出题评估
    └── dashboard.html     # 驾驶舱
```

相关工具脚本在仓库根 `scripts/`(PK 驱动、体检探针、权重同步、安全检查等)。

---

## 🔌 API 端点总览(31 个)

| 分组 | 端点 |
|---|---|
| **对弈** | `/api/start` `/api/play` `/api/state` |
| **局面存档** | `/api/positions/{save,list,delete}` |
| **可视化** | `/api/mcts_viz` `/api/network_viz` `/api/weight_viz` `/api/net-trend` `/api/opening_dist` |
| **权重要** | `/api/weights` |
| **训练监控/同步** | `/api/training` `/api/sync-weights` `/api/sync-state` |
| **棋力测评** | `/api/eval/{start,step,state,reset}` |
| **出题评估** | `/api/quiz/{run,result,save_result,results,trend}` |
| **棋谱分析** | `/api/analysis/{start,status,cancel,list,probe,records}` |
| **摆谱研究** | `/api/study/analyze` |

---

## 🛠️ 技术栈

Python 3.11 · Flask · PyTorch · NumPy · gym / boardgame2 · 原生 HTML/JS Canvas 可视化(无前端框架)

---

## 🔐 安全

- 凭证只在 `creds.local.json`(已 gitignore),代码自动读取,支持环境变量覆盖
- 提交前自查:`git diff --cached | grep -iE "sshpass|password|passwd|seetacloud|[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}"`
- 服务器巡检:`python scripts/security_check.py`(检查登录记录、可疑进程、异常文件)
- 本应用只监听 `127.0.0.1`,不对外暴露
