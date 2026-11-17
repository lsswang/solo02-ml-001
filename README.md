# 可微掩码注意力：独立算子 + 张量图融合

Python 3.11 / CPU 工程。把原有“固定缩放、无旁路输出的注意力树”扩展为可训练
`scale` 与 `bias`、支持概率旁路输出、并正确处理整行无允许 key（全掩码行）的
可微注意力。所有数值计算仅依赖 NumPy。

## 安装与复现

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python demo.py                 # 端到端示例（见下文场景）
python -m pytest tests/ -q     # 全部正确性测试（40+ 用例）
```

无需下载模型、训练网络、GPU 或外部数据。PyTorch 不是依赖；测试使用一个与
实现完全独立的 NumPy 参考实现和中心有限差分做交叉验证，避免“融合图与对照图
共用同一个错误 VJP”。

## 模块

| 文件 | 职责 |
| --- | --- |
| `attention.py` | 独立掩码注意力算子：`forward` / `vjp` 纯函数，外加薄封装 `Attention`。不依赖图即可直接调用。 |
| `graph.py` | 张量 DAG 解释器：结构/形状/dtype 校验、前向、联合反向 VJP，以及融合后的 `attention` 节点执行。 |
| `optimizer.py` | 子图融合：把合法注意力链改写为 `attention` 节点，并做保守死代码消除。 |
| `examples.py` | 旧模型（兼容签名）、单块/两块新模型、随机合法图与 feed 生成器。 |
| `demo.py` | 可观察的端到端演示。 |
| `tests/` | 算子手算例、独立参考、有限差分、图对照、优化器与错误路径测试。 |

## 独立算子

```python
import numpy as np, attention

q = np.random.randn(2, 3, 4)          # [B, L, D]
k = np.random.randn(2, 5, 4)          # [B, S, D]
v = np.random.randn(2, 5, 6)          # [B, S, E]
scale = np.asarray(0.25)              # 零维、可微
bias = np.random.randn(1, 3, 5)       # 可广播到 [B, L, S]，可省略
allow = np.ones((2, 3, 5), bool)      # 可广播布尔掩码，可省略（默认全允许）
allow[0, 1] = False                   # 整行不允许任何 key

out, cache = attention.forward(q, k, v, scale, allow=allow, bias=bias)
grads = attention.vjp(cache, np.ones_like(out))
grads['q'].shape == q.shape           # 各梯度形状与输入一致
grads['bias'].shape == bias.shape     # 广播梯度已合并回 bias 原形状
```

数值合同（按实数表达式解释，不绑定具体舍入顺序）：

* 分数 `scores = scale * Q @ Kᵀ + bias`，只在 `allow=True` 的位置归一化。
* 全掩码行：概率行与输出行定义为零，且对 Q/K/V/scale/bias 的贡献梯度全零。
* `scale` 是零维可微标量；`bias` 缺省为零且不返回其梯度；`allow` 不求导。
* 输出 dtype 与 Q/K/V 一致（float32 或 float64，一次运行统一）。

## 融合范围

`optimizer.optimize(spec, feeds=None)` 实际融合两种等价合法子图（`bias` 可
省略，`mul` 两个操作数顺序均可）：

1. `softmax_last(where_mask(scale·(QKᵀ) [+ bias], allow)) · V`
2. `softmax_last(where_mask((scale·Q)Kᵀ [+ bias], allow)) · V`

融合策略与边界：

* 融合在输出 matmul 位置原地改写为 `attention` 节点（保留原节点 id，输出与
  其他支路引用不悬空），其 `scale` 字段指向运行时 input 节点。
* **仅零维 scale 视为合法融合**。提供 `feeds` 时按运行时形状判定；不提供时
  保守要求 scale 操作数是 input 节点，并在 `evaluate` 时再次检查。其他可广播
  `mul` 是普通图运算，原样保留。
* 旧导出器的 `fixed_scale` 提示一律剥离，绝不替代运行时 input，也不会让该
  input 的梯度消失。
* 链上节点只有在“不存在融合外部存活消费者”时才删除（迭代到不动点）。因此
  概率旁路、被其他支路复用的中间张量会自动保留；无旁路时整条逐节点链消失。
* 不同 `allow`/`bias`/`scale` 的子图各自生成独立节点与独立前向缓存，不共享
  数值或反传状态。
* 源 `spec` 与 `feeds` 深拷贝处理，绝不被修改；同一图可反复替换形状/参数
  复用；重复 `optimize` 幂等；无融合模式的合法图语义不变；不会返回悬空
  引用。

## 反传中间量策略

`attention.forward` 的缓存只保存 VJP 必需的量：Q/K/V、零维 scale、未缩放的
`QKᵀ`、概率 `P`、已物化的布尔掩码（广播到 `[B,L,S]`）、bias 原形状。不缓存
分数或掩码后的 `-inf` 张量。VJP 路径：

```
dV   = Pᵀ · dOut
dP   = dOut · Vᵀ
dZ   = where(allow, P · (dP - Σ(dP·P)), 0)     # 全掩码行 P=0，自动为零
dQK  = scale · dZ
dQ   = dQK · K
dK   = dQKᵀ · Q
dscale = Σ(dZ · QKᵀ)
dbias  = 将 dZ 按广播轴求和还原到 bias 原形状
```

图级 `graph.backward` 按拓扑逆序传播，所有浮点 input（含被多分支共享的参数、
零维 scale）的 cotangent 用 `unbroadcast` 还原形状后累加；布尔 input 不返回
梯度；多个输出的 seeds 联合反传。

## 验证方法与容差

* **未融合对照**：每个随机/确定性案例都同时跑逐节点原图与融合图，比较每个
  输出值和全部浮点 input 的联合梯度。
* **独立参考**：`tests/test_attention.py` 中用显式掩码归一化（逐元素 Python
  循环 + 独立公式）重写注意力，与算子互查。
* **手算小例**：B=L=D=E=1、S=2 的硬编码数值与硬编码导数；另含手算全掩码行。
* **中心有限差分**：float64、步长 `h=1e-6`，对 `sum(dOut·Out)` 标量损失扰动
  Q/K/V/scale/bias 的每个元素；全图另做一次端到端 FD。FD 输入取远离数值极端
  的小值（正态分布、scale 约 0.5 以内）。
* 容差：float64 `atol=1e-8, rtol=1e-6`；float32 `atol=2e-5, rtol=2e-4`。
  实测 float64 FD 最大误差约 4e-9。
* 覆盖：B∈[1,3]、L,S∈[1,8]、D,E∈[1,6]；稠密/稀疏/全假掩码；bias 与 allow
  的省略前导维、长度 1 轴广播；最多 3 个注意力块、60 节点；float32/float64；
  两种融合模式；概率旁路；两块共享参数；形状不相容、未知引用、重复 id、非
  拓扑、缺失 feed、混合 dtype、非布尔掩码等错误路径。

## `demo.py` 可观察的场景

1. 旧 DAG + 全掩码行 + 动态（可训练）scale：零行输出、概率行和、scale 梯度、
   `fixed_scale` 已剥离。
2. 新单块（两种融合模式）：广播 bias、空行零输出/零概率、bias 梯度还原形状。
3. 无旁路输出时融合后真正执行的节点（整条 softmax/mask 链消失）。
4. 两块共享全部参数：两个 `attention` 节点，scale 梯度累加两条支路。
5. float32 下按合同容差通过。
