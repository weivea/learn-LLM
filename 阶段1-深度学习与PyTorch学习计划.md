# 阶段 1 · 深度学习与 PyTorch 学习计划

> 配套总路线：[大模型学习路线.md](大模型学习路线.md)
> 前置进度：[数学知识点掌握表.md](数学知识点掌握表.md)
> 计划周期：**3 周，12 个独立 Step，约 30–36 小时**
> 最终项目：**用纯 PyTorch 从零训练 Embedding + MLP 字符级语言模型**
> 当前状态：**准备开始 Step 1.1**

---

## 1. 为什么现在可以进入阶段 1

阶段 0 的 5 个核心门槛已经全部通过：矩阵与张量形状、链式法则、梯度下降、
softmax、交叉熵和困惑度都已经能够独立复现。现有 37 个数学知识点中，
31 个为 `🟢 能独立复现`，6 个为 `🟡 学过但模糊`。

这 6 个模糊点不应继续阻塞实践，而应在代码中按需回补：

| 待巩固知识 | 在阶段 1 的回补位置 |
|------------|---------------------|
| 2.7 计算图与自动求导 | Step 1.2–1.3 |
| 2.8 常见函数的导数 | Step 1.2 |
| 2.9 局部最小与鞍点 | Step 1.7 |
| 2.10 梯度消失与爆炸 | Step 1.8 |
| 2.11 积分基础 | 本阶段不阻塞，后续遇到连续分布再复习 |
| 3.11 最大似然估计 | Step 1.6、1.9–1.11 |

### 当前环境结论

- 机器：macOS Apple Silicon `arm64`
- 现有 `.venv`：Python 3.9.6，已安装 NumPy，**尚未安装 PyTorch**
- PyTorch 当前官方建议 Python 3.10–3.14，因此 Step 1.1 会先安全地准备
  Python 3.11 或 3.12 环境，再验证 CPU/MPS；不直接破坏现有数学环境

---

## 2. 阶段目标与边界

### 2.1 完成本阶段后必须具备的能力

1. **张量能力**：不靠试错判断常见张量操作的 shape、dtype 和 device。
2. **反传能力**：能从链式法则解释反向传播，并实现一个标量自动求导引擎。
3. **框架能力**：熟练使用 `Tensor`、`autograd`、`nn.Module`、
   `Dataset/DataLoader`、损失函数、优化器和 `state_dict`。
4. **训练能力**：不看模板写出训练、验证、保存、加载与推理流程。
5. **诊断能力**：能发现并修复 shape/device/dtype、梯度累加、数值不稳定、
   过拟合和 `train()/eval()` 使用错误。
6. **语言模型直觉**：理解“根据前文预测下一个字符”如何变成数据、logits、
   交叉熵、梯度更新和采样。

### 2.2 本阶段刻意不学

- CNN、图像增广和完整计算机视觉路线
- RNN/LSTM/GRU
- Self-Attention、Transformer、位置编码和 LayerNorm 的细节
- 混合精度、分布式训练和性能优化
- Hugging Face 高层 API

这些内容不会丢失；Transformer 会在阶段 2 单独深入。当前原则是：
**只学进入 Transformer 前真正需要的深度学习地基。**

---

## 3. 每个 Step 的固定学习协议

每个 Step 使用一个全新会话，延续之前已经验证有效的方式：

1. **前测（5–10 分钟）**：先回答 3–5 个预测题，暴露真实盲点。
2. **中文讲义**：先直觉，再公式，再对应到 PyTorch 代码与 LLM 场景。
3. **动手实验**：先预测结果和 shape，再亲手实现，最后用断言验证。
4. **少而必要的自测**：固定为 5 道综合题，不堆大量机械小问。
5. **批改与升级**：保留原答案，写回评分和订正，达到标准后更新掌握度。

每步默认产出：

```text
notes/05-深度学习与PyTorch/S1.x-主题.md
notes/05-深度学习与PyTorch/S1.x-自测题.md
examples/stage1/s1_x_topic.py
```

最终项目单独放入：

```text
projects/makemore/
```

### 过关规则

- 自测至少 **85/100**
- 实验脚本可独立运行，关键结果有断言
- 能解释关键张量的每一维
- 能回答“为什么这样写”，而不是只复述 API
- 未过关时只修补当前 Step，不提前进入下一 Step

---

## 4. 三周总览

| 周 | Step | 主题 | 建议时长 | 可验证产出 | 状态 | 自测 | 完成日期 |
|----|------|------|:--------:|------------|:----:|:----:|:--------:|
| 1 | 1.1 | PyTorch 环境、Tensor、dtype 与 device | 2–2.5h | 张量与 MPS/CPU 实验 | ⬜ | — | — |
| 1 | 1.2 | 从链式法则手写 micrograd | 3–4h | 标量自动求导引擎 | ⬜ | — | — |
| 1 | 1.3 | PyTorch autograd 与梯度语义 | 2–2.5h | micrograd / PyTorch 梯度对照 | ⬜ | — | — |
| 1 | 1.4 | `nn.Module`、参数、层与 MLP | 2–2.5h | 两种写法结果一致的 MLP | ⬜ | — | — |
| 2 | 1.5 | `Dataset`、`DataLoader` 与 mini-batch | 2h | 可复现的数据管道 | ⬜ | — | — |
| 2 | 1.6 | logits、损失函数与完整训练循环 | 2.5–3h | 从零训练分类 MLP | ⬜ | — | — |
| 2 | 1.7 | SGD、Momentum、AdamW 与学习率 | 2–2.5h | 优化器对照实验 | ⬜ | — | — |
| 2 | 1.8 | 泛化、正则化、稳定性与训练诊断 | 3h | 故障注入与修复实验 | ⬜ | — | — |
| 3 | 1.9 | 字符数据集与计数 bigram | 2–2.5h | 非神经 bigram 基线 | ⬜ | — | — |
| 3 | 1.10 | 可训练的神经 bigram | 2.5h | `V × V` 参数模型与采样 | ⬜ | — | — |
| 3 | 1.11 | Embedding + MLP 字符语言模型 | 3–4h | 上下文窗口语言模型 | ⬜ | — | — |
| 3 | 1.12 | 重构、验收与独立复现 | 4h | 完整 makemore 项目 | ⬜ | — | — |

建议每周安排 4 次正式学习，每次只完成一个 Step；Step 1.2、1.11 和 1.12
允许拆成两晚，但仍在同一个会话中完成。

---

## 5. Step-by-Step 详细任务

### Step 1.1 · PyTorch 环境、Tensor、dtype 与 device

**目标**：把已经掌握的 NumPy 张量知识迁移到 PyTorch，并建立设备意识。

**必须掌握**

- 安全准备 Python 3.11/3.12 与 PyTorch 环境
- `shape`、`ndim`、`dtype`、`device`、标量张量
- 创建、索引、切片、`reshape`、`unsqueeze/squeeze`、转置、广播、`matmul`
- NumPy 与 PyTorch 的相同点和不同点
- 随机种子、CPU 与 Apple MPS 的选择和结果对照

**必须完成**

1. 运行环境诊断，记录 Python、PyTorch 和 MPS 状态。
2. 把 5 段已有 NumPy 示例改写为 PyTorch。
3. 先手写预测 10 个操作的 shape，再运行验证。
4. 写一个不硬编码设备的 `get_device()`，同一计算可在 CPU/MPS 执行。

**过关标准**

- 10 道 shape 预测至少答对 8 道
- 能解释为什么索引张量通常使用 `torch.long`
- 能独立修复一次 dtype 或 device 不一致错误

**主资料**

- [D2L · 数据操作](https://zh.d2l.ai/chapter_preliminaries/ndarray.html)
- [李沐 Bilibili · 数据操作](https://www.bilibili.com/video/BV18p4y1h7Dr)
- [PyTorch · Tensors](https://docs.pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html)

### Step 1.2 · 从链式法则手写 micrograd

**目标**：不依赖 PyTorch，用约 100 行核心代码把链式法则变成反向传播。

**必须掌握**

- 动态计算图、叶子节点、局部导数和反向拓扑序
- 路径内梯度相乘、路径间梯度累加
- `Value.data`、`Value.grad`、`_prev`、`_backward`
- 加法、乘法、幂、`tanh` 或 ReLU 的反传
- 节点被重复使用时为什么必须使用 `+=`

**必须完成**

1. 亲手实现最小 `Value` 类，不复制完整成品。
2. 对一个两层标量表达式执行前向和反向传播。
3. 用有限差分做梯度检查。
4. 用同一表达式的 PyTorch 结果做第二重验证。

**过关标准**

- 梯度检查误差小于 `1e-5`
- 能解释拓扑排序为什么必须在反向传播前完成
- 能现场指出“覆盖梯度而非累加梯度”的 bug

**主资料**

- [Karpathy · micrograd 源码](https://github.com/karpathy/micrograd)
- [Bilibili 中英字幕 · micrograd 与反向传播](https://www.bilibili.com/video/BV1aB4y13761)
- [Karpathy 原视频](https://www.youtube.com/watch?v=VMj-3S1tku0)

### Step 1.3 · PyTorch autograd 与梯度语义

**目标**：把自制引擎映射到 PyTorch，掌握最容易写错的梯度生命周期。

**必须掌握**

- `requires_grad`、叶子张量、`grad_fn`、`.grad`
- 标量与非标量输出的 `backward`
- 梯度默认累加及 `zero_grad(set_to_none=True)`
- `detach()`、`torch.no_grad()`、`torch.inference_mode()`
- 原地操作为何可能破坏计算图

**必须完成**

1. 用 micrograd 和 PyTorch 实现同一个神经元并逐项比对梯度。
2. 连续调用两次 `backward()`，观察并解释梯度累加。
3. 分别演示训练、验证和推理时的梯度上下文。
4. 故意制造一次非叶子张量 `.grad` 误解并修正。

**过关标准**

- 能不看资料说出 `zero_grad → forward → loss → backward → step`
- 能解释 `detach` 与 `no_grad` 的区别
- 能说清为什么验证阶段不应构建反向图

**主资料**

- [D2L · 自动微分](https://zh.d2l.ai/chapter_preliminaries/autograd.html)
- [李沐 Bilibili · 自动求导](https://www.bilibili.com/video/BV1KA411N7Px)
- [PyTorch · Automatic Differentiation](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html)

### Step 1.4 · `nn.Module`、参数、层与 MLP

**目标**：理解 PyTorch 模型封装做了什么，而不是把 `nn.Module` 当黑盒。

**必须掌握**

- `nn.Module`、`forward`、`Parameter`、参数注册与 `state_dict`
- `nn.Linear` 对应的矩阵形状与 bias 广播
- ReLU、tanh、GELU 的作用和基本差异
- `nn.Sequential`、子模块、参数量计算
- 初始化对前向数值和梯度的影响

**必须完成**

1. 用裸张量矩阵运算写一个两层 MLP。
2. 用 `nn.Module` 写同一个 MLP，并同步权重。
3. 验证两种实现的 logits 与参数量完全一致。
4. 打印每层输入输出 shape 和参数名称。

**过关标准**

- 能手算每个 `Linear` 的参数量
- 能解释普通 Tensor 为什么不会自动成为模型参数
- 能根据输入 batch 写出每层 shape

**主资料**

- [D2L · 多层感知机](https://zh.d2l.ai/chapter_multilayer-perceptrons/mlp.html)
- [李沐 Bilibili · 多层感知机](https://www.bilibili.com/video/BV1hh411U7gn)
- [D2L · 模型构造](https://zh.d2l.ai/chapter_deep-learning-computation/model-construction.html)
- [PyTorch · Build Model](https://docs.pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html)

### Step 1.5 · `Dataset`、`DataLoader` 与 mini-batch

**目标**：理解训练数据如何变成稳定、可复现的批次张量。

**必须掌握**

- 样本、特征、标签与 batch 维
- 自定义 `Dataset` 的 `__len__` / `__getitem__`
- `DataLoader` 的 `batch_size`、`shuffle`、`drop_last`
- 训练/验证/测试划分与数据泄漏
- mini-batch 对梯度估计和内存的影响

**必须完成**

1. 只用 PyTorch 生成一个带噪声的二分类数据集。
2. 实现自定义 `Dataset` 和训练/验证 DataLoader。
3. 检查一个 epoch 没有无故丢失或重复样本。
4. 固定随机种子后验证数据划分可复现。

**过关标准**

- 能说清 `X.shape == (B, D)`、`y.shape == (B,)`
- 能解释为什么只对训练集 shuffle
- 能识别一次数据泄漏

**主资料**

- [PyTorch · Datasets & DataLoaders](https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html)
- [小土堆 Bilibili · P15 DataLoader、P16 nn.Module](https://www.bilibili.com/video/BV1hE411t7RN?p=15)

### Step 1.6 · logits、损失函数与完整训练循环

**目标**：把模型输出、MLE、交叉熵和参数更新串成一个闭环。

**必须掌握**

- logits、概率、预测类别三者的区别
- `CrossEntropyLoss = log_softmax + NLLLoss`
- 为什么传入 `CrossEntropyLoss` 前不手动 softmax
- batch 平均损失与梯度
- 训练循环每一步的职责

**必须完成**

1. 用 `logsumexp` 写数值稳定的交叉熵，并与 `F.cross_entropy` 对齐。
2. 为 Step 1.5 的数据训练两层 MLP。
3. 先让模型过拟合一个 batch，再训练完整训练集。
4. 同时记录 loss、accuracy 和梯度范数。

**过关标准**

- 手写交叉熵与官方实现误差小于 `1e-6`
- 单 batch loss 能稳定降到接近 0
- 能完整解释一次参数更新的数据流

**主资料**

- [李沐 Bilibili · Softmax 回归与损失函数](https://www.bilibili.com/video/BV1K64y1Q7wu)
- [PyTorch · Optimization Loop](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)

### Step 1.7 · SGD、Momentum、AdamW 与学习率

**目标**：知道优化器如何影响训练，而不是默认永远使用 Adam。

**必须掌握**

- full-batch、mini-batch 与 stochastic gradient
- SGD、Momentum、Adam/AdamW 的核心直觉
- AdamW 中解耦 weight decay 的意义
- 学习率过大、过小的可观察症状
- 公平对照实验：同一初始化、同一批次顺序

**必须完成**

1. 在同一模型上比较 SGD、Momentum、AdamW。
2. 比较至少 3 个数量级的学习率。
3. 记录收敛速度、最终验证损失和稳定性。
4. 为当前任务写出有证据的优化器选择结论。

**过关标准**

- 能从 loss 曲线或日志判断学习率过大/过小
- 能解释 AdamW 不是“必然更好”
- 至少一个配置稳定收敛且可复现

**主资料**

- [D2L · 优化算法](https://zh.d2l.ai/chapter_optimization/index.html)
- [李沐 Bilibili · 基础优化方法](https://www.bilibili.com/video/BV1PX4y1g7KC?p=2)

### Step 1.8 · 泛化、正则化、稳定性与训练诊断

**目标**：建立一套“模型不学或学坏时怎么查”的固定顺序。

**必须掌握**

- 欠拟合、过拟合、训练集/验证集差距
- weight decay、dropout 与早停的作用边界
- `train()` / `eval()` 对 dropout 等模块的影响
- Xavier/Kaiming 初始化、激活分布、梯度消失/爆炸
- 梯度范数与 `clip_grad_norm_`

**必须完成**

1. 故意构造欠拟合和过拟合，并解释证据。
2. 对比无正则、weight decay、dropout 三个配置。
3. 记录各层激活均值/标准差与梯度范数。
4. 注入并修复至少 3 个训练 bug：未清梯度、错误模式、错误 dtype/device、
   不稳定学习率中任选 3 个。

**过关标准**

- 面对“loss 不下降”能按固定清单排查
- 能解释正则化为何通常提高验证表现而非训练表现
- 能把 2.10 梯度消失/爆炸从 `🟡` 复核到可解释状态

**主资料**

- [D2L · 欠拟合与过拟合](https://zh.d2l.ai/chapter_multilayer-perceptrons/underfit-overfit.html)
- [李沐 Bilibili · 模型选择与过拟合](https://www.bilibili.com/video/BV1kX4y1g7jp)
- [李沐 Bilibili · 权重衰减](https://www.bilibili.com/video/BV1UK4y1o7dy)
- [李沐 Bilibili · Dropout](https://www.bilibili.com/video/BV1Y5411c7aY)
- [李沐 Bilibili · 数值稳定性与初始化](https://www.bilibili.com/video/BV1u64y1i75a)

### Step 1.9 · 字符数据集与计数 bigram

**目标**：在没有神经网络的情况下走通最小语言模型的训练、评估和采样。

**必须掌握**

- 字符词表、特殊边界 token、`stoi/itos`
- `P(next_char | current_char)` 的计数估计
- 训练/验证划分、平滑、NLL 与困惑度
- 自回归采样和随机种子
- 均匀分布、unigram、bigram 三种基线

**必须完成**

1. 从 `names.txt` 构建字符词表和 bigram 计数矩阵。
2. 把计数归一化为条件概率并验证每行和为 1。
3. 计算训练集和验证集 NLL。
4. 从模型采样至少 20 个名字并检查可复现性。

**过关标准**

- 能解释每个训练样本为什么是一对字符
- 能手算一个名字的 NLL
- bigram 验证 NLL 优于均匀分布基线

**主资料**

- [Karpathy · makemore](https://github.com/karpathy/makemore)
- [Bilibili 中文讲解 · makemore Part 1](https://www.bilibili.com/video/BV134RwYvEEQ)

### Step 1.10 · 可训练的神经 bigram

**目标**：把计数模型改写为可微参数模型，第一次完整训练“下一个 token 预测”。

**必须掌握**

- `W.shape == (vocab_size, vocab_size)` 的含义
- one-hot 与直接行索引的等价关系
- logits、交叉熵和条件概率
- L2 正则化与计数平滑的联系
- 训练分布与采样分布

**必须完成**

1. 用参数矩阵 `W` 实现神经 bigram。
2. 同时实现 one-hot 版和索引版，验证 logits 一致。
3. 使用 autograd 和交叉熵训练。
4. 比较神经模型与计数模型的概率矩阵和验证 NLL。

**过关标准**

- 神经 bigram 验证 NLL 优于均匀基线
- 能解释计数模型与梯度训练为何会得到相近分布
- 生成循环全程处于 `no_grad` 或 `inference_mode`

**主资料**

- [Karpathy · makemore Part 1 notebook](https://github.com/karpathy/nn-zero-to-hero/blob/master/lectures/makemore/makemore_part1_bigrams.ipynb)
- [Karpathy 中英字幕合集 · P2](https://www.bilibili.com/video/BV1mqrTBvEaf?p=2)

### Step 1.11 · Embedding + MLP 字符语言模型

**目标**：使用多个前文字符预测下一个字符，完成 Transformer 前的核心语言模型。

**必须掌握**

- context window / block size
- `Embedding(vocab_size, embedding_dim)` 的查表含义
- `(B, T) → (B, T, C) → (B, T*C) → (B, H) → (B, V)`
- mini-batch 训练、训练/验证/测试划分
- 参数量、容量、过拟合和超参数对结果的影响

**必须完成**

1. 构建 context-target 数据集。
2. 实现 `Embedding + Flatten + Linear + activation + Linear`。
3. 训练并保存最优验证模型。
4. 调整 embedding 维度、hidden size 或学习率，至少做 3 次受控实验。
5. 采样并解释 temperature 对结果的影响。

**过关标准**

- 能逐层说出所有张量 shape
- 最佳验证 NLL 至少比 bigram 基线低 `0.1`
- 保存再加载后，同一输入 logits 完全一致

**主资料**

- [Bilibili 中英精校 · makemore MLP](https://www.bilibili.com/video/BV1Yz2SYpE2u)
- [Karpathy · makemore Part 2 notebook](https://github.com/karpathy/nn-zero-to-hero/blob/master/lectures/makemore/makemore_part2_mlp.ipynb)

### Step 1.12 · 重构、验收与独立复现

**目标**：脱离教程，把阶段知识整合成可运行、可解释、可复现的小项目。

**必须完成**

1. 不复制前一步成品，按空白骨架重建数据、模型、训练、验证和生成流程。
2. 把代码拆为清楚的配置、数据、模型、训练、评估和采样职责。
3. 支持随机种子、CPU/MPS、checkpoint 保存与恢复。
4. 加入 shape、词表、概率和 checkpoint 一致性检查。
5. 写实验结论：基线、最佳配置、失败配置、生成样例和下一步限制。

**最终验收**

| 维度 | 分值 | 要求 |
|------|:----:|------|
| 原理解释 | 20 | 讲清前向、损失、反向、更新和采样 |
| 张量与代码 | 20 | shape 全部正确，无硬编码 device |
| 训练与泛化 | 20 | MLP 验证 NLL 明显优于 bigram |
| 工程完整性 | 20 | 可复现、可保存加载、关键断言齐全 |
| 综合自测 | 20 | 5 道综合题，能纠正常见误区 |

**通过线：85/100，且“张量与代码”“训练与泛化”不得低于 15 分。**

通过后再进入阶段 2，不在本项目中提前加入 Attention。

---

## 6. 资料使用策略

| 资料 | 定位 | 使用方式 |
|------|------|----------|
| [李沐 D2L 官方课程](https://courses.d2l.ai/zh-v2/) | **中文主线** | 只看本计划链接的基础章节，不刷完整 CNN 路线 |
| [D2L 中文版](https://zh.d2l.ai/) | 讲义与代码 | 课前预习或课后查漏，不逐字通读 |
| [PyTorch Learn the Basics](https://docs.pytorch.org/tutorials/beginner/basics/intro.html) | API 权威来源 | D2L 代码与当前 API 不一致时，以此为准 |
| [邱锡鹏《神经网络与深度学习》第二版](https://nndl.ai/nndl-v2/) | 中文理论字典 | 查前馈网络、优化与正则化，不要求通读 |
| [Karpathy Zero to Hero](https://github.com/karpathy/nn-zero-to-hero) | 原理与项目 | Step 1.2、1.9–1.11 跟代码，不被动抄写 |
| [Karpathy Bilibili 中英字幕合集](https://www.bilibili.com/video/BV1mqrTBvEaf) | 中文字幕辅助 | 本阶段只看 P1–P4，P7 以后留给阶段 2–3 |
| [小土堆 PyTorch 教程](https://www.bilibili.com/video/BV1hE411t7RN) | API 操作补充 | 选看 P15–16、P20–24、P26–29、P32 |

### 关于小土堆教程

该课程讲解直观，但主体录制于 2019–2021 年。`Dataset/DataLoader`、`nn.Module`、
损失函数、优化器和保存加载仍适合作为补充；环境安装、CUDA 和完整 CV 部分不作为
本阶段主线。当前机器是 Apple Silicon，设备代码应学习 MPS/CPU 通用写法。

### 视频学习规则

- 先看本 Step 的目标和自测，再看视频
- 每 15–25 分钟暂停，自己重写关键代码
- 视频代码必须经过官方文档校准
- Bilibili 字幕版可能变动；代码与事实以原作者仓库和官方文档为准
- “看完”不计完成，只有实验与自测通过才更新状态

---

## 7. 总进度追踪

| 子阶段 | 总数 | ⬜未开始 | 🟡学习中/模糊 | 🟢独立复现 | ✅能讲 | 完成率 |
|--------|:----:|:--------:|:-------------:|:--------:|:-----:|:------:|
| A · PyTorch 与反传地基 | 4 | 4 | 0 | 0 | 0 | 0% |
| B · 训练与泛化 | 4 | 4 | 0 | 0 | 0 | 0% |
| C · 字符级语言模型 | 4 | 4 | 0 | 0 | 0 | 0% |
| **合计** | **12** | **12** | **0** | **0** | **0** | **0%** |

更新规则：

- 材料和实验刚完成：`🟡`
- 自测 ≥ 85 且能独立重写核心代码：`🟢`
- 隔一周仍能不看资料讲解和复现：`✅`
- 每完成一步，同时更新第 4 节对应 Step、上方汇总表和相关数学回补状态

---

## 8. 新会话启动模板

每次只把 Step 编号和主题替换掉：

```text
我正在执行《阶段1-深度学习与PyTorch学习计划.md》的
Step 1.x「这里填写主题」。

请先参考：
1. 大模型学习路线.md
2. 阶段1-深度学习与PyTorch学习计划.md
3. 数学知识点掌握表.md
4. 仓库中之前的笔记、实验、自测和相关历史会话

请沿用之前的学习方式，带我完成当前 Step：
前测 → 中文讲义 → 可运行实验 → 5 道综合自测 → 批改 → 更新掌握度。

要求：
- 先让我预测结果和张量 shape，再运行代码验证
- 讲清它与 LLM 的联系
- 不提前进入下一个 Step
- 产物按计划写入 notes/ 和 examples/
```

下一次新会话从 **Step 1.1 · PyTorch 环境、Tensor、dtype 与 device** 开始。
