# examples · 验证脚本

配套 [数学知识点掌握表.md](../数学知识点掌握表.md)，每个知识点一个可运行脚本，
用代码把公式「跑一遍」建立直觉。数学**够用就走**，理解在敲代码时补齐。

## 环境准备（只需一次）

```bash
# 在项目根目录
python3 -m venv .venv          # 创建虚拟环境
source .venv/bin/activate      # 激活（Windows: .venv\Scripts\activate）
pip install -r requirements.txt
```

## 运行

```bash
# 激活环境后
python examples/1_1_vectors.py

# 或不激活，直接用 venv 里的解释器
.venv/bin/python examples/1_1_vectors.py
```

## 脚本清单

| 脚本 | 对应知识点 |
|------|-----------|
| `1_1_vectors.py` | 1.1 向量的概念与表示 |
| `1_2_matrix.py` | 1.2 矩阵的概念与形状 |
| `1_3_tensor.py` | 1.3 张量 / 维度(dim, axis) |
| `1_4_matmul.py` | 1.4 矩阵乘法(matmul) |
| `1_5_dot_product.py` | 1.5 点积(dot product) |
| `1_6_transpose.py` | 1.6 矩阵转置(transpose) |
| `1_7_broadcasting.py` | 1.7 广播(broadcasting) |
| `1_8_cosine_similarity.py` | 1.8 余弦相似度(cosine similarity) |
| `1_9_norm.py` | 1.9 范数(norm) L1 / L2 |
| `1_10_identity_inverse.py` | 1.10 单位矩阵 / 逆矩阵 |
| `1_11_eigen.py` | 1.11 特征值 / 特征向量 |
| `1_12_svd.py` | 1.12 SVD 奇异值分解 |
| `2_1_derivative.py` | 2.1 导数(derivative) |
| `2_2_partial_derivative.py` | 2.2 偏导数(partial derivative) |
| `2_3_gradient.py` | 2.3 梯度(gradient) |
| `2_4_chain_rule.py` | 2.4 链式法则(chain rule) |
| `2_5_gradient_descent.py` | 2.5 梯度下降(gradient descent) |
| `2_6_learning_rate.py` | 2.6 学习率(learning rate η) |
| `2_7_autograd.py` | 2.7 计算图与自动求导 |
| `2_8_common_derivatives.py` | 2.8 常见函数的导数 |
| `2_9_local_minima_saddle_points.py` | 2.9 局部最小 / 鞍点 |
| `2_10_vanishing_exploding_gradients.py` | 2.10 梯度消失 / 爆炸 |
| `3_1_probability_distributions.py` | 3.1 概率分布(离散 / 连续) |
| `3_2_softmax.py` | 3.2 Softmax |
| `3_3_cross_entropy_nll.py` | 3.3 交叉熵 / 负对数似然(NLL) |
| `3_4_logarithms.py` | 3.4 对数(log) |
| `3_5_perplexity.py` | 3.5 困惑度(perplexity) |
| `3_6_conditional_probability.py` | 3.6 条件概率：P(下一个 token \| 前文) |
| `3_7_expectation.py` | 3.7 期望(expectation) |
| `3_8_variance_standard_deviation.py` | 3.8 方差 / 标准差 |
| `3_9_sampling.py` | 3.9 采样(sampling) |
