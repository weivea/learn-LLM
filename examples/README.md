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
