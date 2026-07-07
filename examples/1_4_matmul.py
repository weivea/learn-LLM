"""
1.4 矩阵乘法(matmul) —— 验证脚本

配套知识点：数学知识点掌握表.md · 1.4
目标：理解 (m, k) @ (k, n) -> (m, n)，能手算 2x3 @ 3x2，
      并把它和 Transformer 里的 Q @ K.T、weights @ V 对上。

运行：
    python examples/1_4_matmul.py
"""

import numpy as np


def section(title: str) -> None:
    print("\n" + "=" * 58)
    print(title)
    print("=" * 58)


def stable_softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / exp_x.sum(axis=-1, keepdims=True)


def main() -> None:
    # ---------------------------------------------------------------
    section("1) 形状规则：内相等，取外围")
    A = np.array([[1, 2, 3],
                  [4, 5, 6]])
    B = np.array([[7, 8],
                  [9, 10],
                  [11, 12]])
    print("A.shape =", A.shape)
    print("B.shape =", B.shape)
    print("A @ B   ->", (A @ B).shape, "因为 (2,3) @ (3,2) => (2,2)")
    print("判断能否相乘只看：A.shape[1] == B.shape[0] ->", A.shape[1] == B.shape[0])

    C_bad_left = np.zeros((3, 4))
    C_bad_right = np.zeros((2, 5))
    print("\n反例：", C_bad_left.shape, "@", C_bad_right.shape,
          "不能乘，因为 4 != 2")

    # ---------------------------------------------------------------
    section("2) 手算 2x3 @ 3x2：一行乘一列，填一个格子")
    print("A =\n", A)
    print("B =\n", B)
    print("c00 = 1*7 + 2*9 + 3*11 =", 1 * 7 + 2 * 9 + 3 * 11)
    print("c01 = 1*8 + 2*10 + 3*12 =", 1 * 8 + 2 * 10 + 3 * 12)
    print("c10 = 4*7 + 5*9 + 6*11 =", 4 * 7 + 5 * 9 + 6 * 11)
    print("c11 = 4*8 + 5*10 + 6*12 =", 4 * 8 + 5 * 10 + 6 * 12)
    C = A @ B
    print("A @ B =\n", C)

    # ---------------------------------------------------------------
    section("3) @ 和 * 不是一回事")
    X = np.array([[1, 2],
                  [3, 4]])
    Y = np.array([[10, 20],
                  [30, 40]])
    print("X =\n", X)
    print("Y =\n", Y)
    print("X * Y =\n", X * Y)
    print("X @ Y =\n", X @ Y)
    print("=> * 是逐元素乘；@ 是矩阵乘法")

    # ---------------------------------------------------------------
    section("4) 一般不满足交换律：A @ B 和 B @ A 不一样")
    print("A @ B 的 shape =", (A @ B).shape)
    print(A @ B)
    print("\nB @ A 的 shape =", (B @ A).shape)
    print(B @ A)
    print("=> 不仅值不同，shape 都可能不同")

    # ---------------------------------------------------------------
    section("5) 向量 / 行矩阵 / 列矩阵要分清")
    v = np.array([1, 2, 3])           # (3,)
    row = v.reshape(1, 3)             # (1, 3)
    col = v.reshape(3, 1)             # (3, 1)
    print("v.shape   =", v.shape, " -> 一维向量")
    print("row.shape =", row.shape, " -> 1 行 3 列")
    print("col.shape =", col.shape, " -> 3 行 1 列")
    print("row @ col ->", (row @ col).shape, "值 =", row @ col)
    print("col @ row ->", (col @ row).shape)
    print(col @ row)

    # ---------------------------------------------------------------
    section("6) 线性层：一句话矩阵乘权重矩阵")
    seq_len, d_model, d_out = 4, 3, 5
    hidden = np.arange(seq_len * d_model).reshape(seq_len, d_model)
    W = np.ones((d_model, d_out))
    out = hidden @ W
    print("hidden.shape =", hidden.shape, " -> (seq_len, d_model)")
    print("W.shape      =", W.shape, " -> (d_model, d_out)")
    print("hidden @ W   =", out.shape, " -> (seq_len, d_out)")

    # ---------------------------------------------------------------
    section("7) 注意力直觉：Q @ K.T 先算相关度，weights @ V 再搬内容")
    tokens = ["我", "爱", "猫"]
    Q = np.array([[1., 0.],
                  [0., 1.],
                  [1., 1.]])
    K = np.array([[1., 0.],
                  [0., 1.],
                  [1., 1.]])
    V = np.array([[10., 0.],
                  [0., 10.],
                  [5., 5.]])

    scores = Q @ K.T
    weights = stable_softmax(scores)
    context = weights @ V

    print("Q.shape =", Q.shape, "K.shape =", K.shape, "V.shape =", V.shape)
    print("scores = Q @ K.T ->", scores.shape)
    print(scores)
    print("\n对 scores 每一行做 softmax，得到注意力权重：")
    print(np.round(weights, 3))
    print("每一行加起来 =", np.round(weights.sum(axis=1), 3))
    print("\ncontext = weights @ V ->", context.shape)
    print(np.round(context, 3))
    print()
    for token, weight_row, ctx in zip(tokens, weights, context):
        print(f"{token} 这一行权重 = {np.round(weight_row, 3)} -> 新表示 {np.round(ctx, 3)}")
    print("=> 先决定看谁（Q @ K.T），再决定看多少（softmax），最后拿内容（@ V）")

    print("\n✅ 全部跑通！你已经能看懂 matmul 的 shape 规则和注意力里的两次矩阵乘法。")


if __name__ == "__main__":
    main()
