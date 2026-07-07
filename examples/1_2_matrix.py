"""
1.2 矩阵的概念与形状(shape) —— 验证脚本

配套知识点：数学知识点掌握表.md · 1.2
目标：看到 (m, n) 能说清「几行几列」，理解矩阵 = 一堆向量叠起来。

运行：
    python examples/1_2_matrix.py
"""

import numpy as np


def section(title: str) -> None:
    print("\n" + "=" * 50)
    print(title)
    print("=" * 50)


def main() -> None:
    # ---------------------------------------------------------------
    section("1) 矩阵 = 数排成的矩形网格")
    A = np.array([
        [1, 2, 3],
        [4, 5, 6],
    ])
    print("A =\n", A)
    print("A.shape =", A.shape, " -> (2, 3)：2 行 3 列")
    print("A.ndim  =", A.ndim, "   -> 二维，所以是矩阵")

    # ---------------------------------------------------------------
    section("2) shape 先行后列：(m, n) = (行, 列)")
    print("行数 m = A.shape[0] =", A.shape[0], " (axis=0)")
    print("列数 n = A.shape[1] =", A.shape[1], " (axis=1)")
    print("元素总数 A.size =", A.size, " = m * n")

    # ---------------------------------------------------------------
    section("3) 按 [行, 列] 取元素（从 0 数起）")
    print("A[0, 0] =", A[0, 0], " -> 第 0 行第 0 列")
    print("A[1, 2] =", A[1, 2], " -> 第 1 行第 2 列")
    print("A[0]    =", A[0], " -> 第 0 行（一个行向量）")
    print("A[:, 1] =", A[:, 1], " -> 第 1 列（一个列，取出来是 1 维）")
    print("A[0, :] =", A[0, :], " -> 第 0 行（一个行向量）")

    # ---------------------------------------------------------------
    section("4) 矩阵 = 多个向量叠起来")
    r0 = np.array([1, 2, 3])
    r1 = np.array([4, 5, 6])
    stacked = np.stack([r0, r1])           # 把两个行向量叠成矩阵
    print("stack([r0, r1]) =\n", stacked)
    print("和 A 相等吗?", np.array_equal(stacked, A))

    # ---------------------------------------------------------------
    section("5) 易错点：(3,) 向量 vs (1,3) 矩阵")
    v = np.array([1, 2, 3])
    row = v.reshape(1, 3)
    print("v.shape        =", v.shape, "   ndim =", v.ndim, " -> 向量(一维)")
    print("row.shape      =", row.shape, " ndim =", row.ndim, " -> 矩阵(二维)")
    print("值一样但维度不同，写代码时别混！")
    print("v =", v)
    print("row =", row)

    # ---------------------------------------------------------------
    section("6) 直觉：LLM 里常见矩阵的 shape 含义")
    vocab_size, d_model, seq_len = 50000, 768, 128
    embedding = np.zeros((vocab_size, d_model))
    sentence = np.zeros((seq_len, d_model))
    print(f"词嵌入表 embedding.shape = {embedding.shape} -> {vocab_size} 个词，每个 {d_model} 维")
    print(f"一句话   sentence.shape  = {sentence.shape}  -> {seq_len} 个 token，每个 {d_model} 维")
    print("=> 一句话就是一叠词向量 = 一个矩阵")

    print("\n✅ 全部跑通！你已经能看着 shape 说清『几行几列』。")


if __name__ == "__main__":
    main()
