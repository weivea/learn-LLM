"""
1.6 矩阵转置(transpose) —— 验证脚本

配套知识点：数学知识点掌握表.md · 1.6
目标：看清「转置=行列对调、数值不变」，掌握四条性质，
      并搞懂注意力里 Q @ K.T 的 .T 到底解决了什么维度问题。

运行：
    python examples/1_6_transpose.py
"""

import numpy as np


def section(title: str) -> None:
    print("\n" + "=" * 58)
    print(title)
    print("=" * 58)


def main() -> None:
    # ---------------------------------------------------------------
    section("1) 定义：行变列、列变行，A[i,j] -> A.T[j,i]")
    A = np.array([[1, 2, 3],
                  [4, 5, 6]])
    print("A =\n", A, "  shape =", A.shape)
    print("A.T =\n", A.T, "  shape =", A.T.shape, " <- (2,3) 变 (3,2)")
    print("原 A[0,2] =", A[0, 2], " 跑到 A.T[2,0] =", A.T[2, 0], " <- 数值没变，只换位置")

    # ---------------------------------------------------------------
    section("2) 性质：转两次回到原样  (A.T).T == A")
    print("(A.T).T =\n", A.T.T)
    print("和 A 相等吗？", np.array_equal(A.T.T, A))

    # ---------------------------------------------------------------
    section("3) 性质：(AB).T == B.T A.T（乘法要换序！）")
    B = np.array([[7, 8],
                  [9, 10],
                  [11, 12]])          # (3,2)
    AB = A @ B                        # (2,3)@(3,2) = (2,2)
    left = AB.T                       # (2,2)
    right = B.T @ A.T                 # (2,3)@(3,2) = (2,2)
    print("(AB).T =\n", left)
    print("B.T @ A.T =\n", right)
    print("(AB).T == B.T A.T ?", np.array_equal(left, right))
    print("那 A.T @ B.T 行不行？ A.T=(3,2), B.T=(2,3) -> (3,3)，形状都对不上：")
    print("A.T @ B.T 的 shape =", (A.T @ B.T).shape, " <- 和 (2,2) 完全不同")

    # ---------------------------------------------------------------
    section("4) 对称矩阵：A.T == A")
    S = np.array([[1, 2, 3],
                  [2, 5, 6],
                  [3, 6, 9]])
    print("S =\n", S)
    print("S 是对称矩阵吗？(S.T == S)", np.array_equal(S.T, S))

    # ---------------------------------------------------------------
    section("5) 坑一：一维向量 .T 无效（没有行列概念）")
    v = np.array([1, 2, 3])
    print("v      =", v, " shape =", v.shape)
    print("v.T    =", v.T, " shape =", v.T.shape, " <- 完全没变！")
    col = v.reshape(-1, 1)
    print("想要列向量得 reshape(-1,1)：\n", col, " shape =", col.shape)

    # ---------------------------------------------------------------
    section("6) 坑二：带 batch 别用 .T，要用 swapaxes(-1,-2)")
    x = np.arange(2 * 3 * 4).reshape(2, 3, 4)   # (batch=2, seq=3, d=4)
    print("x.shape           =", x.shape)
    print("x.T.shape         =", x.T.shape, " <- 所有轴倒过来，batch 都乱了！")
    print("x.swapaxes(-1,-2) =", x.swapaxes(-1, -2).shape,
          " <- 只翻最后两维，正确")
    print("x.transpose(0,2,1)=", x.transpose(0, 2, 1).shape, " <- 等价写法")

    # ---------------------------------------------------------------
    section("7) 注意力：Q @ K.T 的 .T 是为了对齐维度")
    seq, d_k = 3, 4
    rng = np.random.default_rng(0)
    Q = rng.standard_normal((seq, d_k))   # (3,4) 每行一个 query
    K = rng.standard_normal((seq, d_k))   # (3,4) 每行一个 key
    print("Q.shape =", Q.shape, " K.shape =", K.shape)
    print("直接 Q @ K 行吗？内层维度 4 vs 3 不等 -> 报错。所以要转置 K：")
    scores = Q @ K.T                      # (3,4)@(4,3) = (3,3)
    print("K.T.shape =", K.T.shape, " -> Q @ K.T 的 shape =", scores.shape,
          " <- (seq, seq) 注意力分数矩阵")
    print("其中 scores[0,1] = Q第0行 · K第1行 =", np.dot(Q[0], K[1]),
          " <- 本质是点积(1.5)")

    # ---------------------------------------------------------------
    section("8) 坑三：转置是视图(view)，和原数组共享内存")
    M = np.array([[1, 2],
                  [3, 4]])
    Mt = M.T
    Mt[0, 0] = 99          # 改 M.T
    print("改了 M.T[0,0]=99 后，原 M =\n", M, " <- M[0,0] 也变成 99 了！")
    print("需要独立副本请用 M.T.copy()")

    print("\n✅ 全部跑通！你已经理解转置=行列对调，以及注意力里 .T 的作用。")


if __name__ == "__main__":
    main()
