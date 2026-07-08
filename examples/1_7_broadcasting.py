"""
1.7 广播(broadcasting) —— 验证脚本

配套知识点：数学知识点掌握表.md · 1.7
目标：看清「广播=形状不同的数组自动对齐再逐元素运算」，掌握三条规则，
      并搞懂 LLM 里加 bias、缩放分数、LayerNorm 归一化背后都是广播。

运行：
    python examples/1_7_broadcasting.py
"""

1

def section(title: str) -> None:
    print("\n" + "=" * 58)
    print(title)
    print("=" * 58)


def main() -> None:
    # ---------------------------------------------------------------
    section("1) 直观：给 (3,4) 每一行加同一个长度 4 的向量")
    A = np.array([[1, 2, 3, 4],
                  [5, 6, 7, 8],
                  [9, 10, 11, 12]])
    b = np.array([10, 20, 30, 40])
    print("A.shape =", A.shape, " b.shape =", b.shape)
    print("A + b =\n", A + b, " <- b 被当作 (1,4) 沿第0维复制 3 份铺开")

    # ---------------------------------------------------------------
    section("2) 规则：右对齐，逐维取 max（用 broadcast_shapes 验证）")
    print("(3,4) 和 (4,)     ->", np.broadcast_shapes((3, 4), (4,)),
          " <- 左边缺位当作 1，即 (1,4)")
    print("(3,4) 和 (3,1)    ->", np.broadcast_shapes((3, 4), (3, 1)),
          " <- 最后一维 1 拉伸成 4")
    print("(8,1,6,1)和(7,1,5)->", np.broadcast_shapes((8, 1, 6, 1), (7, 1, 5)),
          " <- 每一维取 max")

    # ---------------------------------------------------------------
    section("3) 用 None 造新轴：让 (3,) 和 (4,) 凑成 (3,4)（外积效果）")
    a = np.array([1, 2, 3])          # (3,)
    c = np.array([10, 20, 30, 40])   # (4,)
    out = a[:, None] + c[None, :]    # (3,1) + (1,4) -> (3,4)
    print("a[:,None].shape =", a[:, None].shape,
          " c[None,:].shape =", c[None, :].shape)
    print("a[:,None] + c[None,:] =\n", out, " shape =", out.shape)
    print("其中 [i,j] = a[i]+c[j]，例如 [2,3] =", out[2, 3], "= a[2]+c[3] =", a[2] + c[3])

    # ---------------------------------------------------------------
    section("4) 报错：(3,4) + (2,4)，第0维 3 vs 2 都>1又不等")
    D = np.ones((2, 4))
    try:
        _ = A + D
    except ValueError as e:
        print("A + D 报错 ->", e)
    print("规则：每一维要么相等，要么有一个是 1，否则失败")

    # ---------------------------------------------------------------
    section("5) 坑：keepdims=True 才能广播回去（LayerNorm 式归一化）")
    x = np.array([[1.0, 2.0, 3.0, 4.0],
                  [10.0, 20.0, 30.0, 40.0]])   # (2,4)
    mean_keep = x.mean(axis=-1, keepdims=True)  # (2,1) 保住维度
    mean_drop = x.mean(axis=-1)                 # (2,)  掉了维度
    print("x.shape =", x.shape)
    print("mean(keepdims=True).shape =", mean_keep.shape, " <- 能广播回 (2,4)")
    print("mean(keepdims=False).shape=", mean_drop.shape, " <- 掉维度，广播会出错")
    print("(x - mean_keep) =\n", x - mean_keep, " <- 每行各减自己的均值，正确")
    try:
        _ = x - mean_drop  # (2,4) - (2,) 右对齐 4 vs 2 失败
    except ValueError as e:
        print("x - mean_drop 报错 ->", e, " <- 所以归一化都要 keepdims=True")

    # ---------------------------------------------------------------
    section("6) LLM 场景一：y = x @ W.T + b，bias 靠广播加到每个位置")
    rng = np.random.default_rng(0)
    xb = rng.standard_normal((2, 3, 5))   # (batch=2, seq=3, in=5)
    W = rng.standard_normal((4, 5))       # (out=4, in=5)
    bias = rng.standard_normal((4,))      # (out=4,)
    y = xb @ W.T + bias                   # (2,3,4) + (4,) 广播
    print("x@W.T.shape =", (xb @ W.T).shape, " bias.shape =", bias.shape)
    print("y.shape =", y.shape, " <- (4,) 广播到 batch 和 seq 的每个位置")

    # ---------------------------------------------------------------
    section("7) LLM 场景二：scores / sqrt(d_k)，标量广播到整张分数矩阵")
    d_k = 64
    scores = rng.standard_normal((3, 3))  # (seq, seq)
    scaled = scores / np.sqrt(d_k)        # () 标量广播
    print("scores.shape =", scores.shape, " sqrt(d_k) 是标量 shape =", np.sqrt(d_k).shape)
    print("scaled.shape =", scaled.shape, " <- 每个元素都除以同一个数")

    print("\n✅ 全部跑通！你已经理解广播=自动对齐+逐元素运算，以及它在 LLM 里的作用。")


if __name__ == "__main__":
    main()
