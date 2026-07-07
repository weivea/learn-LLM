"""
1.3 张量 / 维度(dim, axis) —— 验证脚本

配套知识点：数学知识点掌握表.md · 1.3
目标：能解释 (batch, seq_len, d_model) 每一维；看到 shape 会数 ndim；
      理解 axis / dim 到底指哪个方向，沿某个轴做运算时「哪一维会消失」。

运行：
    python examples/1_3_tensor.py
"""

import numpy as np


def section(title: str) -> None:
    print("\n" + "=" * 56)
    print(title)
    print("=" * 56)


def main() -> None:
    # ---------------------------------------------------------------
    section("1) 张量 = 任意维数组的统称（标量→向量→矩阵→更高维）")
    scalar = np.array(3.14)                 # 0 维
    vector = np.array([1, 2, 3])            # 1 维
    matrix = np.array([[1, 2, 3],
                       [4, 5, 6]])          # 2 维
    tensor = np.arange(24).reshape(2, 3, 4)  # 3 维
    for name, x in [("标量 scalar", scalar),
                    ("向量 vector", vector),
                    ("矩阵 matrix", matrix),
                    ("张量 tensor", tensor)]:
        print(f"{name:12s} ndim={x.ndim}  shape={str(x.shape):12s} size={x.size}")
    print("=> ndim(维度数) = shape 里数字的个数 = 需要几个下标才能定位一个元素")

    # ---------------------------------------------------------------
    section("2) axis / dim：shape 从左到右就是 axis 0,1,2...")
    print("tensor.shape =", tensor.shape, " -> axis0=2, axis1=3, axis2=4")
    print("axis=0 是最外层（最先被剥开），axis=-1 是最内层（最里面那排数）")
    print("取一个元素要 3 个下标：tensor[1, 2, 3] =", tensor[1, 2, 3])
    print("tensor[0] 剥掉 axis0，得到一个 (3,4) 矩阵：\n", tensor[0])

    # ---------------------------------------------------------------
    section("3) 沿某个 axis 做运算：那一维会被『压掉』")
    M = np.array([[1, 2, 3],
                  [4, 5, 6]])           # shape (2, 3)
    print("M =\n", M, "  shape", M.shape)
    print("M.sum(axis=0) =", M.sum(axis=0), " -> 沿行方向相加，压掉 axis0，剩 (3,)")
    print("M.sum(axis=1) =", M.sum(axis=1), "     -> 沿列方向相加，压掉 axis1，剩 (2,)")
    print("记忆口诀：sum(axis=k) 就是把第 k 维求和吃掉，shape 里去掉第 k 个数")
    print("keepdims=True 可保留那一维为 1：",
          M.sum(axis=1, keepdims=True).shape, "-> (2, 1) 方便广播")

    # ---------------------------------------------------------------
    section("3b) 3 维也是同一规则：把 shape 第 k 个数删掉")
    t = np.arange(24).reshape(2, 3, 4)      # 想成『2 张 3x4 矩阵叠起来』
    print("t.shape =", t.shape, " (想成 2 张 3x4 的矩阵)")
    print("t[0] =\n", t[0])
    print("t[1] =\n", t[1])
    print("\nsum(axis=0) ->", t.sum(axis=0).shape,
          "：2 张矩阵『对应位置相加』叠成 1 张")
    print(t.sum(axis=0))
    print(f"  验证 [0,0]: t[0,0,0]={t[0,0,0]} + t[1,0,0]={t[1,0,0]} = {t.sum(axis=0)[0,0]}")
    print("\nsum(axis=1) ->", t.sum(axis=1).shape,
          "：每张矩阵内部把 3 行相加（纵向压掉行）")
    print(t.sum(axis=1))
    print(f"  验证 第0张第0列: {t[0,0,0]}+{t[0,1,0]}+{t[0,2,0]} = {t.sum(axis=1)[0,0]}")
    print("\nsum(axis=2) ->", t.sum(axis=2).shape,
          "：每行 4 个数横向相加（压掉列）")
    print(t.sum(axis=2))
    print(f"  验证 第0张第0行: {'+'.join(map(str, t[0, 0]))} = {t.sum(axis=2)[0,0]}")
    print("\n一次压多维 axis=(1,2) ->", t.sum(axis=(1, 2)).shape,
          "：只剩 batch，每张矩阵的总和", t.sum(axis=(1, 2)))
    print("通用心法：固定其它下标，只让第 k 个下标从头扫到尾，把扫过的元素加起来")

    # ---------------------------------------------------------------
    section("4) 增 / 减 维度：reshape、newaxis、squeeze")
    v = np.array([1, 2, 3])                     # (3,)
    col = v[:, np.newaxis]                      # (3, 1) 变成列
    row = v[np.newaxis, :]                      # (1, 3) 变成行
    print("v.shape        =", v.shape)
    print("v[:, None].shape =", col.shape, " -> 加一维当列")
    print("v[None, :].shape =", row.shape, " -> 加一维当行")
    print("col.squeeze().shape =", col.squeeze().shape, " -> squeeze 去掉所有长度为 1 的维")
    print("reshape 只改看法不改数据：",
          np.arange(12).reshape(3, 4).shape, "总元素数必须不变(=12)")

    # ---------------------------------------------------------------
    section("5) 调换维度顺序：transpose / 指定 axes")
    x = np.arange(24).reshape(2, 3, 4)         # (batch=2, seq=3, dim=4)
    xt = x.transpose(0, 2, 1)                   # 交换后两维 -> (2, 4, 3)
    print("x.shape          =", x.shape)
    print("x.transpose(0,2,1).shape =", xt.shape, " -> 只换 axis1 和 axis2 的顺序")
    print("注意力里常见：把 (B, seq, dim) 转成 (B, dim, seq) 好做矩阵乘法")

    # ---------------------------------------------------------------
    section("6) 直觉：LLM 里最常见的 3 维张量 (batch, seq_len, d_model)")
    batch, seq_len, d_model = 8, 128, 768
    hidden = np.zeros((batch, seq_len, d_model))
    print(f"hidden.shape = {hidden.shape}")
    print(f"  axis0 batch   = {batch:4d}  -> 一次喂进 {batch} 句话（并行处理）")
    print(f"  axis1 seq_len = {seq_len:4d}  -> 每句话 {seq_len} 个 token")
    print(f"  axis2 d_model = {d_model:4d}  -> 每个 token 用 {d_model} 维向量表示")
    print("=> 矩阵(一句话)再叠一层 batch，就成了 3 维张量")
    print("取第 0 句话：hidden[0].shape =", hidden[0].shape, " -> 回到 (seq_len, d_model) 矩阵")
    print("取第 0 句第 5 个 token：hidden[0, 5].shape =", hidden[0, 5].shape, " -> 一个 768 维向量")

    print("\n✅ 全部跑通！你已经能对着 shape 说清每一维、并知道 axis 指哪个方向。")


if __name__ == "__main__":
    main()
