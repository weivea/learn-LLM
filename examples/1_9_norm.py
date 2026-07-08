"""
1.9 范数(norm) L1 / L2 —— 验证脚本

配套知识点：数学知识点掌握表.md · 1.9
目标：理解「范数 = 向量的长度」，分清 L1 / L2，看清
      L2=直线长度、L1=拐着走，以及它们在正则化 / 梯度裁剪 / 归一化里的用处。

运行：
    python examples/1_9_norm.py
"""

import numpy as np


def section(title: str) -> None:
    print("\n" + "=" * 58)
    print(title)
    print("=" * 58)


def l1_norm(a: np.ndarray) -> float:
    """L1 范数：各分量绝对值之和。"""
    return float(np.sum(np.abs(a)))


def l2_norm(a: np.ndarray) -> float:
    """L2 范数：平方和开根号 = sqrt(a·a)。"""
    return float(np.sqrt(a @ a))


def main() -> None:
    # ---------------------------------------------------------------
    section("1) 定义：L2=直线长度, L1=拐着走（用 [3,4] 看区别）")
    a = np.array([3.0, 4.0])
    print("a =", a)
    print(f"L1 |a|_1 = |3|+|4|        = {l1_norm(a)}   （曼哈顿：先走3再走4）")
    print(f"L2 |a|_2 = sqrt(3^2+4^2)  = {l2_norm(a)}   （欧氏：斜边直线）")
    print("手写 vs NumPy 对照：")
    print("  np.linalg.norm(a, 1) =", np.linalg.norm(a, 1), " -> L1")
    print("  np.linalg.norm(a, 2) =", np.linalg.norm(a, 2), " -> L2（默认）")
    print("  np.linalg.norm(a)    =", np.linalg.norm(a), " -> 不写就是 L2")

    # ---------------------------------------------------------------
    section("2) L2 范数 = 向量和自己点积再开根号")
    v = np.array([1.0, 2.0, 3.0])
    print("v =", v)
    print("v · v            =", v @ v)
    print("sqrt(v·v)        =", round(np.sqrt(v @ v), 4))
    print("np.linalg.norm(v)=", round(float(np.linalg.norm(v)), 4), " <- 完全一致")

    # ---------------------------------------------------------------
    section("3) Lp 家族：p 越大越偏袒大分量，L∞ 只看最大分量")
    x = np.array([1.0, 2.0, 4.0])
    print("x =", x)
    for p in [1, 2, 3]:
        print(f"L{p} = (Σ|xi|^{p})^(1/{p}) = {np.linalg.norm(x, p):.4f}")
    print("L∞ = max|xi|          =", np.linalg.norm(x, np.inf), " <- 只剩最大的 4")

    # ---------------------------------------------------------------
    section("4) 归一化：除以 L2 范数 -> 单位向量（长度=1）")
    a_hat = a / np.linalg.norm(a)
    print("a        =", a, " |a|_2 =", l2_norm(a))
    print("a / |a|  =", np.round(a_hat, 4), " |a_hat|_2 =",
          round(float(np.linalg.norm(a_hat)), 4), " <- 正好 1")
    print("=> 这就是 1.8『先归一化再点积 = 余弦相似度』的那一步")

    # ---------------------------------------------------------------
    section("5) L1 偏爱稀疏：稀疏 vs 稠密权重，L1 差得多、L2 差不多")
    # w 是「模型权重向量」：模型学出来的一堆数字，每个决定某个输入有多重要。
    # 正则化就是对这个 w 求范数当惩罚项 —— L1 会把没用的权重逼成 0（稀疏）。
    sparse = np.array([3.0, 0.0, 0.0, 0.0])   # 能量集中在一个分量
    dense = np.array([1.5, 1.5, 1.5, 1.5])    # 能量摊平到四个分量
    print("稀疏 w =", sparse, "  L1 =", l1_norm(sparse), " L2 =", round(l2_norm(sparse), 4))
    print("稠密 w =", dense, "  L1 =", l1_norm(dense), " L2 =", round(l2_norm(dense), 4))
    print("=> L2 都是 3.0（一样），但 L1：稀疏 3.0 < 稠密 6.0")
    print("   L1 更青睐『集中在少数分量』的解 -> 正则时把其它逼成 0（稀疏）")

    # ---------------------------------------------------------------
    section("6) 梯度裁剪：梯度 L2 范数超阈值就等比缩回（方向不变）")
    grad = np.array([6.0, 8.0])               # L2 范数 = 10
    max_norm = 5.0
    gnorm = np.linalg.norm(grad)
    clipped = grad * (max_norm / gnorm) if gnorm > max_norm else grad
    print("原始梯度 grad =", grad, " L2 范数 =", gnorm)
    print(f"阈值 max_norm = {max_norm}，超了 -> 缩放系数 = {max_norm/gnorm}")
    print("裁剪后 grad   =", clipped, " L2 范数 =", round(float(np.linalg.norm(clipped)), 4))
    print("=> 方向不变（还是 3:4），只把长度压回阈值，防止梯度爆炸")

    print("\n✅ 全部跑通！范数 = 向量的长度：L2 管大小/稳定，L1 管稀疏。")


if __name__ == "__main__":
    main()
