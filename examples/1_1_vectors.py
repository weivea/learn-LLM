"""
1.1 向量的概念与表示 —— 验证脚本

配套知识点：数学知识点掌握表.md · 1.1
目标：用代码验证「向量 = 一组有序的数」，并建立到 LLM embedding 的直觉。

运行：
    python examples/1_1_vectors.py
"""

import numpy as np


def section(title: str) -> None:
    print("\n" + "=" * 50)
    print(title)
    print("=" * 50)


def main() -> None:
    # ---------------------------------------------------------------
    section("1) 向量就是一组有序的数")
    v = np.array([3, 1])
    print("v            =", v)
    print("v.shape      =", v.shape, " -> (2,) 说明是 2 维向量")
    print("v.ndim       =", v.ndim, "    -> 一维数组，就是向量")
    print("v 的分量: v1 =", v[0], ", v2 =", v[1])

    # 有序很重要：[3,1] 与 [1,3] 是不同的向量
    print("\n有序性验证:")
    print("[3,1] == [1,3] ?", np.array_equal(np.array([3, 1]), np.array([1, 3])))

    # ---------------------------------------------------------------
    section("2) 代数运算：数乘 与 加法")
    print("2 * v            =", 2 * v, "   (每个分量都乘 2)")
    print("v + [1, 4]       =", v + np.array([1, 4]), "   (对应分量相加)")
    print("v - [1, 4]       =", v - np.array([1, 4]))

    # ---------------------------------------------------------------
    section("3) 几何意义：向量的长度（范数，预习 1.9）")
    # 长度 = sqrt(3^2 + 1^2)
    length = np.linalg.norm(v)
    print("v =", v, "的长度 |v| =", round(float(length), 4))
    print("手算校验: sqrt(3^2 + 1^2) =", round((3**2 + 1**2) ** 0.5, 4))

    # ---------------------------------------------------------------
    section("4) 高维向量：模拟一个 768 维的词向量 (embedding)")
    rng = np.random.default_rng(seed=42)  # 固定随机种子，结果可复现
    cat = rng.standard_normal(768)
    print("embedding('猫').shape =", cat.shape, " -> 768 维，画不出来但数学一样")
    print("前 5 个分量:", np.round(cat[:5], 3))

    # ---------------------------------------------------------------
    section("5) 直觉预告：方向相近 = 意思相近（点积，预习 1.5 / 1.8）")
    # 造两个方向接近的向量 + 一个方向不同的向量
    dog = cat + 0.1 * rng.standard_normal(768)   # 和「猫」很像
    bank = rng.standard_normal(768)              # 和「猫」无关

    def cosine(a: np.ndarray, b: np.ndarray) -> float:
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

    print("cos(猫, 狗 )  =", round(cosine(cat, dog), 4), " -> 接近 1，很像")
    print("cos(猫, 银行) =", round(cosine(cat, bank), 4), " -> 接近 0，不相关")

    print("\n✅ 全部跑通！你已经用代码验证了「向量是什么」。")


if __name__ == "__main__":
    main()
