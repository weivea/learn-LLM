"""
1.10 单位矩阵(identity) / 逆矩阵(inverse) —— 验证脚本

配套知识点：数学知识点掌握表.md · 1.10
目标：看清「单位矩阵 = 矩阵界的 1」「逆矩阵 = 矩阵界的倒数」，
      亲眼验证 A·I=A、A·A⁻¹=I、奇异矩阵没有逆，以及
      为什么实战里用 solve 而不是先求逆。

运行：
    python examples/1_10_identity_inverse.py
"""

import numpy as np


def section(title: str) -> None:
    print("\n" + "=" * 58)
    print(title)
    print("=" * 58)


def inv_2x2(m: np.ndarray) -> np.ndarray:
    """2x2 逆的手算公式：1/(ad-bc) * [[d,-b],[-c,a]]。"""
    (a, b), (c, d) = m
    det = a * d - b * c
    if np.isclose(det, 0.0):
        raise ValueError("det=0，矩阵奇异，没有逆")
    return (1.0 / det) * np.array([[d, -b], [-c, a]])


def main() -> None:
    # ---------------------------------------------------------------
    section("1) 单位矩阵 I = 矩阵界的『1』：A·I = I·A = A")
    I3 = np.eye(3)
    print("np.eye(3) =\n", I3)
    A = np.array([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0],
                  [7.0, 8.0, 10.0]])
    print("\nA =\n", A)
    print("\nA @ I 是否等于 A ? ->", np.allclose(A @ I3, A))
    print("I @ A 是否等于 A ? ->", np.allclose(I3 @ A, A))
    print("=> 乘单位矩阵，原样不变（就像任何数 ×1）")

    # ---------------------------------------------------------------
    section("2) 逆矩阵 A⁻¹ = 矩阵界的『倒数』：A·A⁻¹ = I")
    M = np.array([[4.0, 7.0],
                  [2.0, 6.0]])
    hand = inv_2x2(M)                 # 手算公式
    numpy_inv = np.linalg.inv(M)      # NumPy 求逆
    print("M =\n", M)
    print("\n手算逆 inv_2x2(M) =\n", np.round(hand, 4))
    print("np.linalg.inv(M)  =\n", np.round(numpy_inv, 4))
    print("\n两者一致 ? ->", np.allclose(hand, numpy_inv))
    print("M @ M⁻¹ =\n", np.round(M @ numpy_inv, 6))
    print("=> M @ M⁻¹ 正好是单位矩阵 I（乘回了 1）")

    # ---------------------------------------------------------------
    section("3) 不是所有方阵都有逆：det=0 -> 奇异 -> 求逆报错")
    S = np.array([[1.0, 2.0],
                  [2.0, 4.0]])         # 第二行 = 第一行×2，被『压扁』
    print("S =\n", S)
    print("det(S) =", round(float(np.linalg.det(S)), 6), " <- 等于 0")
    try:
        np.linalg.inv(S)
    except np.linalg.LinAlgError as e:
        print("np.linalg.inv(S) 抛错 ->", type(e).__name__, ":", e)
    print("=> det=0 把二维压成一条线，信息丢了，无法还原 -> 没有逆")

    # ---------------------------------------------------------------
    section("4) 用逆解方程 Ax=b：x=A⁻¹b vs np.linalg.solve")
    A2 = np.array([[3.0, 2.0],
                   [1.0, 2.0]])
    b = np.array([12.0, 8.0])
    x_by_inv = np.linalg.inv(A2) @ b        # 先求逆再乘（不推荐）
    x_by_solve = np.linalg.solve(A2, b)     # 直接解（推荐：更快更稳）
    print("A =\n", A2, "\nb =", b)
    print("x = A⁻¹ b        =", np.round(x_by_inv, 6))
    print("x = solve(A, b)  =", np.round(x_by_solve, 6))
    print("验证 A @ x =", np.round(A2 @ x_by_solve, 6), " == b ?",
          np.allclose(A2 @ x_by_solve, b))
    print("=> 结果一样，但实战用 solve：别真的先算 A⁻¹（又慢又不稳）")

    # ---------------------------------------------------------------
    section("5) 别混淆：逆 A⁻¹  ≠  转置 Aᵀ")
    print("M =\n", M)
    print("M⁻¹ (inv) =\n", np.round(np.linalg.inv(M), 4))
    print("Mᵀ  (转置) =\n", M.T)
    print("inv(M) 等于 Mᵀ 吗 ? ->", np.allclose(np.linalg.inv(M), M.T),
          " <- 一般不相等，是两回事")

    print("\n✅ 全部跑通！单位矩阵=矩阵界的1（残差里天天用），"
          "逆矩阵=矩阵界的倒数（重在理解，实战几乎不直接算）。")


if __name__ == "__main__":
    main()
