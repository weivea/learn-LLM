"""
1.11 特征值(eigenvalue) / 特征向量(eigenvector) —— 验证脚本

配套知识点：数学知识点掌握表.md · 1.11
目标：看清「特征向量 = 变换中方向不变、只被拉伸的特殊方向」
      「特征值 = 拉伸的倍数」，亲眼验证 Av=λv、几何上特征向量不转向、
      特征值之和=迹/之积=行列式，以及 PCA 降维在借它做什么。

运行：
    python examples/1_11_eigen.py
"""

import numpy as np


def section(title: str) -> None:
    print("\n" + "=" * 58)
    print(title)
    print("=" * 58)


def angle_deg(u: np.ndarray, v: np.ndarray) -> float:
    """两个向量的夹角（度）。用来看『变换后有没有转向』。"""
    cos = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
    cos = np.clip(cos, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


def main() -> None:
    A = np.array([[2.0, 1.0],
                  [1.0, 2.0]])          # 对称矩阵，特征值 3 和 1

    # ---------------------------------------------------------------
    section("1) 定义：A·v = λ·v （每个特征对都满足）")
    eigvals, eigvecs = np.linalg.eig(A)   # 列 eigvecs[:, i] 对应 eigvals[i]
    print("A =\n", A)
    print("\n特征值 λ =", np.round(eigvals, 4))
    print("特征向量（按列，已归一化到长度 1）=\n", np.round(eigvecs, 4))
    for i in range(len(eigvals)):
        lam, v = eigvals[i], eigvecs[:, i]
        print(f"\nλ={lam:.3f}:  A@v =", np.round(A @ v, 4),
              " vs  λ*v =", np.round(lam * v, 4),
              " 相等? ->", np.allclose(A @ v, lam * v))
    print("=> 对特征向量，矩阵变换退化成『只乘一个数 λ』")

    # ---------------------------------------------------------------
    section("2) 几何直觉：普通向量会转向，特征向量不转向")
    r = np.array([1.0, 0.0])              # 一个普通方向
    print("普通向量 r =", r, " 经过 A 后 A@r =", np.round(A @ r, 3))
    print("  变换前后夹角 =", round(angle_deg(r, A @ r), 2), "度  <- 转向了")
    v = eigvecs[:, 0]
    print("\n特征向量 v =", np.round(v, 3), " 经过 A 后 A@v =", np.round(A @ v, 3))
    print("  变换前后夹角 =", round(angle_deg(v, A @ v), 2),
          "度  <- 不转向（只被拉伸）")

    # ---------------------------------------------------------------
    section("3) 手算特征多项式 det(A-λI)=0 的根 == np.linalg.eig")
    # (2-λ)^2 - 1 = λ^2 - 4λ + 3 = 0  ->  λ = 3, 1
    roots = np.sort(np.roots([1.0, -4.0, 3.0]))          # 手算多项式的根
    print("手算 det(A-λI)=λ²-4λ+3=0 的根 =", np.round(roots, 4))
    print("np.linalg.eig 的特征值        =", np.round(np.sort(eigvals), 4))
    print("一致? ->", np.allclose(roots, np.sort(eigvals)))

    # ---------------------------------------------------------------
    section("4) 好用性质：Σλ = 迹(trace)， Πλ = 行列式(det)")
    print("特征值之和 =", round(float(eigvals.sum()), 4),
          "   迹 tr(A) =", round(float(np.trace(A)), 4))
    print("特征值之积 =", round(float(eigvals.prod()), 4),
          "   行列式 det(A) =", round(float(np.linalg.det(A)), 4))
    # 造一个奇异矩阵：有一个 λ=0 <=> det=0 <=> 不可逆（呼应 1.10）
    S = np.array([[1.0, 2.0],
                  [2.0, 4.0]])            # 第二行=第一行×2，被压扁
    s_eig = np.linalg.eig(S)[0]
    print("\n奇异矩阵 S 的特征值 =", np.round(s_eig, 4),
          " 含 0 -> det=0 -> 不可逆（呼应 1.10）")

    # ---------------------------------------------------------------
    section("5) 对称矩阵：不同特征值的特征向量互相垂直（正交）")
    v0, v1 = eigvecs[:, 0], eigvecs[:, 1]
    print("v0 =", np.round(v0, 3), " v1 =", np.round(v1, 3))
    print("v0 · v1 =", round(float(np.dot(v0, v1)), 6),
          " ≈ 0 -> 垂直（呼应 1.5 点积）")

    # ---------------------------------------------------------------
    section("6) PCA 直觉：协方差矩阵最大特征值方向 = 数据铺得最开的方向")
    rng = np.random.default_rng(0)
    # 造一批沿 (1,1) 方向拉长的相关数据
    base = rng.normal(size=(500, 2)) * np.array([3.0, 0.6])
    theta = np.radians(45)
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    data = base @ R.T                      # 旋转 45°，主散布方向变成 (1,1)
    C = np.cov(data, rowvar=False)         # 2x2 协方差矩阵（对称）
    vals, vecs = np.linalg.eig(C)
    order = np.argsort(vals)[::-1]         # 特征值从大到小
    vals, vecs = vals[order], vecs[:, order]
    print("协方差矩阵 C =\n", np.round(C, 3))
    print("特征值(方差) =", np.round(vals, 3), " <- 第一个明显更大")
    print("最大特征值对应的主方向 =", np.round(vecs[:, 0], 3),
          " ≈ 数据拉长的 (0.707, 0.707) 方向")
    print("=> 降维 = 只保留特征值大的前 k 个方向，丢掉方差极小的方向")

    print("\n✅ 全部跑通！特征向量=矩阵不改方向的特殊方向，"
          "特征值=缩放倍数；PCA 降维就靠它（保留方差最大的几个方向）。")


if __name__ == "__main__":
    main()
