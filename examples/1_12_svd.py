"""
1.12 SVD 奇异值分解(Singular Value Decomposition) —— 验证脚本

配套知识点：数学知识点掌握表.md · 1.12
目标：亲眼看清「任意矩阵 A = U·Σ·Vᵀ = 旋转→按奇异值拉伸→旋转」，
      奇异值 = √(AᵀA 的特征值)、非零奇异值个数 = 秩、单位圆被拉成椭圆，
      以及低秩近似（LoRA / 压缩的原理）到底省了多少、丢了多少。

运行：
    python examples/1_12_svd.py
"""

import numpy as np


def section(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def main() -> None:
    A = np.array([[3.0, 0.0],
                  [4.0, 5.0]])          # 手算例子：AᵀA=[[25,20],[20,25]]

    # ---------------------------------------------------------------
    section("1) 定义：A = U · Σ · Vᵀ，乘回去 == 原矩阵")
    # full_matrices=False：Σ 只给对角线上的奇异值（一维数组 s）
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    print("A =\n", A)
    print("\nU (左奇异向量，按列) =\n", np.round(U, 4))
    print("奇异值 σ =", np.round(s, 4), " <- 恒 ≥ 0，且从大到小排")
    print("Vᵀ (右奇异向量，按行) =\n", np.round(Vt, 4))
    A_rebuilt = U @ np.diag(s) @ Vt
    print("\nU·Σ·Vᵀ 还原 A? ->", np.allclose(A_rebuilt, A))
    print("U 正交(UᵀU=I)? ->", np.allclose(U.T @ U, np.eye(2)),
          "  V 正交? ->", np.allclose(Vt @ Vt.T, np.eye(2)))

    # ---------------------------------------------------------------
    section("2) 奇异值 == √(AᵀA 的特征值)（SVD 建立在特征分解上）")
    AtA = A.T @ A
    eigvals = np.linalg.eigvalsh(AtA)          # 对称矩阵，特征值升序
    sigma_from_eig = np.sqrt(np.sort(eigvals)[::-1])
    print("AᵀA =\n", AtA, "  (对称半正定)")
    print("AᵀA 的特征值 =", np.round(np.sort(eigvals)[::-1], 4),
          " (手算: 25±20 = 45, 5)")
    print("√特征值        =", np.round(sigma_from_eig, 4))
    print("np.linalg.svd 的 σ =", np.round(s, 4),
          "  一致? ->", np.allclose(sigma_from_eig, s))

    # ---------------------------------------------------------------
    section("3) 特征分解做不到的事：非方阵也能 SVD")
    B = np.array([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0]])            # 2×3，根本不是方阵
    Ub, sb, Vtb = np.linalg.svd(B, full_matrices=False)
    print("B (2×3) =\n", B)
    print("B 的奇异值 =", np.round(sb, 4))
    print("非零奇异值个数 =", int(np.sum(sb > 1e-10)),
          " == rank(B) =", np.linalg.matrix_rank(B))
    print("=> 奇异值 0 的方向被压扁；非零个数就是秩（呼应 1.10/1.11）")

    # ---------------------------------------------------------------
    section("4) 几何：单位圆 --A--> 椭圆，半轴长 = 奇异值")
    ts = np.linspace(0, 2 * np.pi, 400)
    circle = np.stack([np.cos(ts), np.sin(ts)])   # 2×400，单位圆上的点
    ellipse = A @ circle                          # 变换后
    radii = np.linalg.norm(ellipse, axis=0)       # 每个点到原点的距离
    print("单位圆经过 A 后，点到原点的最远/最近距离 =",
          round(radii.max(), 4), "/", round(radii.min(), 4))
    print("对应奇异值 σ_max / σ_min              =",
          round(s.max(), 4), "/", round(s.min(), 4))
    print("=> 椭圆长/短半轴 == 最大/最小奇异值（A 的最大/最小拉伸率）")

    # ---------------------------------------------------------------
    section("5) 好用规律：Πσ = |det A|， σ_max/σ_min = 条件数")
    print("σ 之积 =", round(float(np.prod(s)), 4),
          "   |det A| =", round(abs(float(np.linalg.det(A))), 4))
    print("条件数 σ_max/σ_min =", round(float(s.max() / s.min()), 4),
          "  (越大越病态，越难数值求逆)")

    # ---------------------------------------------------------------
    section("6) ★ 低秩近似：留前 k 项 = 最优压缩（LoRA / 压缩的原理）")
    rng = np.random.default_rng(0)
    m, n, true_rank = 60, 40, 3
    # errstate：屏蔽 macOS Accelerate BLAS 在 numpy 2.0 上对 matmul 的
    # 「假」浮点告警（结果完全正确，只是底层库多报了一下）
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        # 造一个「本质低秩 + 一点噪声」的矩阵：秩≈3 的信号
        M = rng.normal(size=(m, true_rank)) @ rng.normal(size=(true_rank, n))
        M += 0.01 * rng.normal(size=(m, n))           # 加一丢丢噪声
        Um, sm, Vtm = np.linalg.svd(M, full_matrices=False)
        print("60×40 矩阵的前 8 个奇异值 =", np.round(sm[:8], 3))
        print("=> 前 3 个很大、之后骤降到近 0 -> 矩阵『近似秩 3』\n")

        full_store = m * n
        print(f"{'k':>2} | {'相对误差':>10} | {'存储量':>8} | {'压缩比':>6}")
        print("-" * 40)
        for k in (1, 2, 3, 5, 10):
            Mk = Um[:, :k] @ np.diag(sm[:k]) @ Vtm[:k, :]   # 秩-k 近似
            err = np.linalg.norm(M - Mk) / np.linalg.norm(M)
            store = k * (m + n)                             # 只需存 U_k, σ, V_k
            print(f"{k:>2} | {err:>10.4%} | {store:>8} | {full_store/store:>5.1f}x")
    print("\n=> k=3 就几乎无损，存储从", full_store, "降到", 3 * (m + n),
          "。这正是 LoRA『只学低秩补丁 ΔW=BA』省参数的原理。")

    print("\n✅ 全部跑通！任意矩阵 = 旋转→按奇异值拉伸→旋转；"
          "留前 k 项 = 最优低秩近似（LoRA/PCA/压缩的地基）。")


if __name__ == "__main__":
    main()
