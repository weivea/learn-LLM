"""
1.8 余弦相似度(cosine similarity) —— 验证脚本

配套知识点：数学知识点掌握表.md · 1.8
目标：理解「余弦相似度 = 点积 ÷ 两个长度 = 只看方向的相似度」，
      看清它为什么不受向量长度影响，并对上 embedding 检索 / RAG。

运行：
    python examples/1_8_cosine_similarity.py
"""

import numpy as np


def section(title: str) -> None:
    print("\n" + "=" * 58)
    print(title)
    print("=" * 58)


def cosine_similarity(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    """手写余弦相似度：点积 / (|a| * |b|)，分母加 eps 防止除零。"""
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + eps))


def main() -> None:
    # ---------------------------------------------------------------
    section("1) 定义：余弦相似度 = 点积 / (|a| * |b|)")
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([4.0, 5.0, 6.0])
    dot = a @ b
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    cos = dot / (na * nb)
    print("a =", a, " b =", b)
    print(f"点积 a·b           = {dot}")
    print(f"|a| = {na:.4f}   |b| = {nb:.4f}")
    print(f"cosθ = a·b/(|a||b|) = {cos:.4f}")
    print("手写函数 cosine_similarity =", round(cosine_similarity(a, b), 4))

    # ---------------------------------------------------------------
    section("2) 另一种等价算法：先归一化成单位向量，再点积")
    a_hat = a / np.linalg.norm(a)
    b_hat = b / np.linalg.norm(b)
    print("单位向量 â =", np.round(a_hat, 4), " |â| =", round(np.linalg.norm(a_hat), 4))
    print("单位向量 b̂ =", np.round(b_hat, 4), " |b̂| =", round(np.linalg.norm(b_hat), 4))
    print("â · b̂ =", round(float(a_hat @ b_hat), 4), " <- 和上面 cosθ 完全一致")
    print("=> 余弦相似度 = 归一化后的点积")

    # ---------------------------------------------------------------
    section("3) 关键性质：放大向量，点积会变，余弦相似度不变")
    b2 = b * 10          # 方向不变，长度放大 10 倍
    print("b       =", b, " -> 点积 =", a @ b, " cos =", round(cosine_similarity(a, b), 4))
    print("b*10    =", b2, " -> 点积 =", a @ b2, " cos =", round(cosine_similarity(a, b2), 4))
    print("=> 点积从", a @ b, "变成", a @ b2, "，但余弦相似度纹丝不动（只看方向）")

    # ---------------------------------------------------------------
    section("4) 取值范围恒在 [-1, 1]")
    pairs = [
        ("完全同向", np.array([1.0, 0.0]), np.array([1.0, 0.0])),
        ("同向放大", np.array([1.0, 0.0]), np.array([2.0, 0.0])),
        ("45°",     np.array([1.0, 0.0]), np.array([1.0, 1.0])),
        ("垂直",     np.array([1.0, 0.0]), np.array([0.0, 1.0])),
        ("钝角",     np.array([1.0, 0.0]), np.array([-1.0, 1.0])),
        ("完全反向", np.array([1.0, 0.0]), np.array([-1.0, 0.0])),
    ]
    for name, u, v in pairs:
        print(f"{name:6}: {u} vs {v} -> cos = {cosine_similarity(u, v):+.4f}")

    # ---------------------------------------------------------------
    section("5) 坑：零向量会除以 0")
    zero = np.array([0.0, 0.0])
    x = np.array([1.0, 2.0])
    with np.errstate(invalid="ignore", divide="ignore"):
        naive = x @ zero / (np.linalg.norm(x) * np.linalg.norm(zero))
    print("不加 eps: cos =", naive, "  <- nan（分母为 0）")
    print("加 eps:   cos =", round(cosine_similarity(x, zero), 4), "  <- 安全返回 0")

    # ---------------------------------------------------------------
    section("6) 批量：一次算出所有 pair 的相似度矩阵（检索/RAG 核心）")
    tokens = ["猫", "小猫", "银行"]
    E = np.array([[1.0, 0.2],   # 猫
                  [0.9, 0.3],   # 小猫（和“猫”方向接近）
                  [0.1, 1.0]])  # 银行（方向不同）
    # 先把每个向量归一化成单位向量，再 A @ A.T 就是余弦相似度矩阵
    E_norm = E / np.linalg.norm(E, axis=1, keepdims=True)
    sim = E_norm @ E_norm.T
    print("相似度矩阵（行/列 = 猫, 小猫, 银行）：")
    print(np.round(sim, 3))

    query = "猫"
    qi = tokens.index(query)
    scores = sim[qi].copy()
    scores[qi] = -np.inf                      # 排除自己
    top = int(np.argmax(scores))
    print(f"\n查询『{query}』最相似的是 -> 『{tokens[top]}』"
          f"（cos = {sim[qi, top]:.3f}）")
    print("=> 这就是向量检索：按余弦相似度排序取 Top-K")

    print("\n✅ 全部跑通！余弦相似度 = 只看方向的点积，检索/RAG 的排序标准。")


if __name__ == "__main__":
    main()
