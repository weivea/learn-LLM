"""
1.5 点积(dot product) —— 验证脚本

配套知识点：数学知识点掌握表.md · 1.5
目标：理解「点积 = 对应位置相乘再求和 = 相似度」，
      看清代数算法与几何算法一致，并把它和注意力里的 q · k 对上。

运行：
    python examples/1_5_dot_product.py
"""

import numpy as np


def section(title: str) -> None:
    print("\n" + "=" * 58)
    print(title)
    print("=" * 58)


def stable_softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / exp_x.sum(axis=-1, keepdims=True)


def main() -> None:
    # ---------------------------------------------------------------
    section("1) 定义：对应位置相乘，再全部加起来 -> 一个标量")
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    print("a =", a, " b =", b)
    print("a * b        =", a * b, " <- 逐元素相乘，还没求和，这不是点积")
    print("sum(a * b)   =", np.sum(a * b), " <- 再求和才是点积")
    print("np.dot(a, b) =", np.dot(a, b))
    print("a @ b        =", a @ b, " <- 一维时和 np.dot 等价")
    print("结果是标量，shape =", np.dot(a, b).shape, "（0 维）")

    # ---------------------------------------------------------------
    section("2) 两种算法一定相等：代数 == 几何")
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([4.0, 5.0, 6.0])
    algebra = np.dot(x, y)
    norm_x = np.linalg.norm(x)
    norm_y = np.linalg.norm(y)
    cos_theta = algebra / (norm_x * norm_y)
    geometry = norm_x * norm_y * cos_theta
    print("代数：sum(xi*yi)          =", algebra)
    print("几何：|x|*|y|*cosθ         =", geometry)
    print("|x| =", round(norm_x, 4), " |y| =", round(norm_y, 4),
          " cosθ =", round(cos_theta, 4))
    print("两者相等吗？", np.isclose(algebra, geometry))

    # ---------------------------------------------------------------
    section("3) 符号看方向：正=同向  零=垂直  负=反向")
    pairs = [
        ("同向", np.array([1, 0]), np.array([1, 0])),
        ("垂直", np.array([1, 0]), np.array([0, 1])),
        ("反向", np.array([1, 0]), np.array([-1, 0])),
        ("锐角", np.array([2, 1]), np.array([1, 1])),
        ("钝角", np.array([2, 1]), np.array([-1, -3])),
    ]
    for name, u, v in pairs:
        d = np.dot(u, v)
        sign = "> 0 同向" if d > 0 else ("= 0 垂直" if d == 0 else "< 0 反向")
        print(f"{name}: {u} · {v} = {d:>3}  -> {sign}")

    # ---------------------------------------------------------------
    section("4) 性质：a·a = |a|^2（点积衍生出长度/范数）")
    a = np.array([3.0, 4.0])
    print("a =", a)
    print("a · a          =", np.dot(a, a))
    print("|a|^2          =", np.linalg.norm(a) ** 2)
    print("sqrt(a · a)=|a|=", np.sqrt(np.dot(a, a)), "（这就是 1.9 范数）")

    # ---------------------------------------------------------------
    section("5) 点积是矩阵乘法的最小单元：一行 · 一列")
    A = np.array([[1, 2, 3],
                  [4, 5, 6]])
    B = np.array([[7, 8],
                  [9, 10],
                  [11, 12]])
    C = A @ B
    print("A @ B =\n", C)
    print("其中 C[0,0] = A第0行 · B第0列 =",
          "np.dot([1,2,3],[7,9,11]) =", np.dot(A[0], B[:, 0]))
    print("其中 C[1,1] = A第1行 · B第1列 =",
          "np.dot([4,5,6],[8,10,12]) =", np.dot(A[1], B[:, 1]))
    print("=> matmul 就是把一堆点积拼成矩阵")

    # ---------------------------------------------------------------
    section("6) 注意力直觉：q · k 越大 -> softmax 后权重越高")
    tokens = ["猫", "狗", "银行"]
    # 每个词一个 2 维“语义向量”：猫和狗方向接近，银行方向不同
    K = np.array([[1.0, 0.2],   # 猫
                  [0.9, 0.3],   # 狗
                  [0.1, 1.0]])  # 银行
    q = np.array([1.0, 0.15])   # 一个偏“猫/狗”方向的查询
    scores = K @ q              # 等价于对每个 k 做 q · k
    weights = stable_softmax(scores)
    print("查询 q =", q)
    for tok, k, s in zip(tokens, K, scores):
        print(f"  q · {tok}{k} = {s:.3f}")
    print("softmax 后的注意力权重：")
    for tok, w in zip(tokens, weights):
        print(f"  {tok}: {w:.3f}")
    print("权重加起来 =", round(weights.sum(), 3))
    print("=> 点积大的（猫/狗）拿到高注意力，点积小的（银行）被忽略")

    print("\n✅ 全部跑通！你已经理解点积 = 相似度，以及它在注意力里的作用。")


if __name__ == "__main__":
    main()
