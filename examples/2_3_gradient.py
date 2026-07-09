"""
2.3 梯度(gradient) —— 验证脚本

配套知识点：数学知识点掌握表.md · 2.3
目标：用代码把「梯度 = 把所有偏导打包成的向量」跑一遍，看清：
      1) 梯度 = 偏导摞成向量：∇f = (∂f/∂x, ∂f/∂y, ...)
      2) 数值梯度(循环各分量做中心差分) ≈ 解析梯度（gradient check 原理）
      3) 方向导数 D_u f = ∇f·u：沿梯度方向坡度最大 = ‖∇f‖，反方向最小（下山最快）
      4) 梯度垂直于等高线：梯度·(等高线切线) ≈ 0
      5) 单神经元例子：算出梯度向量 ∇L，反着它走一步，损失下降到 0

运行：
    .venv/bin/python examples/2_3_gradient.py
"""

import numpy as np


def section(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


# 示例二元函数 f(x, y) = x^2 * y + 3y（沿用 2.2 的例子）
def f(x, y):
    return x ** 2 * y + 3 * y


# 解析梯度：把每个偏导求出来摞成向量
def grad_f(x, y):
    """∇f = (∂f/∂x, ∂f/∂y) = (2xy, x^2 + 3)。"""
    return np.array([2 * x * y, x ** 2 + 3], dtype=float)


def numerical_gradient(func, point, h=1e-5):
    """数值梯度：对每个分量做中心差分（只推第 i 个坐标，其余冻住），拼成向量。

    func 接收一个一维数组；point 是当前点(一维数组/列表)。
    """
    point = np.array(point, dtype=float)
    grad = np.zeros_like(point)
    for i in range(point.size):
        p_plus = point.copy()
        p_minus = point.copy()
        p_plus[i] += h
        p_minus[i] -= h
        grad[i] = (func(p_plus) - func(p_minus)) / (2 * h)
    return grad


def main() -> None:
    # ---------------------------------------------------------------
    section("1) 梯度 = 把所有偏导打包成一个向量（f = x^2·y + 3y）")
    for (x, y) in [(2.0, 5.0), (1.0, 1.0), (-3.0, 2.0)]:
        g = grad_f(x, y)
        print(f"  在 ({x:>4},{y:>4}): ∇f = ({g[0]:>6.1f}, {g[1]:>6.1f})")
    print("=> 偏导是分量，梯度是整根向量；换个点，方向和长度都变(∇f 是向量值函数)")

    # ---------------------------------------------------------------
    section("2) 数值梯度 ≈ 解析梯度（gradient check 的原理）")

    def f_vec(p):
        return f(p[0], p[1])

    for (x, y) in [(2.0, 5.0), (1.0, 1.0), (-3.0, 2.0)]:
        num = numerical_gradient(f_vec, [x, y])
        ana = grad_f(x, y)
        print(f"  在 ({x:>4},{y:>4}): 数值∇={np.round(num, 4)}  解析∇={ana}  差={np.abs(num-ana).max():.1e}")
    print("=> 循环每个分量做中心差分 → 拼成数值梯度；框架就靠它验证反向传播")

    # ---------------------------------------------------------------
    section("3) 方向导数 D_u f = ∇f·u：梯度方向坡度最大，反方向最小")
    # 用碗形 f2 = x^2 + y^2，梯度 = (2x, 2y)，在 (3,4) 处 ∇ = (6,8)，‖∇‖ = 10
    def f2(x, y):
        return x ** 2 + y ** 2

    def grad_f2(x, y):
        return np.array([2 * x, 2 * y], dtype=float)

    x, y = 3.0, 4.0
    g = grad_f2(x, y)
    gnorm = np.linalg.norm(g)
    print(f"f2 = x^2+y^2 在 (3,4): ∇f = {g}, 长度 ‖∇f‖ = {gnorm:.1f}")
    directions = {
        "沿梯度方向(最陡上坡)": g / gnorm,
        "反梯度方向(最陡下坡)": -g / gnorm,
        "垂直梯度(沿等高线)  ": np.array([-g[1], g[0]]) / gnorm,
        "任意斜方向          ": np.array([1.0, 0.0]),
    }
    for name, u in directions.items():
        u = u / np.linalg.norm(u)          # 保证是单位向量
        D = g @ u                          # 方向导数 = 点积(回忆 1.5)
        print(f"  {name}: u={np.round(u,3)}  方向导数 ∇f·u = {D:>7.3f}")
    print(f"=> 最大坡度 = +‖∇f‖ = {gnorm:.1f}(沿梯度)，最小 = -‖∇f‖ = {-gnorm:.1f}(反梯度=下山最快)")
    print("   沿等高线方向导数为 0 → 走这方向 f 不变")

    # ---------------------------------------------------------------
    section("4) 梯度垂直于等高线（∇f · 等高线切线 ≈ 0）")
    # 等高线 f2=常数 是同心圆；在 (3,4) 圆的切线方向是 (-y, x)=(-4,3)
    x, y = 3.0, 4.0
    g = grad_f2(x, y)
    tangent = np.array([-y, x])            # 圆 x^2+y^2=r^2 在 (x,y) 的切线方向
    dot = g @ tangent
    cos = dot / (np.linalg.norm(g) * np.linalg.norm(tangent))
    print(f"在 (3,4): 梯度 ∇f={g}, 等高线(圆)切线={tangent}")
    print(f"  点积 ∇f·切线 = {dot:.3f} → cosθ = {cos:.3f} → 夹角 = {np.degrees(np.arccos(np.clip(cos,-1,1))):.1f}°")
    print("=> 点积为 0、夹角 90° → 梯度垂直于等高线，指向数值增大的方向")

    # ---------------------------------------------------------------
    section("5) 单神经元：算出梯度向量 ∇L，反着它走，损失降到 0")
    # 线性神经元 ŷ = w·x + b, 损失 L=(ŷ-t)^2, 参数向量 (w, b)
    x_in, t = 2.0, 3.0        # 输入 & 目标
    w, b = 1.5, 0.5           # 当前参数
    eta = 0.1                 # 学习率
    print(f"神经元 ŷ = w·x + b, L=(ŷ-t)^2, 样本 x={x_in}, 目标 t={t}, 学习率 η={eta}")
    for step in range(1, 5):
        y_hat = w * x_in + b
        L = (y_hat - t) ** 2
        grad = np.array([2 * (y_hat - t) * x_in,   # ∂L/∂w
                         2 * (y_hat - t)])         # ∂L/∂b
        print(f"  第{step}步: (w,b)=({w:.4f},{b:.4f}) ŷ={y_hat:.4f} L={L:.6f} | ∇L=({grad[0]:.4f},{grad[1]:.4f})")
        w, b = np.array([w, b]) - eta * grad       # 整根参数向量反着梯度挪一步
        print(f"    更新后: (w,b)=({w:.4f},{b:.4f})")
    y_hat = w * x_in + b
    print(f"  收敛后: (w,b)=({w:.4f},{b:.4f}) ŷ={y_hat:.4f} L={(y_hat-t)**2:.8f}")
    print("=> 把两个偏导打包成梯度 ∇L，w←w-η∇L 一步步走，损失一路降到 0")
    print("   几十亿参数只是把这根梯度向量拉得很长很长")

    print("\n✅ 全部跑通！梯度 = 偏导打包成向量；负梯度 = 下山最快方向(2.5)。")


if __name__ == "__main__":
    main()
