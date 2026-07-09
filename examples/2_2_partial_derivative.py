"""
2.2 偏导数(partial derivative) —— 验证脚本

配套知识点：数学知识点掌握表.md · 2.2
目标：用代码把「偏导 = 冻住其它变量，只对一个变量求导」跑一遍，看清：
      1) 冻住其它变量 → 多元退化成一元，直接套一元法则
      2) 数值偏导(只推一个坐标) ≈ 解析偏导，并拼成整个梯度向量
      3) 偏导正负/大小 = 每个参数各自往哪调、影响多大
      4) 多元一阶近似 Δf ≈ ∂x·Δx + ∂y·Δy（= 偏导向量·改动向量 的点积）
      5) 单神经元例子：算 ∂L/∂w、∂L/∂b，做一步更新，损失下降到 0

运行：
    python examples/2_2_partial_derivative.py
"""

import numpy as np


def section(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


# 示例二元函数 f(x, y) = x^2 * y + 3y
def f(x, y):
    return x ** 2 * y + 3 * y


# 解析偏导：把另一个变量当常数
def df_dx(x, y):
    """∂f/∂x：把 y 当常数 -> 2x·y。"""
    return 2 * x * y


def df_dy(x, y):
    """∂f/∂y：把 x 当常数 -> x^2 + 3。"""
    return x ** 2 + 3


def numerical_partial(func, point, i, h=1e-5):
    """对第 i 个分量做中心差分偏导：只给第 i 个坐标 +/- h，其余冻住不动。

    func 接收一个一维数组；point 是当前点(一维数组/列表)。
    """
    p_plus = np.array(point, dtype=float)
    p_minus = np.array(point, dtype=float)
    p_plus[i] += h
    p_minus[i] -= h
    return (func(p_plus) - func(p_minus)) / (2 * h)


def main() -> None:
    x0, y0 = 2.0, 5.0  # 全程盯着 f 在 (2,5) 这一点

    # ---------------------------------------------------------------
    section("1) 冻住其它变量，只对一个变量求导（f = x^2·y + 3y）")
    print("在点 (x,y)=(2,5)：")
    print(f"  ∂f/∂x = 2x·y      (把 y 当常数) = 2·2·5 = {df_dx(x0, y0):.1f}")
    print(f"  ∂f/∂y = x^2 + 3   (把 x 当常数) = 2^2+3 = {df_dy(x0, y0):.1f}")
    print("=> 求 ∂/∂x 时把 y 冻成常数，多元问题当场退化成 2.1 的一元求导")

    # ---------------------------------------------------------------
    section("2) 数值偏导 ≈ 解析偏导，并拼成梯度向量")
    # 把 f 包成「吃一个向量」的形式，方便数值偏导只推某个分量
    def f_vec(p):
        return f(p[0], p[1])

    for (x, y) in [(2.0, 5.0), (1.0, 1.0), (-3.0, 2.0)]:
        num = np.array([numerical_partial(f_vec, [x, y], 0),
                        numerical_partial(f_vec, [x, y], 1)])
        ana = np.array([df_dx(x, y), df_dy(x, y)])
        print(f"  在 ({x:>4},{y:>4}): 数值梯度={np.round(num, 4)}  解析梯度={ana}  差={np.abs(num-ana).max():.1e}")
    print("=> 每次只推一个坐标(其余冻住)得到一个偏导，n 个偏导拼起来 = 梯度向量∇f")
    print("   (框架做 gradient check 验证反向传播，就是这么算的)")

    # ---------------------------------------------------------------
    section("3) 偏导的正负/大小 = 每个参数各自怎么调")
    x, y = 2.0, 5.0
    gx, gy = df_dx(x, y), df_dy(x, y)
    print(f"在 (2,5)：∂f/∂x={gx:.1f}，∂f/∂y={gy:.1f}（都>0）")
    print("  ∂f/∂x>0：只调大 x → f 变大；想让 f 变小就调小 x")
    print("  ∂f/∂y>0：只调大 y → f 变大；想让 f 变小就调小 y")
    print(f"  |∂f/∂x|={abs(gx):.0f} > |∂f/∂y|={abs(gy):.0f} → 此处 x 比 y 更『敏感』，动 x 影响更大")

    # ---------------------------------------------------------------
    section("4) 多元一阶近似：Δf ≈ ∂x·Δx + ∂y·Δy（就是点积）")
    x, y = 2.0, 5.0
    grad = np.array([df_dx(x, y), df_dy(x, y)])   # 偏导向量
    print(f"在 (2,5), 梯度=∂=[{grad[0]:.0f}, {grad[1]:.0f}]. 预测 Δf ≈ 梯度·[Δx,Δy]：")
    for dx, dy in [(0.1, 0.1), (0.05, -0.1), (-0.2, 0.0)]:
        delta = np.array([dx, dy])
        predict = grad @ delta               # 一阶近似(点积, 回忆 1.5)
        true = f(x + dx, y + dy) - f(x, y)   # 真实变化
        print(f"  Δ=[{dx:>5},{dy:>5}]: 预测Δf={predict:>8.4f}  真实Δf={true:>8.4f}  误差={abs(predict-true):.4f}")
    print("=> 改动越小，一阶近似越准。这就是『各参数一起变一点，损失变多少』的原型")

    # ---------------------------------------------------------------
    section("5) 单神经元：分别对 w、b 求偏导，做一步梯度下降")
    # 线性神经元 ŷ = w·x + b, 损失 L = (ŷ - t)^2, 依赖 w 和 b 两个参数
    x_in, t = 2.0, 3.0        # 输入 & 目标
    w, b = 1.5, 0.5           # 当前参数
    eta = 0.1                 # 学习率
    print(f"神经元 ŷ = w·x + b, 损失 L=(ŷ-t)^2, 样本 x={x_in}, 目标 t={t}")
    for step in range(1, 4):
        y_hat = w * x_in + b
        L = (y_hat - t) ** 2
        # 偏导：把另一个参数当常数
        dL_dw = 2 * (y_hat - t) * x_in   # ∂L/∂w
        dL_db = 2 * (y_hat - t) * 1      # ∂L/∂b
        print(f"  第{step}步: w={w:.4f} b={b:.4f} ŷ={y_hat:.4f} L={L:.5f} | ∂L/∂w={dL_dw:.4f} ∂L/∂b={dL_db:.4f}")
        w = w - eta * dL_dw              # 各参数反着自己的偏导挪一步
        b = b - eta * dL_db
    y_hat = w * x_in + b
    print(f"  收敛后: w={w:.4f} b={b:.4f} ŷ={y_hat:.4f} L={(y_hat-t)**2:.6f}")
    print("=> 分别用 ∂L/∂w、∂L/∂b 更新，损失一路下降到 0")
    print("   这就是一层网络训练的核心；几十亿参数只是把它放大重复")

    print("\n✅ 全部跑通！偏导 = 冻住其它变量只对一个求导；把所有偏导打包就是梯度(2.3)。")


if __name__ == "__main__":
    main()
