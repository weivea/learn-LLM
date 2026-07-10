"""
2.4 链式法则(chain rule) —— 验证脚本

配套知识点：数学知识点掌握表.md · 2.4
目标：用代码把「链式法则 = 复合函数求导 = 各段导数连乘」跑一遍，看清：
      1) 单变量：z=(3x+1)^2、z=√sin(x^2)，解析(连乘) ≈ 数值
      2) 多路径：z=u·v, u=t^2, v=sin t，两条路径「相乘再相加」≈ 数值
      3) 计算图反传：线性神经元 L=(wx+b-t)^2，逐边乘本地导数 → ∂L/∂w, ∂L/∂b
      4) 两层网络：前向 + 反向传播一次，链式法则梯度 ≈ 数值梯度(gradient check)
      5) 连乘的副作用：几十层导数相乘 → 梯度消失/爆炸

运行：
    .venv/bin/python examples/2_4_chain_rule.py
"""

import numpy as np


def section(title: str) -> None:
    print("\n" + "=" * 64)
    print(title)
    print("=" * 64)


def numerical_derivative(func, x, h=1e-5):
    """中心差分：把复合函数当整体，直接数值求 df/dx。"""
    return (func(x + h) - func(x - h)) / (2 * h)


def main() -> None:
    # ---------------------------------------------------------------
    section("1) 单变量链式法则：各段导数连乘 ≈ 数值导数")

    # 例1: z = (3x+1)^2，拆 y=3x+1, z=y^2 → dz/dx = 2y·3 = 6(3x+1)
    def z1(x):
        return (3 * x + 1) ** 2

    def dz1_analytic(x):
        return 2 * (3 * x + 1) * 3          # 外层 2y · 内层 3

    # 例3(三层): z = √(sin(x^2))，u=x^2, v=sin u, z=√v
    def z3(x):
        return np.sqrt(np.sin(x ** 2))

    def dz3_analytic(x):
        u = x ** 2
        v = np.sin(u)
        return (1 / (2 * np.sqrt(v))) * np.cos(u) * (2 * x)   # 三段连乘

    for x in [0.5, 1.0, 1.2]:
        a1, n1 = dz1_analytic(x), numerical_derivative(z1, x)
        print(f"  z=(3x+1)^2   x={x}: 解析(连乘)={a1:.5f}  数值={n1:.5f}  差={abs(a1-n1):.1e}")
    for x in [0.6, 1.0, 1.3]:
        a3, n3 = dz3_analytic(x), numerical_derivative(z3, x)
        print(f"  z=√sin(x^2)  x={x}: 解析(3段连乘)={a3:.5f}  数值={n3:.5f}  差={abs(a3-n3):.1e}")
    print("=> 拆成几层，就有几个因子；全部相乘 = 复合函数的导数")

    # ---------------------------------------------------------------
    section("2) 多路径链式法则：一条路径相乘，多条路径相加")
    # z = u·v, u = t^2, v = sin t
    #   dz/dt = (∂z/∂u)(du/dt) + (∂z/∂v)(dv/dt) = v·2t + u·cos t
    def z_multi(t):
        return (t ** 2) * np.sin(t)

    def dz_multi_analytic(t):
        u, v = t ** 2, np.sin(t)
        path_u = v * (2 * t)        # 经 u 的路径
        path_v = u * np.cos(t)      # 经 v 的路径
        return path_u + path_v, path_u, path_v

    for t in [0.5, 1.0, 2.0]:
        total, pu, pv = dz_multi_analytic(t)
        num = numerical_derivative(z_multi, t)
        print(f"  t={t}: 路径u={pu:.4f} + 路径v={pv:.4f} = {total:.4f}  数值={num:.4f}  差={abs(total-num):.1e}")
    print("=> 分叉的两条路径各自连乘，再相加 —— 反向传播里到处是这种「梯度相加」")

    # ---------------------------------------------------------------
    section("3) 计算图反向传播：线性神经元 L=(wx+b-t)^2")
    x_in, t = 2.0, 3.0
    w, b = 1.5, 0.5

    # --- 前向 forward ---
    prod = w * x_in          # wx
    y_hat = prod + b         # ŷ = wx + b
    e = y_hat - t            # 误差
    L = e ** 2               # 损失
    print(f"前向: w={w}, b={b}, x={x_in}, t={t}")
    print(f"      wx={prod}, ŷ={y_hat}, e=ŷ-t={e}, L=e^2={L}")

    # --- 反向 backward：从 L 出发，逐边乘本地导数 ---
    dL_de = 2 * e            # L=e^2      → ∂L/∂e = 2e
    de_dyhat = 1.0          # e=ŷ-t      → ∂e/∂ŷ = 1
    dyhat_dw = x_in         # ŷ=wx+b     → ∂ŷ/∂w = x
    dyhat_db = 1.0          # ŷ=wx+b     → ∂ŷ/∂b = 1

    dL_dw = dL_de * de_dyhat * dyhat_dw   # 连乘
    dL_db = dL_de * de_dyhat * dyhat_db
    print(f"反向: ∂L/∂e=2e={dL_de}  (被 w、b 两条路复用)")
    print(f"      ∂L/∂w = 2e·1·x = {dL_dw}   ∂L/∂b = 2e·1·1 = {dL_db}")
    print(f"=> 链式法则算出的梯度 ∇L=({dL_dw}, {dL_db})，与 2.3 神经元一致")

    # ---------------------------------------------------------------
    section("4) 两层网络：前向 + 反向传播一次，链式法则 ≈ 数值梯度")
    # 结构: x --w1,b1--> h=relu(w1·x+b1) --w2,b2--> ŷ=w2·h+b2, L=(ŷ-t)^2
    x_in, t = 1.0, 2.0
    params = {"w1": 0.8, "b1": 0.1, "w2": -0.5, "b2": 0.3}

    def relu(z):
        return max(0.0, z)

    def forward(p):
        z1 = p["w1"] * x_in + p["b1"]
        h = relu(z1)
        y = p["w2"] * h + p["b2"]
        return (y - t) ** 2

    # --- 解析反向传播（手推的链式法则）---
    z1 = params["w1"] * x_in + params["b1"]
    h = relu(z1)
    y_hat = params["w2"] * h + params["b2"]
    L = (y_hat - t) ** 2

    dL_dy = 2 * (y_hat - t)                    # ∂L/∂ŷ
    grad = {}
    grad["w2"] = dL_dy * h                     # ∂L/∂ŷ · ∂ŷ/∂w2
    grad["b2"] = dL_dy * 1.0
    dL_dh = dL_dy * params["w2"]               # 回传到隐藏层
    dh_dz1 = 1.0 if z1 > 0 else 0.0            # relu 的本地导数
    dL_dz1 = dL_dh * dh_dz1
    grad["w1"] = dL_dz1 * x_in                 # ∂L/∂z1 · ∂z1/∂w1
    grad["b1"] = dL_dz1 * 1.0

    print(f"前向: z1={z1:.3f}, h={h:.3f}, ŷ={y_hat:.3f}, L={L:.4f}")
    print("参数    解析梯度(链式法则)   数值梯度(中心差分)      差")
    for name in ["w1", "b1", "w2", "b2"]:
        def f_scalar(val, k=name):
            p = dict(params)
            p[k] = val
            return forward(p)
        num = numerical_derivative(f_scalar, params[name])
        print(f"  {name}:   {grad[name]:>12.6f}      {num:>12.6f}    {abs(grad[name]-num):.1e}")
    print("=> 反向传播 = 链式法则一层层往回乘；解析梯度 ≈ 数值梯度 → 推导正确(gradient check)")

    # ---------------------------------------------------------------
    section("5) 连乘的副作用：梯度消失 / 爆炸")
    for layer_deriv, tag in [(0.5, "每层导数 0.5 (<1)"), (1.5, "每层导数 1.5 (>1)")]:
        print(f"  {tag}:")
        for depth in [5, 20, 50]:
            g = layer_deriv ** depth           # 链式法则：深层梯度 ≈ 各层导数连乘
            print(f"     {depth:>3} 层连乘 = {layer_deriv}^{depth} = {g:.3e}")
    print("=> <1 连乘 → 0 (梯度消失，底层学不动)；>1 连乘 → 爆炸")
    print("   残差连接给梯度开一条「导数=1」的路，让连乘别失控 → 2.10")

    print("\n✅ 全部跑通！链式法则 = 各段导数连乘(串) + 多路径相加(并) = 反向传播的全部原理。")


if __name__ == "__main__":
    main()
