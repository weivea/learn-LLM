"""
2.8 常见函数的导数 —— 验证脚本

配套知识点：数学知识点掌握表.md · 2.8
目标：用数值差分验证常见解析导数，并看清：
      1) 幂函数、指数函数、对数函数的导数
      2) sigmoid、tanh、ReLU、softplus 的导数
      3) sigmoid 饱和区为什么容易产生很小的梯度
      4) exp、log、sigmoid 怎样组合成一个常见损失
      5) 多层很小的本地导数连乘为什么会导致梯度消失

运行：
    .venv/bin/python examples/2_8_common_derivatives.py
"""

import math
from typing import Callable

import numpy as np


ScalarFunction = Callable[[float], float]


def section(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def central_difference(
    function: ScalarFunction,
    x: float,
    h: float = 1e-5,
) -> float:
    return (function(x + h) - function(x - h)) / (2.0 * h)


def sigmoid(x: float) -> float:
    """避免 x 很负时直接计算 exp(-x) 溢出。"""
    if x >= 0:
        inverse = math.exp(-x)
        return 1.0 / (1.0 + inverse)

    exponential = math.exp(x)
    return exponential / (1.0 + exponential)


def sigmoid_derivative(x: float) -> float:
    value = sigmoid(x)
    return value * (1.0 - value)


def relu(x: float) -> float:
    return max(0.0, x)


def relu_derivative(x: float) -> float:
    # ReLU 在 0 处不可导；这里采用深度学习框架常见的返回 0 约定。
    return 1.0 if x > 0 else 0.0


def softplus(x: float) -> float:
    """稳定计算 log(1 + exp(x))。"""
    return max(0.0, x) + math.log1p(math.exp(-abs(x)))


def binary_cross_entropy_from_logit(logit: float, target: float) -> float:
    """稳定计算 BCE：softplus(z) - y*z。"""
    return softplus(logit) - target * logit


def main() -> None:
    # ------------------------------------------------------------------
    section("1) 解析导数与中心差分一致")

    cases: list[tuple[str, ScalarFunction, ScalarFunction, float]] = [
        ("x²", lambda x: x**2, lambda x: 2.0 * x, 3.0),
        ("x³", lambda x: x**3, lambda x: 3.0 * x**2, -2.0),
        ("1/x", lambda x: 1.0 / x, lambda x: -1.0 / x**2, 2.0),
        ("sqrt(x)", math.sqrt, lambda x: 1.0 / (2.0 * math.sqrt(x)), 4.0),
        ("exp(x)", math.exp, math.exp, 0.7),
        ("ln(x)", math.log, lambda x: 1.0 / x, 2.0),
        ("sigmoid(x)", sigmoid, sigmoid_derivative, 1.2),
        ("tanh(x)", math.tanh, lambda x: 1.0 - math.tanh(x) ** 2, -0.8),
        ("softplus(x)", softplus, sigmoid, 0.6),
    ]

    print("函数            x       解析导数       数值导数       绝对误差")
    for name, function, derivative, x in cases:
        analytic = derivative(x)
        numeric = central_difference(function, x)
        error = abs(analytic - numeric)
        print(f"{name:<14} {x:>5.1f}   {analytic:>11.7f}   {numeric:>11.7f}   {error:.1e}")
        assert np.isclose(analytic, numeric, atol=1e-8)

    print("=> 公式负责高效求梯度，中心差分可以检查公式或自动求导实现")

    # ------------------------------------------------------------------
    section("2) 幂法则：d(x^n)/dx = n*x^(n-1)")

    x = 2.0
    exponents = [3.0, 2.0, 1.0, 0.0, -1.0, 0.5]

    print("n        x^n         n*x^(n-1)      数值导数")
    for exponent in exponents:
        function = lambda value, n=exponent: value**n
        analytic = exponent * x ** (exponent - 1.0)
        numeric = central_difference(function, x)
        print(f"{exponent:>4.1f}   {function(x):>10.6f}   {analytic:>12.6f}   {numeric:>12.6f}")
        assert np.isclose(analytic, numeric, atol=1e-8)

    print("=> 指数可以是正数、0、负数或分数，但必须留意函数定义域")

    # ------------------------------------------------------------------
    section("3) exp 与 ln 互为反函数，链式求导后导数为 1")

    exp_input = 1.3
    log_input = 2.5

    log_after_exp = lambda value: math.log(math.exp(value))
    exp_after_log = lambda value: math.exp(math.log(value))

    first_numeric = central_difference(log_after_exp, exp_input)
    second_numeric = central_difference(exp_after_log, log_input)

    print(f"ln(exp(x)) 在 x={exp_input}：函数值={log_after_exp(exp_input):.6f}，导数≈{first_numeric:.6f}")
    print(f"exp(ln(x)) 在 x={log_input}：函数值={exp_after_log(log_input):.6f}，导数≈{second_numeric:.6f}")

    assert np.isclose(log_after_exp(exp_input), exp_input)
    assert np.isclose(exp_after_log(log_input), log_input)
    assert np.isclose(first_numeric, 1.0, atol=1e-8)
    assert np.isclose(second_numeric, 1.0, atol=1e-8)
    print("=> (ln(exp(x)))'=(1/exp(x))*exp(x)=1；反函数的局部缩放互相抵消")

    # ------------------------------------------------------------------
    section("4) sigmoid' = sigmoid*(1-sigmoid)：中间最大，两端趋近 0")

    sigmoid_inputs = [-10.0, -4.0, 0.0, 4.0, 10.0]
    sigmoid_gradients: list[float] = []

    print("x          sigmoid(x)       sigmoid'(x)")
    for value in sigmoid_inputs:
        output = sigmoid(value)
        gradient = sigmoid_derivative(value)
        sigmoid_gradients.append(gradient)
        print(f"{value:>5.1f}       {output:>10.7f}       {gradient:>11.8f}")

    assert np.isclose(sigmoid_derivative(0.0), 0.25)
    assert max(sigmoid_gradients) == sigmoid_derivative(0.0)
    assert sigmoid_derivative(-10.0) < 1e-4
    assert sigmoid_derivative(10.0) < 1e-4
    print("=> sigmoid 的最大导数只有 1/4；进入饱和区后，传回去的梯度会非常小")

    # ------------------------------------------------------------------
    section("5) 常见激活函数：ReLU、tanh 与 softplus")

    activation_inputs = [-3.0, -1.0, 0.0, 1.0, 3.0]

    print("x       ReLU'约定      tanh'        softplus'=sigmoid")
    for value in activation_inputs:
        tanh_gradient = 1.0 - math.tanh(value) ** 2
        print(
            f"{value:>4.1f}      {relu_derivative(value):>7.3f}"
            f"       {tanh_gradient:>9.6f}       {sigmoid(value):>12.7f}"
        )

        if value != 0.0:
            assert np.isclose(
                relu_derivative(value),
                central_difference(relu, value),
                atol=1e-8,
            )
        assert np.isclose(
            tanh_gradient,
            central_difference(math.tanh, value),
            atol=1e-8,
        )
        assert np.isclose(
            sigmoid(value),
            central_difference(softplus, value),
            atol=1e-8,
        )

    relu_at_zero_numeric = central_difference(relu, 0.0)
    print(f"\nReLU 在 0 处：左导数=0，右导数=1，中心差分={relu_at_zero_numeric:.1f}")
    assert np.isclose(relu_at_zero_numeric, 0.5)
    print("=> ReLU 在 0 处不可导；框架可约定一个次梯度，不能把约定误当成普通导数")

    # ------------------------------------------------------------------
    section("6) 一个完整组合：logit → sigmoid → 交叉熵 → 参数梯度")

    x = 2.0
    target = 1.0
    w = -0.5
    b = 0.2
    logit = w * x + b
    probability = sigmoid(logit)
    loss = binary_cross_entropy_from_logit(logit, target)

    # 对 BCE(sigmoid(z), y)，dL/dz = sigmoid(z)-y。
    grad_logit = probability - target
    grad_w = grad_logit * x
    grad_b = grad_logit

    loss_as_function_of_w = lambda candidate_w: binary_cross_entropy_from_logit(
        candidate_w * x + b,
        target,
    )
    loss_as_function_of_b = lambda candidate_b: binary_cross_entropy_from_logit(
        w * x + candidate_b,
        target,
    )
    numeric_grad_w = central_difference(loss_as_function_of_w, w)
    numeric_grad_b = central_difference(loss_as_function_of_b, b)

    print(f"z=w*x+b={logit:.6f}")
    print(f"p=sigmoid(z)={probability:.6f}")
    print(f"L=-ln(p)={loss:.6f}")
    print(f"dL/dz=p-y={grad_logit:.6f}")
    print(f"dL/dw=(p-y)*x={grad_w:.6f}，数值检查={numeric_grad_w:.6f}")
    print(f"dL/db=p-y={grad_b:.6f}，数值检查={numeric_grad_b:.6f}")

    assert np.isclose(loss, -math.log(probability))
    assert np.isclose(grad_w, numeric_grad_w, atol=1e-8)
    assert np.isclose(grad_b, numeric_grad_b, atol=1e-8)

    learning_rate = 0.1
    new_w = w - learning_rate * grad_w
    new_b = b - learning_rate * grad_b
    new_logit = new_w * x + new_b
    new_loss = binary_cross_entropy_from_logit(new_logit, target)

    print(f"一步梯度下降后：w={new_w:.6f}，b={new_b:.6f}，loss={new_loss:.6f}")
    assert new_loss < loss
    print("=> 常见导数通过链式法则组合，最终给出模型参数的更新方向")

    # ------------------------------------------------------------------
    section("7) 很小的本地导数连续相乘：梯度快速衰减")

    depths = [1, 2, 4, 8, 16]
    center_local_gradient = sigmoid_derivative(0.0)
    saturated_local_gradient = sigmoid_derivative(5.0)

    print("层数      每层导数=0.25       每层导数=sigmoid'(5)")
    for depth in depths:
        center_product = center_local_gradient**depth
        saturated_product = saturated_local_gradient**depth
        print(f"{depth:>3}       {center_product:>14.6e}       {saturated_product:>20.6e}")

    assert center_local_gradient**16 < 1e-9
    assert saturated_local_gradient**8 < 1e-16
    print("=> 路径内导数相乘会把梯度压得很小，这正是 2.10 梯度消失的数学入口")

    print("\n✅ 全部跑通！先记住函数的本地导数，再用求导规则和链式法则把它们组合起来。")


if __name__ == "__main__":
    main()
