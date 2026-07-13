"""
2.9 局部最小 / 鞍点 —— 验证脚本

配套知识点：数学知识点掌握表.md · 2.9
目标：用数值实验看清：
      1) 局部最小、全局最小、局部最大和驻点的区别
      2) 一元二阶导数与多元 Hessian 的分类规则
      3) 鞍点在不同方向上具有不同符号的曲率
      4) 梯度下降为何可能停在鞍点，微小扰动为何能帮助逃离
      5) 梯度很小为何不能单独证明已经到达最小值

运行：
    .venv/bin/python examples/2_9_local_minima_saddle_points.py
"""

from collections.abc import Callable

import numpy as np


Vector = np.ndarray
GradientFunction = Callable[[Vector], Vector]


def section(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def one_dimensional_loss(x: float) -> float:
    return x**4 / 4.0 - x**3 / 3.0 - x**2


def one_dimensional_gradient(x: float) -> float:
    return x**3 - x**2 - 2.0 * x


def one_dimensional_second_derivative(x: float) -> float:
    return 3.0 * x**2 - 2.0 * x - 2.0


def classify_hessian(hessian: Vector, tolerance: float = 1e-10) -> tuple[str, Vector]:
    eigenvalues = np.linalg.eigvalsh(hessian)

    if np.all(eigenvalues > tolerance):
        classification = "严格局部最小"
    elif np.all(eigenvalues < -tolerance):
        classification = "严格局部最大"
    elif np.any(eigenvalues > tolerance) and np.any(eigenvalues < -tolerance):
        classification = "鞍点"
    else:
        classification = "二阶检验无法判断（存在平坦方向）"

    return classification, eigenvalues


def saddle_loss(point: Vector) -> float:
    x, y = point
    return float(x**2 + y**4 - y**2)


def saddle_gradient(point: Vector) -> Vector:
    x, y = point
    return np.array([2.0 * x, 4.0 * y**3 - 2.0 * y])


def gradient_descent(
    initial_point: Vector,
    gradient: GradientFunction,
    learning_rate: float,
    steps: int,
) -> Vector:
    point = initial_point.astype(float).copy()
    for _ in range(steps):
        point -= learning_rate * gradient(point)
    return point


def one_dimensional_gradient_descent(
    initial_x: float,
    learning_rate: float,
    steps: int,
) -> float:
    x = initial_x
    for _ in range(steps):
        x -= learning_rate * one_dimensional_gradient(x)
    return x


def main() -> None:
    # ------------------------------------------------------------------
    section("1) 同一函数：局部最小、局部最大与全局最小")

    critical_points = np.array([-1.0, 0.0, 2.0])
    expected_types = ["局部最小", "局部最大", "全局最小"]

    print("临界点      f(x)          f''(x)       分类")
    for point, expected_type in zip(critical_points, expected_types):
        value = one_dimensional_loss(point)
        curvature = one_dimensional_second_derivative(point)
        print(f"{point:>6.1f}   {value:>11.6f}   {curvature:>11.6f}   {expected_type}")
        assert np.isclose(one_dimensional_gradient(point), 0.0)

    assert one_dimensional_second_derivative(-1.0) > 0.0
    assert one_dimensional_second_derivative(0.0) < 0.0
    assert one_dimensional_second_derivative(2.0) > 0.0
    assert one_dimensional_loss(2.0) < one_dimensional_loss(-1.0)
    print("=> 梯度同为 0，仍可能对应不同类型；局部谷底也不一定是最低谷底")

    # ------------------------------------------------------------------
    section("2) f''=0 只能说明二阶检验失效")

    examples = [
        ("x²", 0.0, 2.0, "全局最小"),
        ("-x²", 0.0, -2.0, "全局最大"),
        ("x³", 0.0, 0.0, "平坦拐点"),
        ("x⁴", 0.0, 0.0, "全局最小"),
    ]

    print("函数       f'(0)       f''(0)      实际分类")
    for name, first, second, actual_type in examples:
        print(f"{name:<6}   {first:>8.2f}    {second:>8.2f}      {actual_type}")

    epsilon = 1e-3
    assert (-epsilon) ** 4 > 0.0 and epsilon**4 > 0.0
    assert (-epsilon) ** 3 < 0.0 < epsilon**3
    print("=> x³ 与 x⁴ 在原点的一阶、二阶导数相同，附近函数值却揭示了不同类型")

    # ------------------------------------------------------------------
    section("3) Hessian 特征值：观察各主要方向的曲率")

    hessians = {
        "碗形 x²+y²": np.diag([2.0, 2.0]),
        "山顶 -x²-y²": np.diag([-2.0, -2.0]),
        "鞍形 x²-y²": np.diag([2.0, -2.0]),
        "平坦谷 x²": np.diag([2.0, 0.0]),
    }

    for name, hessian in hessians.items():
        classification, eigenvalues = classify_hessian(hessian)
        formatted = np.array2string(eigenvalues, precision=1)
        print(f"{name:<16} 特征值={formatted:<12} → {classification}")

    assert classify_hessian(hessians["碗形 x²+y²"])[0] == "严格局部最小"
    assert classify_hessian(hessians["山顶 -x²-y²"])[0] == "严格局部最大"
    assert classify_hessian(hessians["鞍形 x²-y²"])[0] == "鞍点"
    assert "无法判断" in classify_hessian(hessians["平坦谷 x²"])[0]
    print("=> 特征值有正有负时，一定存在上弯和下弯方向")

    # ------------------------------------------------------------------
    section("4) 鞍点附近：一个方向上升，另一个方向下降")

    radius = 0.1
    directional_points = {
        "+x 方向": np.array([radius, 0.0]),
        "-x 方向": np.array([-radius, 0.0]),
        "+y 方向": np.array([0.0, radius]),
        "-y 方向": np.array([0.0, -radius]),
    }

    origin_value = saddle_loss(np.zeros(2))
    print(f"原点损失：{origin_value:.6f}")
    for name, point in directional_points.items():
        value = saddle_loss(point)
        relation = "高于原点" if value > origin_value else "低于原点"
        print(f"{name:<8} point={point}  loss={value:>9.6f}  {relation}")

    assert saddle_loss(np.array([radius, 0.0])) > origin_value
    assert saddle_loss(np.array([0.0, radius])) < origin_value
    print("=> 任意小邻域内同时存在更高点和更低点，所以原点不是局部最小")

    # ------------------------------------------------------------------
    section("5) 梯度下降：正好停在鞍点，微小扰动后逃离")

    learning_rate = 0.1
    steps = 100
    starts = {
        "正好在鞍点": np.array([0.0, 0.0]),
        "沿稳定方向": np.array([0.8, 0.0]),
        "带微小扰动": np.array([0.8, 1e-3]),
    }

    results: dict[str, Vector] = {}
    for name, start in starts.items():
        final = gradient_descent(start, saddle_gradient, learning_rate, steps)
        results[name] = final
        print(
            f"{name:<12} start={start} → final={np.round(final, 6)}"
            f"  loss={saddle_loss(final):.6f}"
        )

    target_y = 1.0 / np.sqrt(2.0)
    assert np.allclose(results["正好在鞍点"], np.zeros(2))
    assert np.linalg.norm(results["沿稳定方向"]) < 1e-9
    assert np.isclose(results["带微小扰动"][0], 0.0, atol=1e-9)
    assert np.isclose(results["带微小扰动"][1], target_y, atol=1e-6)
    assert np.isclose(saddle_loss(results["带微小扰动"]), -0.25, atol=1e-10)
    print("=> 精确落在特殊路径上会靠近鞍点；非零扰动会沿负曲率方向逃向真正谷底")

    # ------------------------------------------------------------------
    section("6) 不同初始化可能到达不同深度的谷底")

    initial_values = [-1.8, -0.5, 0.5, 2.8]
    print("初始 x        最终 x        最终 loss")
    final_values: list[float] = []
    for initial_x in initial_values:
        final_x = one_dimensional_gradient_descent(
            initial_x,
            learning_rate=0.05,
            steps=300,
        )
        final_values.append(final_x)
        print(
            f"{initial_x:>7.2f}      {final_x:>9.6f}"
            f"      {one_dimensional_loss(final_x):>11.6f}"
        )

    assert np.allclose(final_values[:2], [-1.0, -1.0], atol=1e-6)
    assert np.allclose(final_values[2:], [2.0, 2.0], atol=1e-6)
    assert one_dimensional_loss(final_values[2]) < one_dimensional_loss(final_values[0])
    print("=> 非凸地形中，初始化所在的吸引区域会影响梯度下降进入哪个谷底")

    # ------------------------------------------------------------------
    section("7) 梯度很小，不足以证明当前位置是最小值")

    near_saddle = np.array([0.0, 1e-10])
    gradient = saddle_gradient(near_saddle)
    gradient_norm = np.linalg.norm(gradient)
    lower_point = np.array([0.0, 0.1])

    print(f"靠近鞍点的位置：{near_saddle}")
    print(f"梯度：{gradient}，梯度范数={gradient_norm:.3e}")
    print(f"当前位置 loss={saddle_loss(near_saddle):.6e}")
    print(f"附近更低点 {lower_point} 的 loss={saddle_loss(lower_point):.6e}")

    assert gradient_norm < 1e-8
    assert saddle_loss(lower_point) < saddle_loss(near_saddle)
    print("=> 停止条件还应结合 loss 轨迹、更新量、验证指标和训练上下文")

    print("\n✅ 全部跑通！记住：梯度为 0 只是候选，曲率和附近函数值才负责分类。")


if __name__ == "__main__":
    main()
