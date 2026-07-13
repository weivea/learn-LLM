"""
2.10 梯度消失 / 爆炸 —— 验证脚本

配套知识点：数学知识点掌握表.md · 2.10
目标：用数值实验看清：
      1) 长链式法则为何产生指数级缩小或放大
      2) sigmoid 饱和为何容易导致梯度消失
      3) 矩阵奇异值如何影响梯度范数
      4) 残差恒等路径为何有助于保留梯度
      5) 梯度范数裁剪怎样限制爆炸但不修复消失
      6) Xavier / He 初始化如何影响 ReLU 网络的信号尺度
      7) 极端梯度在 float32 中如何下溢或溢出

运行：
    .venv/bin/python examples/2_10_vanishing_exploding_gradients.py
"""

import numpy as np


def section(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def repeated_scalar_gradient(local_derivative: float, depth: int) -> float:
    gradient = 1.0
    for _ in range(depth):
        gradient *= local_derivative
    return gradient


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + np.exp(-value))


def sigmoid_derivative(value: float) -> float:
    activation = sigmoid(value)
    return float(activation * (1.0 - activation))


def propagate_matrix_gradient(scale: float, depth: int) -> float:
    rotation = np.array([[0.0, -1.0], [1.0, 0.0]])
    weight = scale * rotation
    gradient = np.array([1.0, 0.0])

    for _ in range(depth):
        gradient = weight.T @ gradient

    return float(np.linalg.norm(gradient))


def clip_gradient_by_norm(
    gradient: np.ndarray,
    max_norm: float,
) -> tuple[np.ndarray, float]:
    norm = float(np.linalg.norm(gradient))
    scale = min(1.0, max_norm / norm) if norm > 0.0 else 1.0
    return gradient * scale, norm


def root_mean_square(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(values**2)))


def simulate_relu_stack(
    weight_std: float,
    *,
    width: int,
    depth: int,
    batch_size: int,
    seed: int,
) -> tuple[list[float], list[float]]:
    rng = np.random.default_rng(seed)
    activations = rng.standard_normal((batch_size, width))
    activation_rms = [root_mean_square(activations)]
    weights: list[np.ndarray] = []
    masks: list[np.ndarray] = []

    for _ in range(depth):
        weight = rng.normal(0.0, weight_std, size=(width, width))
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            pre_activation = activations @ weight
        if not np.all(np.isfinite(pre_activation)):
            raise FloatingPointError("ReLU 前向实验出现非有限值")
        activations = np.maximum(pre_activation, 0.0)
        weights.append(weight)
        masks.append(pre_activation > 0.0)
        activation_rms.append(root_mean_square(activations))

    gradient = rng.standard_normal(activations.shape)
    gradient_rms = [root_mean_square(gradient)]

    for weight, mask in zip(reversed(weights), reversed(masks)):
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            gradient = (gradient * mask) @ weight.T
        if not np.all(np.isfinite(gradient)):
            raise FloatingPointError("ReLU 反向实验出现非有限值")
        gradient_rms.append(root_mean_square(gradient))

    return activation_rms, gradient_rms


def repeated_float32_product(multiplier: float, depth: int) -> np.float32:
    result = np.float32(1.0)
    factor = np.float32(multiplier)

    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        for _ in range(depth):
            result = np.float32(result * factor)

    return result


def main() -> None:
    # ------------------------------------------------------------------
    section("1) 标量长连乘：深度让缩小 / 放大呈指数累积")

    derivatives = [0.5, 1.0, 1.5]
    depths = [10, 50]

    print("局部导数      10 层后的梯度       50 层后的梯度")
    for derivative in derivatives:
        values = [
            repeated_scalar_gradient(derivative, depth)
            for depth in depths
        ]
        print(f"{derivative:>8.2f}      {values[0]:>14.6e}      {values[1]:>14.6e}")

    assert np.isclose(repeated_scalar_gradient(0.5, 50), 0.5**50)
    assert repeated_scalar_gradient(0.5, 50) < 1e-15
    assert repeated_scalar_gradient(1.5, 50) > 1e8
    assert repeated_scalar_gradient(1.0, 50) == 1.0
    print("=> 同一个局部缩放反复相乘后，会变成消失、稳定或爆炸三种状态")

    # ------------------------------------------------------------------
    section("2) sigmoid：即使在最佳位置，导数连乘也会很快缩小")

    print("输入 z       sigmoid'(z)       连乘 20 次")
    for value in [0.0, 2.0, 5.0]:
        derivative = sigmoid_derivative(value)
        product = repeated_scalar_gradient(derivative, 20)
        print(f"{value:>6.1f}       {derivative:>12.6e}       {product:>12.6e}")

    maximum_derivative = sigmoid_derivative(0.0)
    saturated_derivative = sigmoid_derivative(5.0)
    assert np.isclose(maximum_derivative, 0.25)
    assert saturated_derivative < 0.01
    assert repeated_scalar_gradient(maximum_derivative, 20) < 1e-12

    tanh_center_derivative = 1.0 - np.tanh(0.0) ** 2
    tanh_saturated_derivative = 1.0 - np.tanh(5.0) ** 2
    print(f"tanh'(0)={tanh_center_derivative:.6f}")
    print(f"tanh'(5)={tanh_saturated_derivative:.6e}")
    assert np.isclose(tanh_center_derivative, 1.0)
    assert tanh_saturated_derivative < 1e-3
    print("=> sigmoid 总会缩小梯度；tanh 中心较稳定，但进入饱和区也会接近 0")

    # ------------------------------------------------------------------
    section("3) 矩阵反向传播：奇异值控制梯度范数的缩放")

    matrix_scales = [0.8, 1.0, 1.2]
    matrix_depth = 50

    print("最大奇异值      50 层后梯度范数      理论值")
    for scale in matrix_scales:
        actual_norm = propagate_matrix_gradient(scale, matrix_depth)
        expected_norm = scale**matrix_depth
        print(f"{scale:>10.2f}      {actual_norm:>16.6e}      {expected_norm:>12.6e}")
        assert np.isclose(actual_norm, expected_norm, rtol=1e-12)

    assert propagate_matrix_gradient(0.8, matrix_depth) < 1e-4
    assert np.isclose(propagate_matrix_gradient(1.0, matrix_depth), 1.0)
    assert propagate_matrix_gradient(1.2, matrix_depth) > 1e3
    print("=> 本实验展示统一缩放；一般 Jacobian 还会用不同奇异值缩放不同方向")

    # ------------------------------------------------------------------
    section("4) 残差路径：从乘 F' 变成乘 (1 + F')")

    depth = 50
    branch_derivative = 0.02
    plain_gradient = repeated_scalar_gradient(branch_derivative, depth)
    residual_gradient = repeated_scalar_gradient(1.0 + branch_derivative, depth)
    saturated_plain = repeated_scalar_gradient(0.0, depth)
    saturated_residual = repeated_scalar_gradient(1.0, depth)

    print(f"普通 50 层：        0.02^50  = {plain_gradient:.6e}")
    print(f"残差 50 层：        1.02^50  = {residual_gradient:.6e}")
    print(f"分支导数为 0，普通路径梯度：{saturated_plain:.1f}")
    print(f"分支导数为 0，残差路径梯度：{saturated_residual:.1f}")

    assert plain_gradient < 1e-80
    assert saturated_plain == 0.0
    assert saturated_residual == 1.0
    assert residual_gradient > 1.0
    print("=> 恒等项保留直达梯度，但 (1 + F') 连乘仍可能增长，残差不是万能保险")

    # ------------------------------------------------------------------
    section("5) 梯度范数裁剪：保留方向，把长度限制到阈值")

    gradient = np.array([30.0, 40.0])
    clipped, original_norm = clip_gradient_by_norm(gradient, max_norm=5.0)
    clipped_norm = float(np.linalg.norm(clipped))

    print(f"原梯度：{gradient}，范数={original_norm:.1f}")
    print(f"裁剪后：{clipped}，范数={clipped_norm:.1f}")

    assert np.allclose(clipped, np.array([3.0, 4.0]))
    assert np.isclose(clipped_norm, 5.0)
    assert np.allclose(
        clipped / clipped_norm,
        gradient / original_norm,
    )

    small_gradient = np.array([0.3, 0.4])
    unchanged, small_norm = clip_gradient_by_norm(small_gradient, max_norm=5.0)
    assert np.isclose(small_norm, 0.5)
    assert np.array_equal(unchanged, small_gradient)
    print("=> 裁剪只缩小过大的梯度；原本很小的梯度不会被放大")

    # ------------------------------------------------------------------
    section("6) ReLU 网络初始化：Xavier 会衰减，He 更能保持尺度")

    width = 64
    relu_depth = 12
    batch_size = 512
    configurations = {
        "Xavier": 1.0 / np.sqrt(width),
        "He": np.sqrt(2.0 / width),
        "过大 std=0.5": 0.5,
    }
    simulation_results: dict[str, tuple[list[float], list[float]]] = {}

    print("初始化          激活 RMS: 输入→输出          梯度 RMS: 输出→输入")
    for index, (name, weight_std) in enumerate(configurations.items()):
        activation_rms, gradient_rms = simulate_relu_stack(
            weight_std,
            width=width,
            depth=relu_depth,
            batch_size=batch_size,
            seed=2026 + index,
        )
        simulation_results[name] = activation_rms, gradient_rms
        print(
            f"{name:<12}"
            f" {activation_rms[0]:>9.3e}→{activation_rms[-1]:>9.3e}"
            f"       {gradient_rms[0]:>9.3e}→{gradient_rms[-1]:>9.3e}"
        )

    xavier_activation, xavier_gradient = simulation_results["Xavier"]
    he_activation, he_gradient = simulation_results["He"]
    large_activation, large_gradient = simulation_results["过大 std=0.5"]

    assert xavier_activation[-1] < 0.1 * xavier_activation[0]
    assert xavier_gradient[-1] < 0.1 * xavier_gradient[0]
    assert 0.2 < he_activation[-1] / he_activation[0] < 3.0
    assert 0.2 < he_gradient[-1] / he_gradient[0] < 3.0
    assert large_activation[-1] > 1e3 * large_activation[0]
    assert large_gradient[-1] > 1e3 * large_gradient[0]
    print("=> ReLU 截掉约一半信号，He 的 2/fan_in 方差用于补偿这种尺度损失")

    # ------------------------------------------------------------------
    section("7) float32 数值范围：极小梯度下溢，极大梯度溢出")

    tiny_gradient = repeated_float32_product(0.5, 200)
    stable_gradient = repeated_float32_product(1.0, 200)
    huge_gradient = repeated_float32_product(2.0, 200)

    print(f"float32: 0.5^200 → {tiny_gradient}")
    print(f"float32: 1.0^200 → {stable_gradient}")
    print(f"float32: 2.0^200 → {huge_gradient}")

    assert tiny_gradient == np.float32(0.0)
    assert stable_gradient == np.float32(1.0)
    assert np.isinf(huge_gradient)
    print("=> 数学上的极小 / 极大梯度还会进一步触发有限精度的下溢 / 溢出")

    print(
        "\n✅ 全部跑通！记住：长 Jacobian 连乘决定梯度尺度；"
        "残差、归一化、初始化和裁剪各自处理不同环节。"
    )


if __name__ == "__main__":
    main()
