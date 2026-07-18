"""
3.8 方差 / 标准差（variance / standard deviation）——验证脚本

配套知识点：数学知识点掌握表.md · 3.8
目标：用数值实验看清：
      1) 方差如何衡量数据围绕期望的波动
      2) Var(X) = E[X^2] - (E[X])^2
      3) 平移、缩放怎样改变方差与标准差
      4) 总体方差与样本方差的区别
      5) LayerNorm 为什么沿最后一维标准化
      6) 参数初始化为什么需要控制方差
      7) batch size 怎样影响平均损失的波动

运行：
    .venv/bin/python examples/3_8_variance_standard_deviation.py
"""

from __future__ import annotations

import numpy as np


def section(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def validate_distribution(
    values: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    outcomes = np.asarray(values, dtype=np.float64)
    weights = np.asarray(probabilities, dtype=np.float64)

    if outcomes.ndim != 1 or outcomes.size == 0:
        raise ValueError("values 必须是一维非空数组")
    if weights.ndim != 1 or weights.size == 0:
        raise ValueError("probabilities 必须是一维非空数组")
    if outcomes.shape != weights.shape:
        raise ValueError("values 与 probabilities 的形状必须相同")
    if not np.all(np.isfinite(outcomes)):
        raise ValueError("values 不能包含 NaN 或 Inf")
    if not np.all(np.isfinite(weights)):
        raise ValueError("probabilities 不能包含 NaN 或 Inf")
    if np.any(weights < 0.0):
        raise ValueError("概率不能为负数")
    if not np.isclose(weights.sum(), 1.0):
        raise ValueError("概率之和必须为 1")

    return outcomes, weights


def weighted_mean(
    values: np.ndarray,
    probabilities: np.ndarray,
) -> float:
    outcomes, weights = validate_distribution(values, probabilities)
    return float(outcomes @ weights)


def weighted_variance(
    values: np.ndarray,
    probabilities: np.ndarray,
) -> float:
    outcomes, weights = validate_distribution(values, probabilities)
    mean = float(outcomes @ weights)
    squared_deviations = (outcomes - mean) ** 2
    return float(squared_deviations @ weights)


def standardize_last_dim(
    values: np.ndarray,
    eps: float = 1e-5,
) -> np.ndarray:
    data = np.asarray(values, dtype=np.float64)
    if data.ndim == 0 or data.size == 0:
        raise ValueError("values 至少需要一维非空输入")
    if not np.all(np.isfinite(data)):
        raise ValueError("values 不能包含 NaN 或 Inf")
    if not np.isfinite(eps) or eps < 0.0:
        raise ValueError("eps 必须是非负有限数")

    mean = data.mean(axis=-1, keepdims=True)
    centered = data - mean
    variance = np.mean(centered**2, axis=-1, keepdims=True)
    denominator = np.sqrt(variance + eps)
    if np.any(denominator == 0.0):
        raise ValueError("方差为 0 时 eps 必须大于 0")
    return centered / denominator


def main() -> None:
    np.set_printoptions(precision=6, suppress=True)

    values = np.arange(1, 7, dtype=np.float64)
    probabilities = np.array(
        [0.05, 0.10, 0.15, 0.20, 0.20, 0.30],
        dtype=np.float64,
    )

    # ------------------------------------------------------------------
    section("1) 从期望到方差：平方偏差的概率加权平均")

    mean = weighted_mean(values, probabilities)
    variance = weighted_variance(values, probabilities)
    standard_deviation = float(np.sqrt(variance))

    print("取值：", values)
    print("概率：", probabilities)
    print(f"期望 E[X]：         {mean:.6f}")
    print(f"方差 Var(X)：       {variance:.6f}")
    print(f"标准差 Std(X)：     {standard_deviation:.6f}")

    assert np.isclose(mean, 4.3)
    assert np.isclose(variance, 2.31)
    assert np.isclose(standard_deviation, np.sqrt(2.31))
    print("=> 中心在 4.3，结果围绕中心的波动尺度约为 1.52")

    # ------------------------------------------------------------------
    section("2) 等价公式与浮点精度：优先计算中心化后的平方")

    expected_square = weighted_mean(values**2, probabilities)
    variance_from_identity = expected_square - mean**2

    print(f"E[X^2]：                    {expected_square:.6f}")
    print(f"(E[X])^2：                  {mean**2:.6f}")
    print(f"E[X^2] - (E[X])^2：         {variance_from_identity:.6f}")
    print(f"E[(X-E[X])^2]：             {variance:.6f}")

    assert np.isclose(expected_square, 20.8)
    assert np.isclose(variance_from_identity, variance)

    large_values = np.array(
        [1e12 + 1.0, 1e12 + 2.0, 1e12 + 3.0, 1e12 + 4.0]
    )
    stable_variance = float(
        np.mean((large_values - large_values.mean()) ** 2)
    )
    cancellation_prone_variance = float(
        np.mean(large_values**2) - large_values.mean() ** 2
    )

    print("\n大偏移、小波动的数据：", large_values)
    print(f"先中心化计算：              {stable_variance:.6f}")
    print(
        "两个大数相减计算：          "
        f"{cancellation_prone_variance:.6f}"
    )

    assert np.isclose(stable_variance, 1.25)
    print("=> 两式数学上等价，但直接用库函数或中心化计算更稳定")

    # ------------------------------------------------------------------
    section("3) 平移与缩放：Var(aX+b) = a^2 Var(X)")

    shifted_values = values + 100.0
    transformed_values = 2.0 * values + 3.0

    shifted_variance = weighted_variance(
        shifted_values,
        probabilities,
    )
    transformed_variance = weighted_variance(
        transformed_values,
        probabilities,
    )
    transformed_std = float(np.sqrt(transformed_variance))

    print(f"Var(X)：              {variance:.6f}")
    print(f"Var(X+100)：          {shifted_variance:.6f}")
    print(f"Var(2X+3)：           {transformed_variance:.6f}")
    print(f"4 Var(X)：            {4.0 * variance:.6f}")
    print(f"Std(2X+3)：           {transformed_std:.6f}")
    print(f"2 Std(X)：            {2.0 * standard_deviation:.6f}")

    assert np.isclose(shifted_variance, variance)
    assert np.isclose(transformed_variance, 4.0 * variance)
    assert np.isclose(
        transformed_std,
        2.0 * standard_deviation,
    )
    print("=> 加常数只移动中心；乘 2 使标准差乘 2、方差乘 4")

    # ------------------------------------------------------------------
    section("4) 总体方差与样本方差：分母 N 和 N-1")

    observations = np.array([2.0, 4.0, 6.0])
    population_variance = float(np.var(observations, ddof=0))
    sample_variance = float(np.var(observations, ddof=1))
    population_std = float(np.std(observations, ddof=0))
    sample_std = float(np.std(observations, ddof=1))

    print("数据：", observations)
    print(f"总体方差，ddof=0：{population_variance:.6f}")
    print(f"样本方差，ddof=1：{sample_variance:.6f}")
    print(f"总体标准差：       {population_std:.6f}")
    print(f"样本标准差：       {sample_std:.6f}")

    assert np.isclose(population_variance, 8.0 / 3.0)
    assert np.isclose(sample_variance, 4.0)
    assert np.isclose(population_std, np.sqrt(8.0 / 3.0))
    assert np.isclose(sample_std, 2.0)
    print("=> NumPy 默认 ddof=0；估计未知总体方差时常使用 ddof=1")

    # ------------------------------------------------------------------
    section("5) LayerNorm 直觉：每个 token 沿 d_model 标准化")

    hidden_states = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [4.0, 4.0, 10.0, 2.0],
        ]
    )
    normalized = standardize_last_dim(hidden_states, eps=0.0)

    print("hidden shape：", hidden_states.shape)
    print("标准化前：")
    print(hidden_states)
    print("标准化后：")
    print(normalized)
    print("每个 token 的均值：", normalized.mean(axis=-1))
    print("每个 token 的方差：", normalized.var(axis=-1))

    assert np.allclose(normalized.mean(axis=-1), 0.0)
    assert np.allclose(normalized.var(axis=-1), 1.0)
    print("=> shape=(tokens, d_model) 时，每一行独立沿最后一维归一化")

    # ------------------------------------------------------------------
    section("6) 参数初始化：让线性层前后的方差大致保持")

    rng = np.random.default_rng(7)
    sample_count = 20_000
    fan_in = 64
    fan_out = 64
    inputs = rng.normal(
        loc=0.0,
        scale=1.0,
        size=(sample_count, fan_in),
    )
    input_variance = float(np.var(inputs))
    weight_stds = [
        0.01,
        np.sqrt(1.0 / fan_in),
        0.50,
    ]
    labels = ["过小", "保持尺度", "过大"]
    observed_variances: list[float] = []

    print(f"输入方差：{input_variance:.6f}")
    for label, weight_std in zip(labels, weight_stds):
        weights = rng.normal(
            loc=0.0,
            scale=weight_std,
            size=(fan_in, fan_out),
        )
        outputs = np.empty((sample_count, fan_out))
        for output_index in range(fan_out):
            outputs[:, output_index] = np.sum(
                inputs * weights[:, output_index],
                axis=-1,
            )
        observed = float(np.var(outputs))
        predicted = fan_in * weight_std**2 * input_variance
        observed_variances.append(observed)
        print(
            f"{label:<8} weight_std={weight_std:.6f}，"
            f"预测输出方差={predicted:.6f}，"
            f"实测={observed:.6f}"
        )

    assert observed_variances[0] < observed_variances[1]
    assert observed_variances[1] < observed_variances[2]
    assert 0.80 < observed_variances[1] < 1.20
    print("=> weight variance 约为 1/fan_in 时，输出尺度大致不变")

    # ------------------------------------------------------------------
    section("7) batch 均值：batch 越大，随机估计通常越稳定")

    loss_mean = 2.0
    loss_std = 2.0
    loss_variance = loss_std**2
    batch_sizes = [4, 16, 64]
    trial_count = 20_000

    for batch_size in batch_sizes:
        batch_losses = rng.normal(
            loc=loss_mean,
            scale=loss_std,
            size=(trial_count, batch_size),
        )
        batch_means = batch_losses.mean(axis=-1)
        observed = float(np.var(batch_means))
        predicted = loss_variance / batch_size

        print(
            f"B={batch_size:>2}："
            f"理论 Var(batch_mean)={predicted:.6f}，"
            f"模拟={observed:.6f}"
        )
        assert np.isclose(observed, predicted, rtol=0.06)

    print("=> 在独立同分布假设下，Var(batch_mean) = Var(loss) / B")

    section("核心结论")
    print("1. Var(X) = E[(X-E[X])^2]：方差衡量围绕期望的波动")
    print("2. Std(X) = sqrt(Var(X))：标准差与原数据单位相同")
    print("3. Var(aX+b) = a^2 Var(X)：平移不变，缩放平方变化")
    print("4. NumPy 默认 ddof=0；样本方差常使用 ddof=1")
    print("5. LayerNorm 沿 d_model 控制每个 token 的激活值尺度")
    print("6. 初始化通过控制权重方差，避免信号逐层过小或过大")
    print("7. Var(batch_mean) = variance / B：大 batch 通常更稳定")


if __name__ == "__main__":
    main()
