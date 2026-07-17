"""
3.7 期望（expectation）——验证脚本

配套知识点：数学知识点掌握表.md · 3.7
目标：用数值实验看清：
      1) 期望是按概率加权的平均
      2) 如何计算随机变量函数的期望
      3) 期望的线性性质
      4) 如何用样本均值近似理论期望
      5) LLM 训练中的期望 NLL
      6) 强化学习中的期望奖励

运行：
    .venv/bin/python examples/3_7_expectation.py
"""

from __future__ import annotations

from collections.abc import Callable

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


def expected_value(
    values: np.ndarray,
    probabilities: np.ndarray,
) -> float:
    outcomes, weights = validate_distribution(values, probabilities)
    return float(outcomes @ weights)


def expected_function(
    values: np.ndarray,
    probabilities: np.ndarray,
    function: Callable[[np.ndarray], np.ndarray],
) -> float:
    outcomes, weights = validate_distribution(values, probabilities)
    transformed = np.asarray(function(outcomes), dtype=np.float64)
    if transformed.shape != outcomes.shape:
        raise ValueError("function 的输出形状必须与 values 相同")
    if not np.all(np.isfinite(transformed)):
        raise ValueError("function 的输出不能包含 NaN 或 Inf")
    return float(transformed @ weights)


def monte_carlo_expectation(
    values: np.ndarray,
    probabilities: np.ndarray,
    sample_size: int,
    seed: int,
) -> float:
    outcomes, weights = validate_distribution(values, probabilities)
    if sample_size <= 0:
        raise ValueError("sample_size 必须为正整数")

    rng = np.random.default_rng(seed)
    samples = rng.choice(outcomes, size=sample_size, p=weights)
    return float(samples.mean())


def main() -> None:
    np.set_printoptions(precision=6, suppress=True)

    values = np.arange(1, 7, dtype=np.float64)
    probabilities = np.array(
        [0.05, 0.10, 0.15, 0.20, 0.20, 0.30],
        dtype=np.float64,
    )

    # ------------------------------------------------------------------
    section("1) 离散期望：不是普通平均，而是概率加权平均")

    ordinary_mean = float(values.mean())
    die_expectation = expected_value(values, probabilities)

    print("取值：", values)
    print("概率：", probabilities)
    print(f"忽略概率的普通平均：{ordinary_mean:.6f}")
    print(f"按概率加权的期望：  {die_expectation:.6f}")

    assert np.isclose(ordinary_mean, 3.5)
    assert np.isclose(die_expectation, 4.3)
    print("=> 大点数概率更高，所以期望从 3.5 移动到 4.3")

    # ------------------------------------------------------------------
    section("2) 函数的期望：E[g(X)] 不等于 g(E[X])")

    expected_square = expected_function(
        values,
        probabilities,
        np.square,
    )
    square_of_expectation = die_expectation**2

    print(f"E[X^2]：   {expected_square:.6f}")
    print(f"(E[X])^2：{square_of_expectation:.6f}")

    assert np.isclose(expected_square, 20.8)
    assert np.isclose(square_of_expectation, 18.49)
    assert not np.isclose(expected_square, square_of_expectation)
    print("=> 先变换再求期望，通常不同于先求期望再变换")

    # ------------------------------------------------------------------
    section("3) 线性性质：E[aX+b] = aE[X]+b")

    transformed_expectation = expected_function(
        values,
        probabilities,
        lambda x: 2.0 * x + 3.0,
    )
    linearity_result = 2.0 * die_expectation + 3.0

    print(f"直接计算 E[2X+3]：{transformed_expectation:.6f}")
    print(f"使用 2E[X]+3：      {linearity_result:.6f}")

    assert np.isclose(transformed_expectation, 11.6)
    assert np.isclose(transformed_expectation, linearity_result)
    print("=> 期望可以穿过加法，并把常数倍提到外面")

    # ------------------------------------------------------------------
    section("4) Monte Carlo：用样本均值估计理论期望")

    sample_sizes = [10, 1_000, 100_000]
    estimates = [
        monte_carlo_expectation(
            values,
            probabilities,
            sample_size,
            seed=7,
        )
        for sample_size in sample_sizes
    ]

    print(f"理论期望：{die_expectation:.6f}")
    for sample_size, estimate in zip(sample_sizes, estimates):
        error = abs(estimate - die_expectation)
        print(
            f"样本数 {sample_size:>6}："
            f"估计={estimate:.6f}，绝对误差={error:.6f}"
        )

    assert abs(estimates[-1] - die_expectation) < 0.02
    print("=> 有限样本会波动；大量采样的均值通常更接近期望")

    # ------------------------------------------------------------------
    section("5) LLM 训练：平均 NLL 是期望损失的样本估计")

    true_token_probabilities = np.array([0.8, 0.5, 0.2, 0.1])
    data_probabilities = np.array([0.50, 0.30, 0.15, 0.05])
    token_nll = -np.log(true_token_probabilities)
    expected_nll = expected_value(token_nll, data_probabilities)
    equivalent_perplexity = float(np.exp(expected_nll))

    print("真实 token 概率：", true_token_probabilities)
    print("对应 NLL：       ", token_nll)
    print("数据出现概率：   ", data_probabilities)
    print(f"期望 NLL：        {expected_nll:.6f}")
    print(f"对应 PPL：        {equivalent_perplexity:.6f}")

    assert np.isclose(
        expected_nll,
        np.sum(data_probabilities * token_nll),
    )
    print("=> 训练用随机 mini-batch 的平均 loss 近似数据分布上的期望 loss")

    # ------------------------------------------------------------------
    section("6) 强化学习：策略概率加权得到期望奖励")

    responses = np.array(["准确简洁", "正确但冗长", "包含错误"])
    rewards = np.array([1.0, 0.4, -1.0])
    old_policy = np.array([0.50, 0.35, 0.15])
    improved_policy = np.array([0.65, 0.30, 0.05])

    old_expected_reward = expected_value(rewards, old_policy)
    new_expected_reward = expected_value(rewards, improved_policy)

    for response, reward, probability in zip(
        responses,
        rewards,
        old_policy,
    ):
        print(
            f"{response:<10}："
            f"概率={probability:.2f}，奖励={reward:+.2f}"
        )
    print(f"原策略期望奖励：{old_expected_reward:.6f}")
    print(f"新策略期望奖励：{new_expected_reward:.6f}")

    assert np.isclose(old_expected_reward, 0.49)
    assert np.isclose(new_expected_reward, 0.72)
    assert new_expected_reward > old_expected_reward
    print("=> 增加高奖励回答的概率，会提高策略的长期平均奖励")

    section("核心结论")
    print("1. E[X] = sum_x x P(X=x)：期望是概率加权平均")
    print("2. E[g(X)] = sum_x g(x) P(X=x)")
    print("3. E[aX+b] = aE[X]+b，但 E[g(X)] 通常不等于 g(E[X])")
    print("4. 样本均值是理论期望的 Monte Carlo 估计")
    print("5. LLM 训练最小化期望 loss，RL 最大化期望 reward")


if __name__ == "__main__":
    main()
