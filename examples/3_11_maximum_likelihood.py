"""
3.11 最大似然估计（MLE）——验证脚本

配套知识点：数学知识点掌握表.md · 3.11
目标：用数值实验看清：
      1) 概率固定数据后如何成为关于参数的似然
      2) Bernoulli MLE 为什么等于样本中的正面频率
      3) 对数似然如何避免概率乘积下溢
      4) 类别分布 MLE 为什么等于经验频率
      5) 固定方差高斯 MLE 为什么等价于最小化 MSE
      6) LLM 的最大似然如何等价于最小化 token NLL
      7) 梯度上升 log-likelihood 与梯度下降 NLL 为什么一致

运行：
    .venv/bin/python examples/3_11_maximum_likelihood.py
"""

from __future__ import annotations

from numbers import Integral

import numpy as np


def section(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def validate_binary_observations(
    observations: np.ndarray,
) -> np.ndarray:
    values = np.asarray(observations, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("observations 必须是一维非空数组")
    if not np.all(np.isfinite(values)):
        raise ValueError("observations 不能包含 NaN 或 Inf")
    if not np.all((values == 0.0) | (values == 1.0)):
        raise ValueError("observations 只能包含 0 或 1")
    return values


def bernoulli_log_likelihood(
    theta: float,
    observations: np.ndarray,
) -> float:
    probability = float(theta)
    if not np.isfinite(probability):
        raise ValueError("theta 必须是有限数")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("theta 必须满足 0 <= theta <= 1")

    values = validate_binary_observations(observations)
    heads = int(values.sum())
    tails = values.size - heads

    if heads > 0 and probability == 0.0:
        return -np.inf
    if tails > 0 and probability == 1.0:
        return -np.inf

    log_likelihood = 0.0
    if heads > 0:
        log_likelihood += heads * np.log(probability)
    if tails > 0:
        log_likelihood += tails * np.log1p(-probability)
    return float(log_likelihood)


def bernoulli_mle(observations: np.ndarray) -> float:
    values = validate_binary_observations(observations)
    return float(values.mean())


def categorical_mle(
    labels: np.ndarray,
    category_count: int,
) -> np.ndarray:
    values = np.asarray(labels)
    if isinstance(category_count, bool) or not isinstance(
        category_count,
        Integral,
    ):
        raise ValueError("category_count 必须是整数")
    if category_count <= 0:
        raise ValueError("category_count 必须大于 0")
    if values.ndim != 1 or values.size == 0:
        raise ValueError("labels 必须是一维非空数组")
    if not np.issubdtype(values.dtype, np.integer):
        raise ValueError("labels 必须包含整数类别索引")
    if np.any(values < 0) or np.any(values >= category_count):
        raise ValueError("labels 必须满足 0 <= label < category_count")

    counts = np.bincount(values, minlength=category_count)
    return counts.astype(np.float64) / values.size


def mean_token_nll(target_probabilities: np.ndarray) -> float:
    probabilities = np.asarray(
        target_probabilities,
        dtype=np.float64,
    )
    if probabilities.ndim != 1 or probabilities.size == 0:
        raise ValueError(
            "target_probabilities 必须是一维非空数组"
        )
    if not np.all(np.isfinite(probabilities)):
        raise ValueError(
            "target_probabilities 不能包含 NaN 或 Inf"
        )
    if np.any(probabilities <= 0.0) or np.any(
        probabilities > 1.0
    ):
        raise ValueError(
            "每个目标 token 概率必须满足 0 < p <= 1"
        )
    return float(-np.log(probabilities).mean())


def gaussian_log_likelihood(
    mean: float,
    observations: np.ndarray,
    standard_deviation: float,
) -> float:
    values = np.asarray(observations, dtype=np.float64)
    candidate_mean = float(mean)
    sigma = float(standard_deviation)

    if values.ndim != 1 or values.size == 0:
        raise ValueError("observations 必须是一维非空数组")
    if not np.all(np.isfinite(values)):
        raise ValueError("observations 不能包含 NaN 或 Inf")
    if not np.isfinite(candidate_mean):
        raise ValueError("mean 必须是有限数")
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("standard_deviation 必须是正有限数")

    residuals = values - candidate_mean
    return float(
        -values.size * np.log(sigma * np.sqrt(2.0 * np.pi))
        - np.sum(residuals**2) / (2.0 * sigma**2)
    )


def sigmoid(value: float) -> float:
    scalar = float(value)
    if not np.isfinite(scalar):
        raise ValueError("value 必须是有限数")
    if scalar >= 0.0:
        return float(1.0 / (1.0 + np.exp(-scalar)))
    exp_value = np.exp(scalar)
    return float(exp_value / (1.0 + exp_value))


def main() -> None:
    np.set_printoptions(precision=6, suppress=True)

    observations = np.array(
        [1, 1, 0, 1, 0, 1, 1, 1, 0, 1],
        dtype=np.float64,
    )
    candidates = np.array([0.5, 0.7, 0.9])

    # ------------------------------------------------------------------
    section("1) 固定数据，比较候选参数的似然")

    log_likelihoods = np.array(
        [
            bernoulli_log_likelihood(theta, observations)
            for theta in candidates
        ]
    )
    likelihoods = np.exp(log_likelihoods)

    print("观测序列：", observations.astype(int))
    print("候选 theta：", candidates)
    print("序列似然：", likelihoods)
    print("对数似然：", log_likelihoods)
    print("候选中的 argmax：", candidates[np.argmax(likelihoods)])

    assert np.allclose(
        likelihoods,
        [0.5**10, 0.7**7 * 0.3**3, 0.9**7 * 0.1**3],
    )
    assert candidates[np.argmax(likelihoods)] == 0.7
    print("=> 数据固定后，似然把不同 theta 对当前数据的解释力进行排序")

    # ------------------------------------------------------------------
    section("2) Bernoulli MLE：网格搜索与闭式解 k/n")

    grid = np.linspace(0.001, 0.999, 999)
    grid_log_likelihoods = np.array(
        [
            bernoulli_log_likelihood(theta, observations)
            for theta in grid
        ]
    )
    grid_mle = float(grid[np.argmax(grid_log_likelihoods)])
    closed_form_mle = bernoulli_mle(observations)

    print(f"正面次数 k：{int(observations.sum())}")
    print(f"总次数 n：{observations.size}")
    print(f"网格搜索 MLE：{grid_mle:.6f}")
    print(f"闭式解 k/n：{closed_form_mle:.6f}")

    assert np.isclose(grid_mle, 0.7)
    assert np.isclose(closed_form_mle, 0.7)
    assert bernoulli_mle(np.ones(4)) == 1.0
    assert bernoulli_mle(np.zeros(4)) == 0.0
    print("=> Bernoulli 参数的 MLE 就是样本中的正面频率")

    # ------------------------------------------------------------------
    section("3) 对数似然：避免大量小概率相乘时下溢")

    repeated_probability = 0.01
    token_count = 1000
    direct_product = np.prod(
        np.full(token_count, repeated_probability)
    )
    stable_log_likelihood = (
        token_count * np.log(repeated_probability)
    )

    print(f"单个 token 概率：{repeated_probability}")
    print(f"token 数量：{token_count}")
    print("直接概率乘积：", direct_product)
    print("直接累计 log 概率：", stable_log_likelihood)

    assert direct_product == 0.0
    assert np.isfinite(stable_log_likelihood)
    assert np.isclose(
        stable_log_likelihood,
        -4605.170185988091,
    )
    print("=> 应直接求 log 概率之和，而不是先乘到 0 再取 log")

    # ------------------------------------------------------------------
    section("4) 类别分布 MLE：每类概率等于经验频率")

    labels = np.array([0, 1, 0, 2, 0, 1])
    category_names = np.array(["A", "B", "C"])
    category_probabilities = categorical_mle(labels, 3)

    print("观测类别：", category_names[labels])
    print("MLE 概率：", dict(zip(
        category_names,
        category_probabilities,
    )))

    assert np.allclose(
        category_probabilities,
        [3.0 / 6.0, 2.0 / 6.0, 1.0 / 6.0],
    )
    assert np.isclose(category_probabilities.sum(), 1.0)
    print("=> Bernoulli 是两个类别时的特殊情况")

    # ------------------------------------------------------------------
    section("5) 固定方差高斯 MLE：最大似然等价于最小 MSE")

    gaussian_observations = np.array([1.0, 2.0, 3.0, 4.0])
    mean_candidates = np.array([2.0, 2.5, 3.0])
    squared_errors = np.array(
        [
            np.sum((gaussian_observations - mean) ** 2)
            for mean in mean_candidates
        ]
    )
    gaussian_log_likelihoods = np.array(
        [
            gaussian_log_likelihood(
                mean,
                gaussian_observations,
                standard_deviation=1.0,
            )
            for mean in mean_candidates
        ]
    )

    print("候选均值：", mean_candidates)
    print("平方误差和：", squared_errors)
    print("高斯对数似然：", gaussian_log_likelihoods)
    print("样本均值：", gaussian_observations.mean())

    assert mean_candidates[np.argmin(squared_errors)] == 2.5
    assert (
        mean_candidates[np.argmax(gaussian_log_likelihoods)]
        == 2.5
    )
    assert np.isclose(
        gaussian_observations.mean(),
        2.5,
    )
    print("=> 固定 sigma 时，平方误差越小，高斯对数似然越大")

    # ------------------------------------------------------------------
    section("6) LLM：最大序列似然等价于最小 token NLL")

    model_a_probabilities = np.array([0.80, 0.50, 0.25])
    model_b_probabilities = np.array([0.60, 0.60, 0.60])
    model_probabilities = {
        "模型 A": model_a_probabilities,
        "模型 B": model_b_probabilities,
    }

    for model_name, probabilities in model_probabilities.items():
        sequence_likelihood = float(np.prod(probabilities))
        log_likelihood = float(np.log(probabilities).sum())
        average_nll = mean_token_nll(probabilities)
        print(
            f"{model_name}："
            f"likelihood={sequence_likelihood:.6f}，"
            f"log_likelihood={log_likelihood:.6f}，"
            f"mean_NLL={average_nll:.6f}"
        )

    assert np.prod(model_b_probabilities) > np.prod(
        model_a_probabilities
    )
    assert mean_token_nll(
        model_b_probabilities
    ) < mean_token_nll(model_a_probabilities)
    print("=> 最大化正确 token 概率乘积，与最小化平均 NLL 排名一致")

    # ------------------------------------------------------------------
    section("7) 优化方向：上升 log-likelihood = 下降 NLL")

    heads = float(observations.sum())
    sample_count = observations.size
    learning_rate = 0.1
    ascent_logit = 0.0
    descent_logit = 0.0

    for _ in range(200):
        ascent_probability = sigmoid(ascent_logit)
        log_likelihood_gradient = (
            heads - sample_count * ascent_probability
        )
        ascent_logit += learning_rate * log_likelihood_gradient

        descent_probability = sigmoid(descent_logit)
        nll_gradient = (
            sample_count * descent_probability - heads
        )
        descent_logit -= learning_rate * nll_gradient

    ascent_probability = sigmoid(ascent_logit)
    descent_probability = sigmoid(descent_logit)

    print(f"梯度上升 log-likelihood：{ascent_probability:.6f}")
    print(f"梯度下降 NLL：{descent_probability:.6f}")
    print(f"闭式 MLE：{closed_form_mle:.6f}")

    assert np.isclose(ascent_probability, closed_form_mle)
    assert np.isclose(descent_probability, closed_form_mle)
    assert np.isclose(ascent_probability, descent_probability)
    print("=> MLE 定义目标；梯度方法只是寻找该目标最优参数的工具")

    print("\n所有断言通过：MLE 的核心关系已通过数值实验验证。")


if __name__ == "__main__":
    main()
