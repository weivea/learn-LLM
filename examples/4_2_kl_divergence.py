"""
4.2 KL 散度（Kullback-Leibler divergence）——验证脚本

配套知识点：数学知识点掌握表.md · 4.2
目标：用数值实验看清：
      1) KL 是 P 下 log(P / Q) 的期望
      2) KL 的单项可以为负，但总和非负
      3) KL 有方向，交换 P 和 Q 后通常不同
      4) P 的正概率事件被 Q 判为零时，KL 为无穷大
      5) H(P, Q) = H(P) + KL(P || Q)
      6) Monte Carlo 平均 log-ratio 如何估计 KL
      7) RLHF 中奖励与 KL 约束如何权衡
      8) 序列 log-ratio 如何拆成逐 token 之和

运行：
    .venv/bin/python examples/4_2_kl_divergence.py
"""

from __future__ import annotations

import numpy as np


def section(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def validate_log_base(base: float) -> float:
    value = float(base)
    if not np.isfinite(value):
        raise ValueError("base 必须是有限数")
    if value <= 1.0:
        raise ValueError("base 必须大于 1")
    return value


def validate_probability_distribution(
    probabilities: np.ndarray,
    name: str,
) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError(f"{name} 必须是一维非空数组")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} 不能包含 NaN 或 Inf")
    if np.any(values < 0.0):
        raise ValueError(f"{name} 不能包含负概率")
    if not np.isclose(
        values.sum(),
        1.0,
        rtol=1e-9,
        atol=1e-12,
    ):
        raise ValueError(f"{name} 的概率之和必须为 1")
    return values


def validate_distribution_pair(
    p: np.ndarray,
    q: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    p_values = validate_probability_distribution(p, "p")
    q_values = validate_probability_distribution(q, "q")
    if p_values.shape != q_values.shape:
        raise ValueError("p 和 q 必须具有相同 shape")
    return p_values, q_values


def entropy(
    probabilities: np.ndarray,
    base: float = np.e,
) -> float:
    values = validate_probability_distribution(
        probabilities,
        "probabilities",
    )
    log_base = validate_log_base(base)
    positive = values > 0.0
    return float(
        -np.sum(values[positive] * np.log(values[positive]))
        / np.log(log_base)
    )


def cross_entropy(
    p: np.ndarray,
    q: np.ndarray,
    base: float = np.e,
) -> float:
    p_values, q_values = validate_distribution_pair(p, q)
    log_base = validate_log_base(base)
    positive_p = p_values > 0.0
    if np.any(q_values[positive_p] == 0.0):
        return np.inf
    return float(
        -np.sum(p_values[positive_p] * np.log(q_values[positive_p]))
        / np.log(log_base)
    )


def kl_divergence(
    p: np.ndarray,
    q: np.ndarray,
    base: float = np.e,
) -> float:
    p_values, q_values = validate_distribution_pair(p, q)
    log_base = validate_log_base(base)
    positive_p = p_values > 0.0
    if np.any(q_values[positive_p] == 0.0):
        return np.inf
    return float(
        np.sum(
            p_values[positive_p]
            * (
                np.log(p_values[positive_p])
                - np.log(q_values[positive_p])
            )
        )
        / np.log(log_base)
    )


def main() -> None:
    np.set_printoptions(precision=6, suppress=True)

    # ------------------------------------------------------------------
    section("1) 逐项理解：KL 是 P 加权的 log(P / Q)")

    p = np.array([0.5, 0.3, 0.2])
    q = np.array([0.6, 0.25, 0.15])
    terms = p * np.log(p / q)
    kl_nats = kl_divergence(p, q)
    kl_bits = kl_divergence(p, q, base=2.0)

    print("P：", p)
    print("Q：", q)
    print("各项 p * ln(p / q)：", terms)
    print(f"KL(P || Q)：{kl_nats:.6f} nats")
    print(f"KL(P || Q)：{kl_bits:.6f} bits")

    assert terms[0] < 0.0
    assert terms[1] > 0.0
    assert terms[2] > 0.0
    assert np.isclose(kl_nats, 0.021072103131565288)
    assert np.isclose(kl_nats, terms.sum())
    assert np.isclose(kl_bits, kl_nats / np.log(2.0))
    print("=> 单项可以为负；所有项相加后的 KL 非负")

    # ------------------------------------------------------------------
    section("2) 交叉熵 = 熵 + KL")

    p_entropy = entropy(p)
    p_q_cross_entropy = cross_entropy(p, q)

    print(f"H(P)：        {p_entropy:.6f}")
    print(f"KL(P || Q)： {kl_nats:.6f}")
    print(f"H(P, Q)：     {p_q_cross_entropy:.6f}")
    print(f"H(P) + KL：   {p_entropy + kl_nats:.6f}")

    assert np.isclose(p_entropy, 1.0296530140645737)
    assert np.isclose(p_q_cross_entropy, 1.0507251171961387)
    assert np.isclose(
        p_q_cross_entropy,
        p_entropy + kl_nats,
    )
    assert np.isclose(kl_divergence(p, p), 0.0)
    print("=> P 固定时，最小化 H(P, Q) 等价于最小化 KL(P || Q)")

    # ------------------------------------------------------------------
    section("3) KL 有方向：交换两个分布后通常不同")

    direction_p = np.array([0.5, 0.5])
    direction_q = np.array([0.9, 0.1])
    forward_kl = kl_divergence(direction_p, direction_q)
    reverse_kl = kl_divergence(direction_q, direction_p)

    print("P：", direction_p)
    print("Q：", direction_q)
    print(f"KL(P || Q)：{forward_kl:.6f}")
    print(f"KL(Q || P)：{reverse_kl:.6f}")

    assert np.isclose(forward_kl, 0.5108256237659907)
    assert np.isclose(reverse_kl, 0.3680642071684971)
    assert not np.isclose(forward_kl, reverse_kl)
    print("=> 第一个分布决定采样频率和期望权重，方向不能省略")

    # ------------------------------------------------------------------
    section("4) 支撑集：P 的可能事件不能被 Q 判为不可能")

    deterministic = np.array([1.0, 0.0])
    uniform = np.array([0.5, 0.5])
    deterministic_to_uniform = kl_divergence(
        deterministic,
        uniform,
    )
    uniform_to_deterministic = kl_divergence(
        uniform,
        deterministic,
    )

    print(f"KL([1, 0] || [0.5, 0.5])：{deterministic_to_uniform}")
    print(f"KL([0.5, 0.5] || [1, 0])：{uniform_to_deterministic}")

    assert np.isclose(deterministic_to_uniform, np.log(2.0))
    assert np.isinf(uniform_to_deterministic)
    print("=> p_i=0 的项为 0；p_i>0 且 q_i=0 时 KL 为 +∞")

    # ------------------------------------------------------------------
    section("5) Gibbs 不等式与 Monte Carlo 估计")

    rng = np.random.default_rng(42)
    random_kl_values = np.array(
        [
            kl_divergence(
                rng.dirichlet(np.ones(5)),
                rng.dirichlet(np.ones(5)),
            )
            for _ in range(1_000)
        ]
    )

    sample_count = 200_000
    samples = rng.choice(
        direction_p.size,
        size=sample_count,
        p=direction_p,
    )
    sample_log_ratios = np.log(
        direction_p[samples] / direction_q[samples]
    )
    estimated_kl = float(sample_log_ratios.mean())

    print(f"1,000 组随机分布的最小 KL：{random_kl_values.min():.12f}")
    print(f"精确 KL(P || Q)：          {forward_kl:.6f}")
    print(f"{sample_count:,} 次采样估计：       {estimated_kl:.6f}")
    print(
        "样本 log-ratio 范围：       "
        f"[{sample_log_ratios.min():.6f}, "
        f"{sample_log_ratios.max():.6f}]"
    )

    assert np.all(random_kl_values >= -1e-12)
    assert sample_log_ratios.min() < 0.0
    assert sample_log_ratios.max() > 0.0
    assert np.isclose(estimated_kl, forward_kl, atol=0.01)
    print("=> 单样本 log-ratio 可为负；大量样本的平均值趋近非负 KL")

    # ------------------------------------------------------------------
    section("6) RLHF：奖励收益与偏离参考模型之间的权衡")

    reference = np.array([0.7, 0.2, 0.1])
    conservative_policy = np.array([0.55, 0.3, 0.15])
    aggressive_policy = np.array([0.1, 0.2, 0.7])
    rewards = np.array([0.0, 0.5, 1.0])

    policies = {
        "保守策略": conservative_policy,
        "激进策略": aggressive_policy,
    }
    betas = [0.1, 1.0]
    regularized_scores: dict[float, dict[str, float]] = {
        beta: {}
        for beta in betas
    }

    print("参考策略：", reference)
    print("候选结果奖励：", rewards)
    for name, policy in policies.items():
        expected_reward = float(policy @ rewards)
        divergence = kl_divergence(policy, reference)
        print(
            f"{name} {policy}："
            f"E[reward]={expected_reward:.6f}, "
            f"KL(current || ref)={divergence:.6f}"
        )
        for beta in betas:
            score = expected_reward - beta * divergence
            regularized_scores[beta][name] = score
            print(f"  beta={beta:.1f} -> 正则化目标={score:.6f}")

    assert (
        regularized_scores[0.1]["激进策略"]
        > regularized_scores[0.1]["保守策略"]
    )
    assert (
        regularized_scores[1.0]["激进策略"]
        < regularized_scores[1.0]["保守策略"]
    )
    print("=> beta 小时奖励更重要；beta 大时偏离参考模型的代价更重要")

    # ------------------------------------------------------------------
    section("7) 序列 log-ratio = 逐 token log-ratio 之和")

    current_token_probabilities = np.array([0.6, 0.5, 0.4])
    reference_token_probabilities = np.array([0.4, 0.5, 0.2])
    current_sequence_probability = current_token_probabilities.prod()
    reference_sequence_probability = (
        reference_token_probabilities.prod()
    )
    token_log_ratios = np.log(
        current_token_probabilities
        / reference_token_probabilities
    )
    sequence_log_ratio = np.log(
        current_sequence_probability
        / reference_sequence_probability
    )

    print("当前策略逐 token 概率：", current_token_probabilities)
    print("参考策略逐 token 概率：", reference_token_probabilities)
    print("逐 token log-ratio：    ", token_log_ratios)
    print(f"逐 token log-ratio 之和：{token_log_ratios.sum():.6f}")
    print(f"整条序列 log-ratio：     {sequence_log_ratio:.6f}")

    assert np.isclose(current_sequence_probability, 0.12)
    assert np.isclose(reference_sequence_probability, 0.04)
    assert np.isclose(sequence_log_ratio, np.log(3.0))
    assert np.isclose(sequence_log_ratio, token_log_ratios.sum())
    print("=> 自回归概率的乘积经过 log 后变成逐 token 项的和")

    print("\n所有断言通过：4.2 KL 散度验证完成。")


if __name__ == "__main__":
    main()
