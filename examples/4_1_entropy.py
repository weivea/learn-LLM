"""
4.1 熵（entropy）——验证脚本

配套知识点：数学知识点掌握表.md · 4.1
目标：用数值实验看清：
      1) 小概率结果为什么有更大的自信息
      2) 熵为什么是自信息的概率加权平均
      3) 确定分布与均匀分布的熵边界
      4) Monte Carlo 平均意外程度为什么趋近熵
      5) temperature 如何改变 token 分布熵
      6) 低熵为什么不代表预测正确
      7) exp(H) 如何表示有效候选数量

运行：
    .venv/bin/python examples/4_1_entropy.py
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
) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("probabilities 必须是一维非空数组")
    if not np.all(np.isfinite(values)):
        raise ValueError("probabilities 不能包含 NaN 或 Inf")
    if np.any(values < 0.0):
        raise ValueError("概率不能为负")
    if not np.isclose(
        values.sum(),
        1.0,
        rtol=1e-9,
        atol=1e-12,
    ):
        raise ValueError("概率之和必须为 1")
    return values


def self_information(
    probability: float,
    base: float = np.e,
) -> float:
    value = float(probability)
    log_base = validate_log_base(base)
    if not np.isfinite(value):
        raise ValueError("probability 必须是有限数")
    if not 0.0 <= value <= 1.0:
        raise ValueError("probability 必须满足 0 <= p <= 1")
    if value == 0.0:
        return np.inf
    return float(-np.log(value) / np.log(log_base))


def entropy(
    probabilities: np.ndarray,
    base: float = np.e,
) -> float:
    values = validate_probability_distribution(probabilities)
    log_base = validate_log_base(base)
    positive = values > 0.0
    return float(
        -np.sum(values[positive] * np.log(values[positive]))
        / np.log(log_base)
    )


def softmax_with_temperature(
    logits: np.ndarray,
    temperature: float,
) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    scale = float(temperature)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("logits 必须是一维非空数组")
    if not np.all(np.isfinite(values)):
        raise ValueError("logits 不能包含 NaN 或 Inf")
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("temperature 必须是正有限数")

    scaled = values / scale
    shifted = scaled - scaled.max()
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum()


def main() -> None:
    np.set_printoptions(precision=6, suppress=True)

    # ------------------------------------------------------------------
    section("1) 自信息：结果越罕见，出现时越意外")

    event_probabilities = np.array([1.0, 0.5, 0.25, 0.01])
    information_bits = np.array(
        [
            self_information(probability, base=2.0)
            for probability in event_probabilities
        ]
    )

    print("事件概率：", event_probabilities)
    print("自信息(bits)：", information_bits)

    assert np.allclose(
        information_bits,
        [0.0, 1.0, 2.0, -np.log2(0.01)],
    )
    assert np.isinf(self_information(0.0))
    print("=> 必然事件没有新信息；小概率事件出现时信息量更大")

    # ------------------------------------------------------------------
    section("2) 熵：自信息的概率加权平均")

    token_probabilities = np.array([0.7, 0.2, 0.1])
    token_information = np.array(
        [
            self_information(probability)
            for probability in token_probabilities
        ]
    )
    entropy_contributions = (
        token_probabilities * token_information
    )
    token_entropy = entropy(token_probabilities)

    print("token 概率：", token_probabilities)
    print("各 token 自信息(nats)：", token_information)
    print("各项 p * (-ln p)：", entropy_contributions)
    print(f"熵：{token_entropy:.6f} nats")

    assert np.allclose(
        entropy_contributions,
        [
            -0.7 * np.log(0.7),
            -0.2 * np.log(0.2),
            -0.1 * np.log(0.1),
        ],
    )
    assert np.isclose(token_entropy, 0.8018185525433373)
    assert np.isclose(token_entropy, entropy_contributions.sum())
    print("=> 熵不是自信息的普通平均，而是按结果概率加权的期望")

    # ------------------------------------------------------------------
    section("3) 熵的边界：确定分布最小，均匀分布最大")

    deterministic = np.array([1.0, 0.0, 0.0, 0.0])
    two_choices = np.array([0.5, 0.5, 0.0, 0.0])
    four_choices = np.full(4, 0.25)

    distributions = {
        "确定分布": deterministic,
        "两个等可能结果": two_choices,
        "四个等可能结果": four_choices,
    }

    for name, probabilities in distributions.items():
        print(
            f"{name:10s} {probabilities}"
            f" -> {entropy(probabilities, base=2.0):.6f} bits"
        )

    assert entropy(deterministic, base=2.0) == 0.0
    assert np.isclose(entropy(two_choices, base=2.0), 1.0)
    assert np.isclose(entropy(four_choices, base=2.0), 2.0)
    assert np.isclose(
        entropy(four_choices),
        np.log(four_choices.size),
    )
    print("=> 对 K 分类离散分布，0 <= H <= log(K)")

    # ------------------------------------------------------------------
    section("4) Monte Carlo：长期平均意外程度趋近熵")

    rng = np.random.default_rng(42)
    sample_count = 200_000
    samples = rng.choice(
        token_probabilities.size,
        size=sample_count,
        p=token_probabilities,
    )
    sampled_information = token_information[samples]
    monte_carlo_entropy = float(sampled_information.mean())

    print(f"理论熵：{token_entropy:.6f} nats")
    print(f"采样次数：{sample_count}")
    print(f"样本平均自信息：{monte_carlo_entropy:.6f} nats")

    assert np.isclose(
        monte_carlo_entropy,
        token_entropy,
        atol=0.01,
    )
    print("=> H(P) = E[-ln P(X)] 可用大量采样的平均值近似")

    # ------------------------------------------------------------------
    section("5) Temperature：改变采样分布的尖锐程度和熵")

    logits = np.array([2.0, 1.0, 0.0, -1.0])
    temperatures = np.array([0.5, 1.0, 2.0])
    temperature_probabilities = [
        softmax_with_temperature(logits, temperature)
        for temperature in temperatures
    ]
    temperature_entropies = np.array(
        [
            entropy(probabilities)
            for probabilities in temperature_probabilities
        ]
    )

    for temperature, probabilities, value in zip(
        temperatures,
        temperature_probabilities,
        temperature_entropies,
    ):
        print(
            f"T={temperature:.1f} -> p={probabilities}"
            f" -> H={value:.6f} nats"
        )

    assert np.all(np.diff(temperature_entropies) > 0.0)
    assert np.allclose(
        softmax_with_temperature(np.ones(4), 0.5),
        np.full(4, 0.25),
    )
    print("=> 非常数 logits 中，temperature 越高，分布通常越平坦")

    # ------------------------------------------------------------------
    section("6) 低熵不等于预测正确")

    confident_distribution = np.array([0.98, 0.01, 0.01])
    less_confident_distribution = np.array([0.4, 0.3, 0.3])
    true_token_index = 2

    confident_entropy = entropy(confident_distribution)
    less_confident_entropy = entropy(less_confident_distribution)
    confident_nll = self_information(
        confident_distribution[true_token_index]
    )
    less_confident_nll = self_information(
        less_confident_distribution[true_token_index]
    )

    print(
        "自信模型：",
        f"H={confident_entropy:.6f},",
        f"真实 token NLL={confident_nll:.6f}",
    )
    print(
        "较平坦模型：",
        f"H={less_confident_entropy:.6f},",
        f"真实 token NLL={less_confident_nll:.6f}",
    )

    assert confident_entropy < less_confident_entropy
    assert confident_nll > less_confident_nll
    print("=> 熵衡量不确定性；真实 token NLL 才反映这次预测代价")

    # ------------------------------------------------------------------
    section("7) exp(H)：分布的有效候选数量")

    effective_choices = np.exp(token_entropy)
    uniform_effective_choices = np.exp(entropy(four_choices))

    print("token 分布：", token_probabilities)
    print(f"有效候选数量 exp(H)：{effective_choices:.6f}")
    print(
        "四分类均匀分布的 exp(H)：",
        f"{uniform_effective_choices:.6f}",
    )

    assert np.isclose(effective_choices, 2.229591873920416)
    assert np.isclose(uniform_effective_choices, 4.0)
    print("=> exp(H) 把对数尺度的不确定性还原成等可能候选数")

    print("\n所有断言通过：4.1 熵验证完成。")


if __name__ == "__main__":
    main()
