"""
3.10 贝叶斯直觉——验证脚本

配套知识点：数学知识点掌握表.md · 3.10
目标：用数值实验看清：
      1) posterior 如何由 prior × likelihood 归一化得到
      2) 医学检测中基础概率为什么不能忽略
      3) 相同证据如何因先验不同产生不同后验
      4) posterior 如何成为下一轮 prior
      5) 支持证据与反对证据怎样改变信念
      6) MLE 与 MAP 为什么可能选择不同假设

运行：
    .venv/bin/python examples/3_10_bayesian_intuition.py
"""

from __future__ import annotations

import numpy as np


def section(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def validate_probability(value: float, name: str) -> float:
    probability = float(value)
    if not np.isfinite(probability):
        raise ValueError(f"{name} 必须是有限数")
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"{name} 必须满足 0 <= p <= 1")
    return probability


def validate_distribution(
    probabilities: np.ndarray,
    name: str,
) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError(f"{name} 必须是一维非空数组")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} 不能包含 NaN 或 Inf")
    if np.any(values < 0.0):
        raise ValueError(f"{name} 不能包含负数")
    if not np.isclose(values.sum(), 1.0):
        raise ValueError(f"{name} 的概率之和必须为 1")
    return values


def validate_likelihoods(likelihoods: np.ndarray) -> np.ndarray:
    values = np.asarray(likelihoods, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("likelihoods 必须是一维非空数组")
    if not np.all(np.isfinite(values)):
        raise ValueError("likelihoods 不能包含 NaN 或 Inf")
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("每个 likelihood 必须满足 0 <= p <= 1")
    return values


def bayes_update(
    priors: np.ndarray,
    likelihoods: np.ndarray,
) -> tuple[np.ndarray, float]:
    prior_values = validate_distribution(priors, "priors")
    likelihood_values = validate_likelihoods(likelihoods)
    if prior_values.shape != likelihood_values.shape:
        raise ValueError("priors 与 likelihoods 的形状必须相同")

    weighted = prior_values * likelihood_values
    evidence = float(weighted.sum())
    if evidence <= 0.0:
        raise ValueError("证据概率必须大于 0，无法对全零权重归一化")

    posterior = weighted / evidence
    return posterior, evidence


def positive_test_posterior(
    prevalence: float,
    sensitivity: float,
    false_positive_rate: float,
) -> float:
    disease_prior = validate_probability(prevalence, "prevalence")
    positive_if_disease = validate_probability(
        sensitivity,
        "sensitivity",
    )
    positive_if_healthy = validate_probability(
        false_positive_rate,
        "false_positive_rate",
    )

    priors = np.array([disease_prior, 1.0 - disease_prior])
    likelihoods = np.array(
        [positive_if_disease, positive_if_healthy]
    )
    posterior, _ = bayes_update(priors, likelihoods)
    return float(posterior[0])


def main() -> None:
    np.set_printoptions(precision=6, suppress=True)

    # ------------------------------------------------------------------
    section("1) 医学检测：posterior = prior × likelihood，再归一化")

    hypothesis_names = np.array(["患病", "未患病"])
    priors = np.array([0.01, 0.99])
    positive_likelihoods = np.array([0.99, 0.05])
    posterior, positive_probability = bayes_update(
        priors,
        positive_likelihoods,
    )
    unnormalized = priors * positive_likelihoods

    print("假设：", hypothesis_names)
    print("先验 P(H)：", priors)
    print("似然 P(阳性 | H)：", positive_likelihoods)
    print("未归一化权重：", unnormalized)
    print(f"证据 P(阳性)：{positive_probability:.4f}")
    print("后验 P(H | 阳性)：", posterior)

    assert np.allclose(unnormalized, [0.0099, 0.0495])
    assert np.isclose(positive_probability, 0.0594)
    assert np.allclose(posterior, [1.0 / 6.0, 5.0 / 6.0])
    assert np.isclose(posterior.sum(), 1.0)
    print("=> 检测灵敏度 99%，不代表阳性后有 99% 的患病概率")

    # ------------------------------------------------------------------
    section("2) 自然频数：用 10000 人看清假阳性")

    population = 10_000
    disease_count = int(population * priors[0])
    healthy_count = population - disease_count
    true_positive_count = int(
        disease_count * positive_likelihoods[0]
    )
    false_positive_count = int(
        healthy_count * positive_likelihoods[1]
    )
    all_positive_count = true_positive_count + false_positive_count
    frequency_posterior = true_positive_count / all_positive_count

    print(f"总人数：{population}")
    print(f"患病人数：{disease_count}")
    print(f"真阳性人数：{true_positive_count}")
    print(f"健康人数：{healthy_count}")
    print(f"假阳性人数：{false_positive_count}")
    print(f"全部阳性人数：{all_positive_count}")
    print(
        "阳性后患病比例："
        f"{true_positive_count}/{all_positive_count}"
        f" = {frequency_posterior:.4f}"
    )

    assert disease_count == 100
    assert true_positive_count == 99
    assert false_positive_count == 495
    assert all_positive_count == 594
    assert np.isclose(frequency_posterior, posterior[0])
    print("=> 稀有疾病中，健康人基数很大，少量误报也可能数量很多")

    # ------------------------------------------------------------------
    section("3) 基础概率：相同检测，不同先验得到不同后验")

    prevalences = np.array([0.01, 0.10, 0.50])
    posteriors = np.array(
        [
            positive_test_posterior(
                prevalence,
                sensitivity=0.99,
                false_positive_rate=0.05,
            )
            for prevalence in prevalences
        ]
    )

    for prevalence, disease_posterior in zip(
        prevalences,
        posteriors,
    ):
        print(
            f"先验患病率 {prevalence:>5.0%}"
            f" -> 阳性后患病概率 {disease_posterior:>7.2%}"
        )

    assert np.allclose(
        posteriors,
        [1.0 / 6.0, 0.6875, 0.9519230769230769],
    )
    assert np.all(np.diff(posteriors) > 0.0)
    print("=> 只看检测性能而忽略先验，会犯基础概率忽视错误")

    # ------------------------------------------------------------------
    section("4) 连续证据：posterior 成为下一轮 prior")

    coin_names = np.array(["公平硬币", "偏正面硬币"])
    coin_posterior = np.array([0.5, 0.5])
    evidence_sequence = ["正面", "正面", "反面"]

    print("初始先验：", dict(zip(coin_names, coin_posterior)))
    for observation in evidence_sequence:
        if observation == "正面":
            likelihoods = np.array([0.5, 0.8])
        else:
            likelihoods = np.array([0.5, 0.2])

        coin_posterior, evidence_probability = bayes_update(
            coin_posterior,
            likelihoods,
        )
        print(
            f"观察到{observation}："
            f"P(偏正面硬币 | 已有证据)={coin_posterior[1]:.6f}，"
            f"本轮证据概率={evidence_probability:.6f}"
        )

    fair_sequence_likelihood = 0.5 * 0.5 * 0.5
    biased_sequence_likelihood = 0.8 * 0.8 * 0.2
    expected_biased_posterior = (
        0.5
        * biased_sequence_likelihood
        / (
            0.5 * fair_sequence_likelihood
            + 0.5 * biased_sequence_likelihood
        )
    )

    assert np.isclose(coin_posterior[1], expected_biased_posterior)
    assert np.isclose(coin_posterior[1], 0.5059288537549407)
    print("=> 两次正面支持偏硬币，随后反面又把信念拉回接近五五开")

    # ------------------------------------------------------------------
    section("5) 证据的方向：支持、反对或没有区分力")

    equal_priors = np.array([0.5, 0.5])
    support_h, _ = bayes_update(
        equal_priors,
        np.array([0.9, 0.2]),
    )
    oppose_h, _ = bayes_update(
        equal_priors,
        np.array([0.2, 0.9]),
    )
    neutral, _ = bayes_update(
        equal_priors,
        np.array([0.4, 0.4]),
    )

    print("先验 P(H)：", equal_priors[0])
    print("支持 H 的证据后 P(H | E)：", support_h[0])
    print("反对 H 的证据后 P(H | E)：", oppose_h[0])
    print("无区分力证据后 P(H | E)：", neutral[0])

    assert support_h[0] > equal_priors[0]
    assert oppose_h[0] < equal_priors[0]
    assert np.isclose(neutral[0], equal_priors[0])
    print("=> 比较 P(E | H) 与 P(E | 非H)，可判断证据更新方向")

    # ------------------------------------------------------------------
    section("6) MLE 与 MAP：先验可能改变最终选择")

    model_names = np.array(["简单模型", "复杂模型"])
    model_priors = np.array([0.90, 0.10])
    data_likelihoods = np.array([0.30, 0.60])
    model_posteriors, _ = bayes_update(
        model_priors,
        data_likelihoods,
    )

    mle_index = int(np.argmax(data_likelihoods))
    map_index = int(np.argmax(model_posteriors))

    print("模型：", model_names)
    print("先验 P(model)：", model_priors)
    print("数据似然 P(data | model)：", data_likelihoods)
    print("后验 P(model | data)：", model_posteriors)
    print("MLE 选择：", model_names[mle_index])
    print("MAP 选择：", model_names[map_index])

    assert mle_index == 1
    assert map_index == 0
    assert np.allclose(model_posteriors, [0.8181818182, 0.1818181818])
    print("=> MLE 只看似然；MAP 比较似然 × 先验")

    section("核心结论")
    print("1. posterior ∝ likelihood × prior，再对所有假设归一化")
    print("2. P(H | E) 与 P(E | H) 的条件方向不可交换")
    print("3. 稀有事件中，基础概率和假阳性数量尤其重要")
    print("4. 本轮 posterior 可以作为下一轮 prior，持续吸收证据")
    print("5. MLE 只看 likelihood；MAP 还会结合 prior")
    print("6. LLM 条件概率与贝叶斯有关，但不等于显式参数后验更新")


if __name__ == "__main__":
    main()
