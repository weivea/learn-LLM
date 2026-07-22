"""
4.3 交叉熵与熵的关系——验证脚本

配套知识点：数学知识点掌握表.md · 4.3
目标：用数值实验看清：
      1) H(q, p) = H(q) + KL(q || p)
      2) 交叉熵的下界是真实熵，而不是模型熵
      3) one-hot 目标下，交叉熵、KL 和 NLL 数值相同
      4) 真实样本的平均 NLL 如何估计交叉熵
      5) 困惑度如何分解为固有不确定性与模型失配倍率

运行：
    .venv/bin/python examples/4_3_cross_entropy_entropy.py
"""

from __future__ import annotations

import numpy as np


def section(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


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
    q: np.ndarray,
    p: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    q_values = validate_probability_distribution(q, "q")
    p_values = validate_probability_distribution(p, "p")
    if q_values.shape != p_values.shape:
        raise ValueError("q 和 p 必须具有相同 shape")
    return q_values, p_values


def entropy(probabilities: np.ndarray) -> float:
    values = validate_probability_distribution(
        probabilities,
        "probabilities",
    )
    positive = values > 0.0
    return float(-np.sum(values[positive] * np.log(values[positive])))


def cross_entropy(q: np.ndarray, p: np.ndarray) -> float:
    q_values, p_values = validate_distribution_pair(q, p)
    positive_q = q_values > 0.0
    if np.any(p_values[positive_q] == 0.0):
        return np.inf
    return float(
        -np.sum(q_values[positive_q] * np.log(p_values[positive_q]))
    )


def kl_divergence(q: np.ndarray, p: np.ndarray) -> float:
    q_values, p_values = validate_distribution_pair(q, p)
    positive_q = q_values > 0.0
    if np.any(p_values[positive_q] == 0.0):
        return np.inf
    return float(
        np.sum(
            q_values[positive_q]
            * (
                np.log(q_values[positive_q])
                - np.log(p_values[positive_q])
            )
        )
    )


def main() -> None:
    np.set_printoptions(precision=6, suppress=True)

    # ------------------------------------------------------------------
    section("1) 核心分解：交叉熵 = 真实熵 + KL")

    q = np.array([0.5, 0.3, 0.2])
    p = np.array([0.6, 0.25, 0.15])
    q_entropy = entropy(q)
    q_p_cross_entropy = cross_entropy(q, p)
    q_p_kl = kl_divergence(q, p)

    print("真实分布 q：", q)
    print("模型分布 p：", p)
    print(f"H(q)：        {q_entropy:.6f}")
    print(f"KL(q || p)： {q_p_kl:.6f}")
    print(f"H(q, p)：     {q_p_cross_entropy:.6f}")

    assert np.isclose(q_entropy, 1.0296530140645737)
    assert np.isclose(q_p_kl, 0.021072103131565288)
    assert np.isclose(q_p_cross_entropy, 1.0507251171961387)
    assert np.isclose(
        q_p_cross_entropy,
        q_entropy + q_p_kl,
    )
    print("=> 总代价 = 数据固有不确定性 + 模型不匹配的额外代价")

    # ------------------------------------------------------------------
    section("2) 模型熵低，不代表真实数据上的交叉熵低")

    binary_q = np.array([0.75, 0.25])
    models = {
        "匹配模型": np.array([0.75, 0.25]),
        "均匀模型": np.array([0.5, 0.5]),
        "自信但方向错误": np.array([0.1, 0.9]),
    }
    metrics: dict[str, tuple[float, float, float]] = {}

    print("真实分布 q：", binary_q)
    for name, model in models.items():
        model_metrics = (
            entropy(model),
            cross_entropy(binary_q, model),
            kl_divergence(binary_q, model),
        )
        metrics[name] = model_metrics
        model_entropy, model_cross_entropy, model_kl = model_metrics
        print(
            f"{name:10s} p={model} -> "
            f"H(p)={model_entropy:.6f}, "
            f"H(q,p)={model_cross_entropy:.6f}, "
            f"KL={model_kl:.6f}"
        )

    true_entropy = entropy(binary_q)
    assert np.isclose(
        metrics["匹配模型"][1],
        true_entropy,
    )
    assert np.isclose(metrics["匹配模型"][2], 0.0)
    assert metrics["自信但方向错误"][0] < true_entropy
    assert (
        metrics["自信但方向错误"][1]
        > metrics["均匀模型"][1]
        > metrics["匹配模型"][1]
    )
    print("=> H(p) 衡量模型有多自信；H(q,p) 才衡量预测真实数据的代价")

    # ------------------------------------------------------------------
    section("3) one-hot 特例：交叉熵 = KL = NLL")

    one_hot_q = np.array([0.0, 1.0, 0.0])
    predicted_p = np.array([0.1, 0.7, 0.2])
    target_index = 1
    one_hot_entropy = entropy(one_hot_q)
    one_hot_cross_entropy = cross_entropy(one_hot_q, predicted_p)
    one_hot_kl = kl_divergence(one_hot_q, predicted_p)
    nll = float(-np.log(predicted_p[target_index]))

    print("one-hot q：", one_hot_q)
    print("模型 p：  ", predicted_p)
    print(f"H(q)：       {one_hot_entropy:.6f}")
    print(f"H(q,p)：     {one_hot_cross_entropy:.6f}")
    print(f"KL(q || p)：{one_hot_kl:.6f}")
    print(f"NLL：        {nll:.6f}")

    assert one_hot_entropy == 0.0
    assert np.isclose(one_hot_cross_entropy, nll)
    assert np.isclose(one_hot_kl, nll)
    print("=> one-hot 分布的熵为 0，因此三个训练量数值相同")

    # ------------------------------------------------------------------
    section("4) 大量真实样本的平均 NLL 趋近交叉熵")

    rng = np.random.default_rng(42)
    sample_count = 200_000
    samples = rng.choice(
        q.size,
        size=sample_count,
        p=q,
    )
    sample_nll_values = -np.log(p[samples])
    sample_true_information = -np.log(q[samples])
    estimated_cross_entropy = float(sample_nll_values.mean())
    estimated_entropy = float(sample_true_information.mean())
    estimated_kl = estimated_cross_entropy - estimated_entropy

    print(f"样本数量：                 {sample_count:,}")
    print(f"理论 H(q,p)：              {q_p_cross_entropy:.6f}")
    print(f"样本平均 NLL：             {estimated_cross_entropy:.6f}")
    print(f"样本估计 H(q)：            {estimated_entropy:.6f}")
    print(f"两个样本平均之差：         {estimated_kl:.6f}")

    assert np.isclose(
        estimated_cross_entropy,
        q_p_cross_entropy,
        atol=0.01,
    )
    assert np.isclose(estimated_entropy, q_entropy, atol=0.01)
    assert np.isclose(estimated_kl, q_p_kl, atol=0.01)
    print("=> 单样本算 NLL；真实样本足够多时，其平均值估计交叉熵")

    # ------------------------------------------------------------------
    section("5) 困惑度 = 固有候选数 × 模型失配倍率")

    perplexity = np.exp(q_p_cross_entropy)
    intrinsic_choices = np.exp(q_entropy)
    mismatch_multiplier = np.exp(q_p_kl)

    print(f"exp(H(q,p))：       {perplexity:.6f}")
    print(f"exp(H(q))：         {intrinsic_choices:.6f}")
    print(f"exp(KL(q || p))：  {mismatch_multiplier:.6f}")
    print(
        "二者乘积：          "
        f"{intrinsic_choices * mismatch_multiplier:.6f}"
    )

    assert np.isclose(
        perplexity,
        intrinsic_choices * mismatch_multiplier,
    )
    assert mismatch_multiplier >= 1.0
    print("=> KL 为 0 时失配倍率为 1，模型达到数据熵决定的最佳困惑度")

    # ------------------------------------------------------------------
    section("6) 零概率边界")

    possible_q = np.array([0.5, 0.5])
    impossible_p = np.array([1.0, 0.0])
    infinite_cross_entropy = cross_entropy(possible_q, impossible_p)
    infinite_kl = kl_divergence(possible_q, impossible_p)

    print(f"H([0.5,0.5], [1,0])：     {infinite_cross_entropy}")
    print(f"KL([0.5,0.5] || [1,0])： {infinite_kl}")

    assert np.isinf(infinite_cross_entropy)
    assert np.isinf(infinite_kl)
    print("=> q_i > 0 但 p_i = 0 时，交叉熵与 KL 都为 +∞")

    print("\n所有断言通过：4.3 交叉熵与熵的关系验证完成。")


if __name__ == "__main__":
    main()
