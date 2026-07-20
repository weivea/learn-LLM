"""
3.9 采样（sampling）——验证脚本

配套知识点：数学知识点掌握表.md · 3.9
目标：用数值实验看清：
      1) categorical sampling 如何按概率抽取 token
      2) 大量采样的经验频率如何接近理论概率
      3) temperature 如何改变分布的尖锐程度
      4) top-k 与 top-p 如何过滤并重新归一化
      5) 多种策略怎样组合
      6) LLM 如何逐 token 自回归生成

运行：
    .venv/bin/python examples/3_9_sampling.py
"""

from __future__ import annotations

from numbers import Integral

import numpy as np


def section(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def validate_probabilities(probabilities: np.ndarray) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("probabilities 必须是一维非空数组")
    if not np.all(np.isfinite(values)):
        raise ValueError("probabilities 不能包含 NaN 或 Inf")
    if np.any(values < 0.0):
        raise ValueError("概率不能为负数")
    if not np.isclose(values.sum(), 1.0):
        raise ValueError("概率之和必须为 1")
    return values


def softmax(logits: np.ndarray) -> np.ndarray:
    scores = np.asarray(logits, dtype=np.float64)
    if scores.ndim != 1 or scores.size == 0:
        raise ValueError("logits 必须是一维非空数组")
    if not np.all(np.isfinite(scores)):
        raise ValueError("logits 不能包含 NaN 或 Inf")

    shifted = scores - np.max(scores)
    exp_values = np.exp(shifted)
    return exp_values / exp_values.sum()


def temperature_distribution(
    logits: np.ndarray,
    temperature: float = 1.0,
) -> np.ndarray:
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature 必须是大于 0 的有限数")
    return softmax(np.asarray(logits, dtype=np.float64) / temperature)


def top_k_distribution(
    probabilities: np.ndarray,
    k: int,
) -> np.ndarray:
    values = validate_probabilities(probabilities)
    if isinstance(k, bool) or not isinstance(k, Integral):
        raise ValueError("k 必须是整数")
    if not 1 <= k <= values.size:
        raise ValueError("k 必须满足 1 <= k <= 词表大小")

    top_indices = np.argsort(values)[-k:]
    filtered = np.zeros_like(values)
    filtered[top_indices] = values[top_indices]
    return filtered / filtered.sum()


def top_p_distribution(
    probabilities: np.ndarray,
    top_p: float,
) -> np.ndarray:
    values = validate_probabilities(probabilities)
    if not np.isfinite(top_p) or not 0.0 < top_p <= 1.0:
        raise ValueError("top_p 必须满足 0 < top_p <= 1")

    order = np.argsort(values)[::-1]
    sorted_probabilities = values[order]
    cumulative = np.cumsum(sorted_probabilities)
    crossing_index = int(
        np.searchsorted(cumulative, top_p, side="left")
    )
    keep_count = min(crossing_index + 1, values.size)
    keep_indices = order[:keep_count]

    filtered = np.zeros_like(values)
    filtered[keep_indices] = values[keep_indices]
    return filtered / filtered.sum()


def prepare_sampling_distribution(
    logits: np.ndarray,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
) -> np.ndarray:
    probabilities = temperature_distribution(logits, temperature)
    if top_k is not None:
        probabilities = top_k_distribution(probabilities, top_k)
    if top_p is not None:
        probabilities = top_p_distribution(probabilities, top_p)
    return probabilities


def inverse_transform_index(
    probabilities: np.ndarray,
    uniform_value: float,
) -> int:
    values = validate_probabilities(probabilities)
    if (
        not np.isfinite(uniform_value)
        or uniform_value < 0.0
        or uniform_value >= 1.0
    ):
        raise ValueError("uniform_value 必须满足 0 <= u < 1")

    index = int(
        np.searchsorted(
            np.cumsum(values),
            uniform_value,
            side="right",
        )
    )
    return min(index, values.size - 1)


def sample_index(
    probabilities: np.ndarray,
    rng: np.random.Generator,
) -> int:
    values = validate_probabilities(probabilities)
    return int(rng.choice(values.size, p=values))


TOKENS = np.array(["我", "喜欢", "学习", "数学", "代码", "。"])

# 第 0 行表示 BOS；之后每一行表示只根据上一个 token 预测下一步。
# 真实 LLM 会使用全部前文，这里只保留最小结构来演示自回归循环。
TOY_TRANSITION_LOGITS = np.array(
    [
        [5.0, -2.0, -2.0, -2.0, -2.0, -3.0],
        [-3.0, 4.0, 1.0, -1.0, -1.0, -3.0],
        [-4.0, -3.0, 3.0, 1.5, 1.3, -2.0],
        [-4.0, -3.0, -3.0, 2.5, 2.4, 0.5],
        [-4.0, -4.0, -3.0, -2.0, 1.2, 3.0],
        [-4.0, -4.0, -3.0, -2.0, -2.0, 4.0],
        [-4.0, -4.0, -4.0, -4.0, -4.0, 4.0],
    ],
    dtype=np.float64,
)


def generate_toy_text(
    *,
    seed: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    max_new_tokens: int = 8,
) -> str:
    rng = np.random.default_rng(seed)
    context_row = 0
    generated: list[str] = []

    for _ in range(max_new_tokens):
        probabilities = prepare_sampling_distribution(
            TOY_TRANSITION_LOGITS[context_row],
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        token_index = sample_index(probabilities, rng)
        token = str(TOKENS[token_index])
        generated.append(token)
        if token == "。":
            break
        context_row = token_index + 1

    return "".join(generated)


def main() -> None:
    np.set_printoptions(precision=4, suppress=True)

    tokens = np.array(["数学", "代码", "模型", "天气"])
    probabilities = np.array([0.50, 0.30, 0.15, 0.05])

    # ------------------------------------------------------------------
    section("1) 手算采样：随机数落入哪个累积概率区间")

    uniform_values = [0.12, 0.63, 0.91, 0.99]
    sampled_tokens = [
        tokens[inverse_transform_index(probabilities, value)]
        for value in uniform_values
    ]

    print("tokens：", tokens)
    print("概率：", probabilities)
    print("累积概率：", np.cumsum(probabilities))
    for value, token in zip(uniform_values, sampled_tokens):
        print(f"U={value:.2f} -> {token}")

    assert sampled_tokens == ["数学", "代码", "模型", "天气"]
    print("=> 选择第一个满足 cumulative_probability > U 的 token")

    # ------------------------------------------------------------------
    section("2) 大量采样：经验频率逐渐接近理论概率")

    rng = np.random.default_rng(7)
    sample_count = 100_000
    samples = rng.choice(
        tokens.size,
        size=sample_count,
        p=probabilities,
    )
    counts = np.bincount(samples, minlength=tokens.size)
    empirical = counts / sample_count

    print("理论概率：", probabilities)
    print("采样次数：", counts)
    print("经验频率：", empirical)

    assert np.allclose(empirical, probabilities, atol=0.005)

    expected_math_count = sample_count * probabilities[0]
    math_count_std = np.sqrt(
        sample_count
        * probabilities[0]
        * (1.0 - probabilities[0])
    )
    print(f"E[数学的计数]：{expected_math_count:.3f}")
    print(f"Std(数学的计数)：{math_count_std:.3f}")
    print("=> 单次计数不必等于期望；大量采样的频率通常接近理论概率")

    # ------------------------------------------------------------------
    section("3) 随机种子：相同 seed 复现相同伪随机序列")

    first_rng = np.random.default_rng(42)
    second_rng = np.random.default_rng(42)
    first_sequence = first_rng.choice(4, size=12, p=probabilities)
    second_sequence = second_rng.choice(4, size=12, p=probabilities)

    print("序列 A：", tokens[first_sequence])
    print("序列 B：", tokens[second_sequence])

    assert np.array_equal(first_sequence, second_sequence)
    print("=> 固定 seed 便于调试，但不等于 greedy decoding")

    # ------------------------------------------------------------------
    section("4) Temperature：低温更尖锐，高温更平坦")

    logits = np.log(probabilities)
    cold = temperature_distribution(logits, temperature=0.5)
    original = temperature_distribution(logits, temperature=1.0)
    hot = temperature_distribution(logits, temperature=2.0)

    print("T=0.5：", cold)
    print("T=1.0：", original)
    print("T=2.0：", hot)

    assert np.allclose(original, probabilities)
    assert cold[0] > original[0] > hot[0]
    assert np.argmax(cold) == np.argmax(hot) == 0
    print("=> 正 temperature 改变概率差距，但不改变 logits 排名")

    # ------------------------------------------------------------------
    section("5) Top-k：固定保留概率最高的 k 个候选")

    top_k_two = top_k_distribution(probabilities, k=2)

    print("原分布：", probabilities)
    print("top-k=2：", top_k_two)

    assert np.allclose(top_k_two, [0.625, 0.375, 0.0, 0.0])
    assert np.isclose(top_k_two.sum(), 1.0)
    print("=> 过滤后必须重新归一化")

    # ------------------------------------------------------------------
    section("6) Top-p：候选数量随分布动态变化")

    top_p_080 = top_p_distribution(probabilities, top_p=0.80)
    top_p_090 = top_p_distribution(probabilities, top_p=0.90)

    print("原分布：", probabilities)
    print("top-p=0.80：", top_p_080)
    print("top-p=0.90：", top_p_090)

    assert np.allclose(top_p_080, [0.625, 0.375, 0.0, 0.0])
    assert np.allclose(
        top_p_090,
        [0.50 / 0.95, 0.30 / 0.95, 0.15 / 0.95, 0.0],
    )
    print("=> 保留第一个让累积概率达到或超过 top-p 的 token")

    # ------------------------------------------------------------------
    section("7) 组合策略：temperature -> top-k -> top-p -> sampling")

    combined = prepare_sampling_distribution(
        logits,
        temperature=0.8,
        top_k=3,
        top_p=0.90,
    )

    print("组合后的分布：", combined)
    print("非零候选数：", np.count_nonzero(combined))

    assert np.isclose(combined.sum(), 1.0)
    assert 1 <= np.count_nonzero(combined) <= 3
    print("=> 每一步都只在上一步留下的候选中继续处理")

    # ------------------------------------------------------------------
    section("8) 自回归生成：采一个 token，再重新计算下一步分布")

    greedy_text = generate_toy_text(seed=1, top_k=1)
    sampled_a = generate_toy_text(
        seed=5,
        temperature=0.9,
        top_p=0.90,
    )
    sampled_b = generate_toy_text(
        seed=4,
        temperature=1.2,
        top_p=0.95,
    )

    print("top-k=1：", greedy_text)
    print("sampling A：", sampled_a)
    print("sampling B：", sampled_b)

    assert greedy_text == "我喜欢学习数学。"
    assert 1 <= len(sampled_a) <= 16
    assert 1 <= len(sampled_b) <= 16
    assert set(sampled_a).issubset(set("我喜欢学习数学代码。"))
    assert set(sampled_b).issubset(set("我喜欢学习数学代码。"))
    print("=> 一个 token 不同，会改变后续上下文和之后的全部条件概率")

    section("核心结论")
    print("1. sampling 按 categorical distribution 抽取，不是均匀随机")
    print("2. 经验频率随采样次数增加，通常趋近理论概率")
    print("3. temperature 调分布形状：低温尖锐，高温平坦")
    print("4. top-k 固定候选数；top-p 使用动态累计概率集合")
    print("5. 过滤后必须重新归一化，再进行随机抽取")
    print("6. LLM 每生成一个 token，都要基于新前文预测下一步")


if __name__ == "__main__":
    main()
