"""
3.1 概率分布（离散 / 连续）—— 验证脚本

配套知识点：数学知识点掌握表.md · 3.1
目标：用数值实验看清：
      1) 离散 PMF 如何归一化并通过求和得到事件概率
      2) token 分类分布如何采样，经验频率如何接近理论概率
      3) 连续 PDF 为什么通过区间面积得到概率
      4) PDF 为什么可以大于 1，而连续单点概率仍为 0
      5) CDF 如何统一描述离散和连续分布
      6) 数值积分如何近似正态分布的区间概率

运行：
    .venv/bin/python examples/3_1_probability_distributions.py
"""

import numpy as np


def section(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def validate_pmf(probabilities: np.ndarray) -> None:
    if probabilities.ndim != 1:
        raise ValueError("PMF 必须是一维概率向量")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("PMF 不能包含 NaN 或 Inf")
    if np.any(probabilities < 0.0):
        raise ValueError("PMF 不能包含负概率")
    if not np.isclose(probabilities.sum(), 1.0):
        raise ValueError("PMF 的概率总和必须为 1")


def uniform_pdf(
    values: np.ndarray,
    lower: float,
    upper: float,
) -> np.ndarray:
    if upper <= lower:
        raise ValueError("均匀分布必须满足 upper > lower")

    density = 1.0 / (upper - lower)
    return np.where(
        (values >= lower) & (values <= upper),
        density,
        0.0,
    )


def uniform_cdf(value: float, lower: float, upper: float) -> float:
    if upper <= lower:
        raise ValueError("均匀分布必须满足 upper > lower")
    return float(np.clip((value - lower) / (upper - lower), 0.0, 1.0))


def normal_pdf(values: np.ndarray) -> np.ndarray:
    return np.exp(-(values**2) / 2.0) / np.sqrt(2.0 * np.pi)


def trapezoid_integral(values: np.ndarray, x: np.ndarray) -> float:
    if values.shape != x.shape:
        raise ValueError("被积函数值与 x 必须具有相同形状")
    if values.ndim != 1 or len(values) < 2:
        raise ValueError("数值积分至少需要两个一维采样点")

    widths = np.diff(x)
    if np.any(widths <= 0.0):
        raise ValueError("x 必须严格递增")

    heights = (values[:-1] + values[1:]) / 2.0
    return float(np.sum(heights * widths))


def discrete_cdf(
    support: np.ndarray,
    probabilities: np.ndarray,
    value: float,
) -> float:
    validate_pmf(probabilities)
    if support.shape != probabilities.shape:
        raise ValueError("离散取值与概率必须一一对应")
    return float(probabilities[support <= value].sum())


def main() -> None:
    # ------------------------------------------------------------------
    section("1) 离散 PMF：事件概率等于对应概率质量之和")

    die_values = np.arange(1, 7)
    die_probabilities = np.array([0.05, 0.10, 0.15, 0.20, 0.20, 0.30])
    validate_pmf(die_probabilities)

    even_probability = die_probabilities[die_values % 2 == 0].sum()
    at_least_five_probability = die_probabilities[die_values >= 5].sum()

    print(f"骰子取值：{die_values}")
    print(f"PMF：    {die_probabilities}")
    print(f"概率总和：{die_probabilities.sum():.2f}")
    print(f"P(X 为偶数)={even_probability:.2f}")
    print(f"P(X >= 5)={at_least_five_probability:.2f}")

    assert np.isclose(even_probability, 0.60)
    assert np.isclose(at_least_five_probability, 0.50)
    print("=> 离散事件包含哪些候选值，就把那些值的概率质量相加")

    # ------------------------------------------------------------------
    section("2) token 分类分布：理论概率与经验频率")

    tokens = np.array(["模型", "学习", "AI", "。"])
    token_probabilities = np.array([0.55, 0.25, 0.10, 0.10])
    validate_pmf(token_probabilities)

    rng = np.random.default_rng(20260713)
    sample_size = 20_000
    sampled_indices = rng.choice(
        len(tokens),
        size=sample_size,
        p=token_probabilities,
    )
    counts = np.bincount(sampled_indices, minlength=len(tokens))
    frequencies = counts / sample_size

    print("token       理论概率       采样次数       经验频率")
    for token, probability, count, frequency in zip(
        tokens,
        token_probabilities,
        counts,
        frequencies,
    ):
        print(
            f"{token:<8}"
            f"{probability:>10.3f}"
            f"{count:>14d}"
            f"{frequency:>14.3f}"
        )

    max_error = float(np.max(np.abs(frequencies - token_probabilities)))
    assert max_error < 0.015
    print(f"最大频率误差：{max_error:.4f}")
    print("=> 一次样本不等于分布；重复采样后，经验频率通常逐渐接近理论概率")

    # ------------------------------------------------------------------
    section("3) 连续均匀分布：区间宽度 × 密度 = 区间概率")

    lower = 2.0
    upper = 6.0
    density = 1.0 / (upper - lower)
    interval_left = 3.0
    interval_right = 5.0
    interval_probability = (interval_right - interval_left) * density
    total_area = (upper - lower) * density

    print(f"X ~ U({lower:.0f}, {upper:.0f})")
    print(f"PDF 高度：1 / ({upper:.0f} - {lower:.0f}) = {density:.2f}")
    print(
        f"P({interval_left:.0f} <= X <= {interval_right:.0f})"
        f" = 区间宽度 {interval_right - interval_left:.0f}"
        f" × 密度 {density:.2f}"
        f" = {interval_probability:.2f}"
    )
    print(f"整个 PDF 面积：{total_area:.2f}")

    assert np.isclose(interval_probability, 0.50)
    assert np.isclose(total_area, 1.0)

    widths = np.array([1.0, 0.1, 0.01, 0.001])
    neighborhood_probabilities = widths * density
    print("\n包含 x=4 的区间逐渐缩小：")
    for width, probability in zip(widths, neighborhood_probabilities):
        print(f"区间宽度={width:<6g}  区间概率={probability:.6f}")
    assert neighborhood_probabilities[-1] < neighborhood_probabilities[0]
    print("精确单点宽度为 0，因此连续随机变量的 P(X=4)=0")

    # ------------------------------------------------------------------
    section("4) PDF 可以大于 1：它是密度，不是概率")

    narrow_lower = 0.0
    narrow_upper = 0.25
    narrow_density = 1.0 / (narrow_upper - narrow_lower)
    narrow_total_area = (narrow_upper - narrow_lower) * narrow_density
    small_interval_probability = (0.10 - 0.05) * narrow_density

    print(f"X ~ U({narrow_lower:.0f}, {narrow_upper:.2f})")
    print(f"PDF 高度：{narrow_density:.1f}")
    print(f"总面积：0.25 × {narrow_density:.1f} = {narrow_total_area:.1f}")
    print(f"P(0.05 <= X <= 0.10)={small_interval_probability:.1f}")

    assert narrow_density > 1.0
    assert np.isclose(narrow_total_area, 1.0)
    assert np.isclose(small_interval_probability, 0.20)
    print("=> 密度可大于 1；区间面积才是必须落在 [0, 1] 内的概率")

    # ------------------------------------------------------------------
    section("5) CDF：从左向右累计概率")

    die_cdf_at_three = discrete_cdf(
        die_values,
        die_probabilities,
        value=3,
    )
    die_cdf_at_six = discrete_cdf(
        die_values,
        die_probabilities,
        value=6,
    )
    uniform_cdf_at_three = uniform_cdf(3.0, lower=2.0, upper=6.0)
    uniform_cdf_at_six = uniform_cdf(6.0, lower=2.0, upper=6.0)

    print(f"沿用第 1 节的非均匀骰子 PMF：{die_probabilities}")
    print(f"非均匀骰子：F(3)=P(X<=3)={die_cdf_at_three:.2f}")
    print(f"非均匀骰子：F(6)=P(X<=6)={die_cdf_at_six:.2f}")
    print(f"连续 U(2,6)：F(3)={uniform_cdf_at_three:.2f}")
    print(f"连续 U(2,6)：F(6)={uniform_cdf_at_six:.2f}")

    assert np.isclose(die_cdf_at_three, 0.30)
    assert np.isclose(die_cdf_at_six, 1.0)
    assert np.isclose(uniform_cdf_at_three, 0.25)
    assert np.isclose(uniform_cdf_at_six, 1.0)
    print("=> PMF 用累加得到 CDF；PDF 用累计面积得到 CDF")

    # ------------------------------------------------------------------
    section("6) 正态 PDF：用数值积分近似区间概率")

    x_total = np.linspace(-8.0, 8.0, 160_001)
    total_probability = trapezoid_integral(normal_pdf(x_total), x_total)

    x_one_sigma = np.linspace(-1.0, 1.0, 20_001)
    one_sigma_probability = trapezoid_integral(
        normal_pdf(x_one_sigma),
        x_one_sigma,
    )
    density_at_zero = float(normal_pdf(np.array([0.0]))[0])

    print(f"标准正态 PDF 在 0 处的高度：f(0)={density_at_zero:.6f}")
    print(f"近似总面积 P(-8<=Z<=8)：{total_probability:.9f}")
    print(f"近似区间面积 P(-1<=Z<=1)：{one_sigma_probability:.6f}")
    print("连续单点概率 P(Z=0)=0")

    assert np.isclose(density_at_zero, 1.0 / np.sqrt(2.0 * np.pi))
    assert np.isclose(total_probability, 1.0, atol=1e-12)
    assert np.isclose(one_sigma_probability, 0.682689, atol=1e-6)
    print("=> f(0) 是密度高度；0.682689 才是 [-1,1] 区间下的概率面积")

    # ------------------------------------------------------------------
    section("7) 概率张量：每个位置在词表维度上归一化")

    batch_size = 2
    sequence_length = 3
    vocabulary_size = 4
    raw_scores = rng.random((batch_size, sequence_length, vocabulary_size))
    probability_tensor = raw_scores / raw_scores.sum(axis=-1, keepdims=True)
    row_sums = probability_tensor.sum(axis=-1)

    print(f"概率张量 shape：{probability_tensor.shape}")
    print("沿 vocab_size 维求和：")
    print(row_sums)

    assert probability_tensor.shape == (2, 3, 4)
    assert np.all(probability_tensor >= 0.0)
    assert np.allclose(row_sums, 1.0)
    print("=> shape=(batch, seq_len, vocab_size)，最后一维就是每个位置的 token 分布")

    section("核心结论")
    print("1. 离散 PMF：单点有概率，事件概率靠求和")
    print("2. 连续 PDF：单点概率为 0，区间概率靠面积 / 积分")
    print("3. CDF：F(x)=P(X<=x)，统一描述离散与连续分布")
    print("4. LLM：每个位置都在整个词表上产生一个离散分类分布")


if __name__ == "__main__":
    main()
