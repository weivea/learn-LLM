"""
3.4 对数（log）—— 验证脚本

配套知识点：数学知识点掌握表.md · 3.4
目标：用数值实验看清：
      1) 对数与指数互为逆运算
      2) 对数如何把乘法变成加法
      3) 概率、log-probability 与 NLL 的符号关系
      4) 长序列为什么要在 log 空间计算
      5) 如何稳定实现 logsumexp 与 log_softmax
      6) 自然对数 NLL 如何连接到困惑度

运行：
    .venv/bin/python examples/3_4_logarithms.py
"""

import numpy as np


def section(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def logsumexp(
    values: np.ndarray,
    axis: int = -1,
    keepdims: bool = False,
) -> np.ndarray:
    numbers = np.asarray(values, dtype=np.float64)

    if numbers.ndim == 0:
        raise ValueError("logsumexp 至少需要一维输入")
    if numbers.size == 0:
        raise ValueError("logsumexp 不能处理空输入")
    if not np.all(np.isfinite(numbers)):
        raise ValueError("输入不能包含 NaN 或 Inf")

    maximum = np.max(numbers, axis=axis, keepdims=True)
    shifted = numbers - maximum
    result = maximum + np.log(
        np.exp(shifted).sum(axis=axis, keepdims=True)
    )

    if keepdims:
        return result
    return np.squeeze(result, axis=axis)


def log_softmax(
    logits: np.ndarray,
    axis: int = -1,
) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)

    if values.ndim == 0:
        raise ValueError("log_softmax 至少需要一维输入")
    if values.size == 0:
        raise ValueError("log_softmax 不能处理空输入")
    if not np.all(np.isfinite(values)):
        raise ValueError("logits 不能包含 NaN 或 Inf")

    return values - logsumexp(
        values,
        axis=axis,
        keepdims=True,
    )


def main() -> None:
    np.set_printoptions(precision=6, suppress=True)

    # ------------------------------------------------------------------
    section("1) 对数与指数互为逆运算")

    exponents = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
    positive_values = np.exp(exponents)
    restored_exponents = np.log(positive_values)

    print("指数 x：       ", exponents)
    print("e^x：          ", positive_values)
    print("ln(e^x)：      ", restored_exponents)
    print(f"log_2(8)：      {np.log2(8.0):.1f}")
    print(f"log_10(0.01)：  {np.log10(0.01):.1f}")

    assert np.allclose(restored_exponents, exponents)
    assert np.isclose(np.log2(8.0), 3.0)
    assert np.isclose(np.log10(0.01), -2.0)
    assert np.isclose(np.log(1.0), 0.0)
    print("=> log_b(x)=y 与 b^y=x 描述的是同一关系")

    # ------------------------------------------------------------------
    section("2) 乘积变加法、除法变减法、幂次移到前面")

    a = 0.8
    c = 0.5
    exponent = 3.0

    product_left = np.log(a * c)
    product_right = np.log(a) + np.log(c)
    quotient_left = np.log(a / c)
    quotient_right = np.log(a) - np.log(c)
    power_left = np.log(a**exponent)
    power_right = exponent * np.log(a)

    sum_left = np.log(2.0 + 3.0)
    incorrect_sum_split = np.log(2.0) + np.log(3.0)

    print(f"ln(a*c)：          {product_left:.6f}")
    print(f"ln(a)+ln(c)：      {product_right:.6f}")
    print(f"ln(a/c)：          {quotient_left:.6f}")
    print(f"ln(a)-ln(c)：      {quotient_right:.6f}")
    print(f"ln(a^3)：          {power_left:.6f}")
    print(f"3*ln(a)：          {power_right:.6f}")
    print(f"ln(2+3)：          {sum_left:.6f}")
    print(f"ln(2)+ln(3)：      {incorrect_sum_split:.6f}")

    assert np.isclose(product_left, product_right)
    assert np.isclose(quotient_left, quotient_right)
    assert np.isclose(power_left, power_right)
    assert not np.isclose(sum_left, incorrect_sum_split)
    print("=> log 可以拆乘法、除法和幂次，但不能拆加法")

    # ------------------------------------------------------------------
    section("3) 概率、log-probability 与 NLL")

    probabilities = np.array([1.0, 0.5, 0.1, 0.01])
    log_probabilities = np.log(probabilities)
    nll_values = -log_probabilities

    print("正确 token 概率： ", probabilities)
    print("log-probability： ", log_probabilities)
    print("NLL：             ", nll_values)

    tenfold_improvement = nll_values[-1] - nll_values[-2]

    assert np.all(log_probabilities <= 0.0)
    assert np.all(nll_values >= 0.0)
    assert np.all(np.diff(nll_values) >= 0.0)
    assert np.isclose(tenfold_improvement, np.log(10.0))
    print(
        "=> 概率从 0.01 提高到 0.1，"
        f"NLL 减少 ln(10)={tenfold_improvement:.6f}"
    )

    # ------------------------------------------------------------------
    section("4) 长序列必须在 log 空间计算")

    long_sequence_probabilities = np.full(200, 0.01)
    direct_product = long_sequence_probabilities.prod()

    with np.errstate(divide="ignore"):
        log_after_product = np.log(direct_product)

    stable_log_probability = np.log(
        long_sequence_probabilities
    ).sum()
    stable_total_nll = -stable_log_probability

    print(f"0.01^200 的浮点乘积：       {direct_product}")
    print(f"先乘后取 log：              {log_after_product}")
    print(f"直接求 log-probability 之和：{stable_log_probability:.6f}")
    print(f"稳定的序列总 NLL：          {stable_total_nll:.6f}")

    assert direct_product == 0.0
    assert np.isneginf(log_after_product)
    assert np.isfinite(stable_log_probability)
    assert np.isclose(
        stable_log_probability,
        200.0 * np.log(0.01),
    )
    print("=> 真实概率不为 0；乘积只是发生下溢，log 之和仍然有限")

    # ------------------------------------------------------------------
    section("5) 稳定的 logsumexp 与 log_softmax")

    logits = np.array([1000.0, 1001.0, 1002.0])

    with np.errstate(over="ignore"):
        naive_exp_sum = np.exp(logits).sum()
        naive_logsumexp = np.log(naive_exp_sum)

    stable_logsumexp = logsumexp(logits)
    log_probabilities = log_softmax(logits)
    normalized_probabilities = np.exp(log_probabilities)

    print(f"logits：                  {logits}")
    print(f"直接 exp 后求和：         {naive_exp_sum}")
    print(f"直接 logsumexp：          {naive_logsumexp}")
    print(f"稳定 logsumexp：          {stable_logsumexp:.6f}")
    print(f"log-probabilities：       {log_probabilities}")
    print(f"还原 probabilities：     {normalized_probabilities}")
    print(
        "probabilities 总和：     "
        f"{normalized_probabilities.sum():.6f}"
    )

    assert np.isinf(naive_logsumexp)
    assert np.isfinite(stable_logsumexp)
    assert np.all(log_probabilities <= 0.0)
    assert np.isclose(normalized_probabilities.sum(), 1.0)
    assert np.allclose(
        log_probabilities,
        logits - stable_logsumexp,
    )
    print("=> 先减最大 logit，避免 e^1000 溢出")

    # ------------------------------------------------------------------
    section("6) 平均 NLL 与困惑度的连接")

    token_probabilities = np.array([0.8, 0.5, 0.25])
    token_nll_values = -np.log(token_probabilities)
    mean_nll = token_nll_values.mean()
    perplexity_from_natural_log = np.exp(mean_nll)

    token_nll_bits = -np.log2(token_probabilities)
    mean_nll_bits = token_nll_bits.mean()
    perplexity_from_base_two = 2.0**mean_nll_bits

    print(f"逐 token NLL：       {token_nll_values}")
    print(f"平均 NLL：           {mean_nll:.6f} nat")
    print(f"exp(平均 NLL)：      {perplexity_from_natural_log:.6f}")
    print(f"平均 NLL（base 2）： {mean_nll_bits:.6f} bit")
    print(f"2^(平均 bit NLL)：   {perplexity_from_base_two:.6f}")

    assert np.isclose(
        perplexity_from_natural_log,
        perplexity_from_base_two,
    )
    print("=> 对数底数会改变损失单位；逆运算保持一致时困惑度相同")

    section("核心结论")
    print("1. log 是指数的逆运算，np.log 表示自然对数")
    print("2. 对数把概率乘积变成 log-probability 之和")
    print("3. 0<p<=1 时，ln(p)<=0，所以 -ln(p)>=0")
    print("4. 长序列应在 log 空间求和，避免概率乘积下溢")
    print("5. logits 需要经过稳定 log_softmax 才是 log-probabilities")


if __name__ == "__main__":
    main()
