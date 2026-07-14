"""
3.2 Softmax —— 验证脚本

配套知识点：数学知识点掌握表.md · 3.2
目标：用数值实验看清：
      1) logits 如何转换为合法概率分布
      2) softmax 为什么保留大小顺序且具有平移不变性
      3) 为什么稳定实现必须先减去最大值
      4) temperature 如何控制分布的尖锐程度
      5) LLM 输出张量应沿 vocab_size 维归一化
      6) 注意力分数如何逐行转换为权重

运行：
    .venv/bin/python examples/3_2_softmax.py
"""

import numpy as np


def section(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def softmax(
    logits: np.ndarray,
    axis: int = -1,
    temperature: float = 1.0,
) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)

    if values.ndim == 0:
        raise ValueError("softmax 至少需要一维输入")
    if values.size == 0:
        raise ValueError("softmax 不能处理空输入")
    if not np.all(np.isfinite(values)):
        raise ValueError("logits 不能包含 NaN 或 Inf")
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature 必须是有限正数")

    maximum = np.max(values, axis=axis, keepdims=True)
    shifted = (values - maximum) / temperature
    exp_values = np.exp(shifted)
    denominator = exp_values.sum(axis=axis, keepdims=True)
    return exp_values / denominator


def main() -> None:
    np.set_printoptions(precision=4, suppress=True)

    # ------------------------------------------------------------------
    section("1) 手算对应：logits -> 指数 -> 归一化概率")

    logits = np.array([2.0, 1.0, 0.0])
    exp_values = np.exp(logits)
    probabilities = softmax(logits)

    print(f"logits：       {logits}")
    print(f"exp(logits)：  {exp_values}")
    print(f"softmax：      {probabilities}")
    print(f"概率总和：     {probabilities.sum():.12f}")

    expected = np.array([0.66524096, 0.24472847, 0.09003057])
    assert np.allclose(probabilities, expected)
    assert np.all(probabilities > 0.0)
    assert np.isclose(probabilities.sum(), 1.0)
    print("=> 指数保证每项为正，除以指数总和保证概率和为 1")

    # ------------------------------------------------------------------
    section("2) Softmax 保留 logits 的大小顺序")

    ranked_logits = np.array([-1.5, 3.2, 0.7, 2.1])
    ranked_probabilities = softmax(ranked_logits)
    logit_order = np.argsort(ranked_logits)
    probability_order = np.argsort(ranked_probabilities)

    print(f"logits：       {ranked_logits}")
    print(f"probabilities：{ranked_probabilities}")
    print(f"logits 最大项位置：       {np.argmax(ranked_logits)}")
    print(f"probabilities 最大项位置：{np.argmax(ranked_probabilities)}")

    assert np.array_equal(logit_order, probability_order)
    assert np.argmax(ranked_logits) == np.argmax(ranked_probabilities)
    print("=> softmax 改变数值表示，但不会改变类别的大小排序")

    # ------------------------------------------------------------------
    section("3) 平移不变性：所有 logits 加同一个常数，结果不变")

    base = np.array([2.0, 1.0, 0.0])
    shifted_up = base + 100.0
    shifted_down = base - np.max(base)

    base_probabilities = softmax(base)
    up_probabilities = softmax(shifted_up)
    down_probabilities = softmax(shifted_down)

    print(f"softmax({base}) = {base_probabilities}")
    print(f"softmax({shifted_up}) = {up_probabilities}")
    print(f"softmax({shifted_down}) = {down_probabilities}")

    assert np.allclose(base_probabilities, up_probabilities)
    assert np.allclose(base_probabilities, down_probabilities)
    print("=> softmax 只关心 logits 之间的相对差距")

    # ------------------------------------------------------------------
    section("4) 数值稳定性：先减最大值，避免 exp 溢出")

    large_logits = np.array([1000.0, 1001.0, 1002.0])
    with np.errstate(over="ignore", invalid="ignore"):
        naive_exp = np.exp(large_logits)
        naive_probabilities = naive_exp / naive_exp.sum()

    stable_probabilities = softmax(large_logits)
    equivalent_probabilities = softmax(np.array([-2.0, -1.0, 0.0]))

    print(f"直接 exp：       {naive_exp}")
    print(f"直接归一化：      {naive_probabilities}")
    print(f"稳定 softmax：   {stable_probabilities}")
    print(f"等价小 logits：  {equivalent_probabilities}")

    assert not np.all(np.isfinite(naive_probabilities))
    assert np.all(np.isfinite(stable_probabilities))
    assert np.allclose(stable_probabilities, equivalent_probabilities)
    assert np.isclose(stable_probabilities.sum(), 1.0)
    print("=> 减去最大值不改变数学结果，却能避免正向指数溢出")

    # ------------------------------------------------------------------
    section("5) Temperature：控制概率分布的尖锐程度")

    temperatures = (0.5, 1.0, 2.0)
    temperature_probabilities = {
        temperature: softmax(base, temperature=temperature)
        for temperature in temperatures
    }

    print("T       probabilities                         最大概率")
    for temperature, values in temperature_probabilities.items():
        print(
            f"{temperature:<7.1f}"
            f"{str(values):<38}"
            f"{values.max():.4f}"
        )

    cold = temperature_probabilities[0.5]
    normal = temperature_probabilities[1.0]
    hot = temperature_probabilities[2.0]

    assert cold.max() > normal.max() > hot.max()
    assert cold.min() < normal.min() < hot.min()
    assert np.argmax(cold) == np.argmax(normal) == np.argmax(hot)
    print("=> T 越低分布越集中；T 越高分布越平坦，但最大项位置不变")

    # ------------------------------------------------------------------
    section("6) LLM 输出张量：沿 vocab_size 维做 softmax")

    output_logits = np.array(
        [
            [[2.0, 1.0, 0.0, -1.0], [0.2, 0.4, 0.1, 0.3]],
            [[-1.0, 0.0, 1.0, 2.0], [3.0, 1.0, 1.0, 1.0]],
        ]
    )
    output_probabilities = softmax(output_logits, axis=-1)
    row_sums = output_probabilities.sum(axis=-1)

    print(f"logits shape：        {output_logits.shape}")
    print(f"probabilities shape： {output_probabilities.shape}")
    print("沿 vocab_size 维求和：")
    print(row_sums)

    assert output_probabilities.shape == (2, 2, 4)
    assert row_sums.shape == (2, 2)
    assert np.allclose(row_sums, 1.0)
    print("=> 每个 (batch, position) 都得到一个独立的词表概率分布")

    # ------------------------------------------------------------------
    section("7) 轴选错：数值可以归一化，但语义已经错误")

    wrong_axis_probabilities = softmax(output_logits, axis=0)
    correct_vocab_sums = wrong_axis_probabilities.sum(axis=-1)
    wrong_batch_sums = wrong_axis_probabilities.sum(axis=0)

    print("错误地沿 batch 轴归一化后，沿 vocab_size 求和：")
    print(correct_vocab_sums)
    print("沿错误选择的 batch 轴求和：")
    print(wrong_batch_sums)

    assert not np.allclose(correct_vocab_sums, 1.0)
    assert np.allclose(wrong_batch_sums, 1.0)
    print("=> 判断 axis 必须看维度语义，不能只看结果中是否出现了和为 1")

    # ------------------------------------------------------------------
    section("8) 注意力：每行分数转换成对可见位置的权重")

    attention_scores = np.array(
        [
            [2.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
            [-1.0, 1.0, 3.0],
        ]
    )
    attention_weights = softmax(attention_scores, axis=-1)

    print("attention scores：")
    print(attention_scores)
    print("attention weights：")
    print(attention_weights)
    print(f"每行权重之和：{attention_weights.sum(axis=-1)}")

    assert np.allclose(attention_weights.sum(axis=-1), 1.0)
    assert np.allclose(attention_weights[1], np.full(3, 1.0 / 3.0))
    print("=> 分数相同得到均匀权重；分数不同则向高分位置分配更多权重")

    # ------------------------------------------------------------------
    section("9) 从词表概率分布中采样")

    tokens = np.array(["模型", "学习", "天气"])
    token_probabilities = softmax(np.array([2.0, 1.0, 0.0]))
    rng = np.random.default_rng(20260713)
    sampled_indices = rng.choice(
        len(tokens),
        size=20_000,
        p=token_probabilities,
    )
    frequencies = np.bincount(
        sampled_indices,
        minlength=len(tokens),
    ) / len(sampled_indices)

    print(f"tokens：       {tokens}")
    print(f"理论概率：      {token_probabilities}")
    print(f"经验频率：      {frequencies}")

    assert np.max(np.abs(frequencies - token_probabilities)) < 0.015
    print("=> softmax 产生分布；选择或采样才产生一个具体 token")

    section("核心结论")
    print("1. softmax 把任意实数 logits 转成总和为 1 的正概率")
    print("2. 所有 logits 加同一个常数不会改变结果")
    print("3. 稳定实现先减最大值，再取指数并归一化")
    print("4. 低温更尖锐，高温更平坦")
    print("5. LLM 输出层通常沿最后的 vocab_size 维做 softmax")


if __name__ == "__main__":
    main()
