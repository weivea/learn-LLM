"""
3.6 条件概率 P(下一个 token | 前文) —— 验证脚本

配套知识点：数学知识点掌握表.md · 3.6
目标：用数值实验看清：
      1) 如何从计数得到下一个 token 的条件分布
      2) 不同前文为什么会产生不同分布
      3) 概率链式法则如何得到整段序列概率
      4) 训练时输入和目标为什么错开一个 token
      5) 自回归生成如何不断更新条件
      6) 条件概率如何连接 NLL 与 PPL

运行：
    .venv/bin/python examples/3_6_conditional_probability.py
"""

from __future__ import annotations

import numpy as np


def section(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def conditional_distribution(counts: np.ndarray) -> np.ndarray:
    values = np.asarray(counts, dtype=np.float64)
    if values.ndim == 0 or values.size == 0:
        raise ValueError("counts 至少需要一个元素")
    if not np.all(np.isfinite(values)):
        raise ValueError("counts 不能包含 NaN 或 Inf")
    if np.any(values < 0.0):
        raise ValueError("counts 不能为负数")

    totals = values.sum(axis=-1, keepdims=True)
    if np.any(totals <= 0.0):
        raise ValueError("每个前文至少要有一次后续 token 计数")
    return values / totals


def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim == 0 or values.size == 0:
        raise ValueError("logits 至少需要一维非空输入")
    if not np.all(np.isfinite(values)):
        raise ValueError("logits 不能包含 NaN 或 Inf")

    maximum = np.max(values, axis=axis, keepdims=True)
    shifted = values - maximum
    exp_values = np.exp(shifted)
    return exp_values / exp_values.sum(axis=axis, keepdims=True)


def sequence_metrics(
    target_conditional_probabilities: np.ndarray,
) -> tuple[float, float, float, float]:
    probabilities = np.asarray(
        target_conditional_probabilities,
        dtype=np.float64,
    )
    if probabilities.ndim != 1 or probabilities.size == 0:
        raise ValueError("条件概率必须是一维非空数组")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("条件概率不能包含 NaN 或 Inf")
    if np.any(probabilities <= 0.0) or np.any(
        probabilities > 1.0
    ):
        raise ValueError("条件概率必须满足 0 < p <= 1")

    log_probability = float(np.log(probabilities).sum())
    sequence_probability = float(np.exp(log_probability))
    total_nll = -log_probability
    mean_nll = total_nll / probabilities.size
    perplexity = float(np.exp(mean_nll))
    return sequence_probability, total_nll, mean_nll, perplexity


def toy_next_token_logits(
    # 已经生成的 token，作为预测下一个 token 时的前文。
    prefix: list[str],
    # 模型允许生成的所有 token。
    vocabulary: list[str],
) -> np.ndarray:
    # 先把所有 token 的 logit 初始化为较低的 -8.0。
    scores = np.full(len(vocabulary), -8.0)
    # 使用字典推导式建立“token -> 词表索引”的映射。
    token_to_id = {
        token: token_id
        for token_id, token in enumerate(vocabulary)
    }

    # 前文为空时，提高“我”的分数，使它最可能成为第一个 token。
    if not prefix:
        scores[token_to_id["我"]] = 4.0
    # 前文最后一个 token 是“我”时，最倾向于生成“喜欢”。
    elif prefix[-1] == "我":
        scores[token_to_id["喜欢"]] = 4.0
    # 前文最后一个 token 是“喜欢”时，可以接“AI”或“数学”。
    elif prefix[-1] == "喜欢":
        # “AI”的分数稍高，因此它比“数学”更容易被采样到。
        scores[token_to_id["AI"]] = 2.0
        scores[token_to_id["数学"]] = 1.5
    # “AI”或“数学”后面最倾向于生成句号。
    elif prefix[-1] in {"AI", "数学"}:
        scores[token_to_id["。"]] = 4.0
    # 其他情况下提高结束标记的分数，让生成过程结束。
    else:
        scores[token_to_id["<EOS>"]] = 4.0

    # 返回每个 token 对应的原始分数，之后再由 softmax 转成概率。
    return scores


def generate_toy_sequence(
    # 模型允许生成的所有 token。
    vocabulary: list[str],
    # 随机种子；相同种子可以复现相同的采样结果。
    seed: int = 7,
    # 最多允许生成的 token 数，防止生成过程无限循环。
    max_new_tokens: int = 8,
) -> tuple[list[str], list[float]]:
    # 创建独立的随机数生成器，用于按概率采样 token。
    rng = np.random.default_rng(seed)
    # 保存按顺序生成的 token。
    generated: list[str] = []
    # 保存每次被选中 token 的条件概率。
    selected_probabilities: list[float] = []

    # 最多执行 max_new_tokens 轮自回归生成。
    for _ in range(max_new_tokens):
        # 根据当前已生成的前文，计算下一个 token 的 logits。
        logits = toy_next_token_logits(generated, vocabulary)
        # 使用 softmax 把 logits 转换成总和为 1 的概率分布。
        probabilities = softmax(logits)
        # 按概率分布随机选择一个词表索引，而不是总选概率最大的 token。
        token_id = int(rng.choice(len(vocabulary), p=probabilities))
        # 根据采样到的索引，从词表中取出对应 token。
        token = vocabulary[token_id]

        # 把新 token 加入前文，供下一轮预测使用。
        generated.append(token)
        # 记录本轮选中 token 的条件概率，用于后续计算序列指标。
        selected_probabilities.append(float(probabilities[token_id]))
        # 如果生成结束标记，就提前终止循环。
        if token == "<EOS>":
            break
    # 只有循环耗尽且从未执行 break 时，才会进入 for 的 else 分支。
    else:
        # 未在限制内生成结束标记，说明生成过程没有正常结束。
        raise RuntimeError("达到 max_new_tokens，但仍未生成 <EOS>")

    # 同时返回生成的 token 序列和每一步对应的条件概率。
    return generated, selected_probabilities


def main() -> None:
    np.set_printoptions(precision=6, suppress=True)

    # ------------------------------------------------------------------
    section("1) 条件概率：只在满足前文的样本中重新计算比例")

    vocabulary = np.array(["苹果", "香蕉", "代码", "文章"])
    contexts = ["我 喜欢 吃", "我 喜欢 写"]
    next_token_counts = np.array(
        [
            [6, 4, 0, 0],
            [0, 0, 7, 3],
        ]
    )
    probabilities = conditional_distribution(next_token_counts)

    for context, row in zip(contexts, probabilities):
        print(f"前文：{context}")
        for token, probability in zip(vocabulary, row):
            print(f"  P({token} | {context}) = {probability:.2f}")

    assert np.allclose(probabilities.sum(axis=-1), 1.0)
    assert np.isclose(probabilities[0, 0], 6.0 / 10.0)
    assert np.isclose(probabilities[1, 2], 7.0 / 10.0)
    print("=> 固定一个前文后，对整个词表的条件概率求和等于 1")

    # ------------------------------------------------------------------
    section("2) 同一个候选 token 的概率会随前文变化")

    apple_id = int(np.where(vocabulary == "苹果")[0][0])
    code_id = int(np.where(vocabulary == "代码")[0][0])

    print(
        "P(苹果 | 我 喜欢 吃) = "
        f"{probabilities[0, apple_id]:.2f}"
    )
    print(
        "P(苹果 | 我 喜欢 写) = "
        f"{probabilities[1, apple_id]:.2f}"
    )
    print(
        "P(代码 | 我 喜欢 吃) = "
        f"{probabilities[0, code_id]:.2f}"
    )
    print(
        "P(代码 | 我 喜欢 写) = "
        f"{probabilities[1, code_id]:.2f}"
    )

    assert probabilities[0, apple_id] > probabilities[1, apple_id]
    assert probabilities[1, code_id] > probabilities[0, code_id]
    print("=> token 没有脱离上下文的固定下一词概率")

    # ------------------------------------------------------------------
    section("3) 概率链式法则：逐步条件概率相乘得到序列概率")

    target_probabilities = np.array([0.5, 0.4, 0.25, 0.8])
    (
        sequence_probability,
        total_nll,
        mean_nll,
        perplexity,
    ) = sequence_metrics(target_probabilities)

    prediction_steps = [
        "P(我 | <BOS>)",
        "P(喜欢 | <BOS>, 我)",
        "P(AI | <BOS>, 我, 喜欢)",
        "P(<EOS> | <BOS>, 我, 喜欢, AI)",
    ]
    for label, probability in zip(
        prediction_steps,
        target_probabilities,
    ):
        print(f"{label:<37} = {probability:.2f}")

    print(f"序列概率：{sequence_probability:.6f}")
    print(f"总 NLL：  {total_nll:.6f}")
    print(f"平均 NLL：{mean_nll:.6f}")
    print(f"PPL：     {perplexity:.6f}")

    assert np.isclose(sequence_probability, 0.04)
    assert np.isclose(total_nll, -np.log(0.04))
    assert np.isclose(perplexity, 0.04 ** (-1.0 / 4.0))
    print("=> P(整段序列) = product_t P(x_t | x_<t)")

    # ------------------------------------------------------------------
    section("4) 训练标签：输入与目标错开一个 token")

    tokens = np.array(["<BOS>", "我", "喜欢", "AI", "<EOS>"])
    inputs = tokens[:-1]
    targets = tokens[1:]

    print("原序列：", tokens.tolist())
    print("模型输入：", inputs.tolist())
    print("预测目标：", targets.tolist())
    print("\n逐位置训练目标：")
    for position, target in enumerate(targets):
        visible_prefix = tokens[: position + 1]
        print(
            f"  可见 {visible_prefix.tolist()}"
            f" -> 预测 {target}"
        )

    assert inputs.shape == targets.shape
    assert np.array_equal(inputs[1:], targets[:-1])
    print("=> causal mask 允许并行训练，但每个位置都不能看到未来")

    # ------------------------------------------------------------------
    section("5) logits 的每个位置都是一套词表条件分布")

    logits = np.array(
        [
            [
                [3.0, 1.0, 0.0, -1.0],
                [0.0, 2.0, 1.0, -2.0],
                [-1.0, 0.0, 3.0, 1.0],
            ],
            [
                [1.0, 1.0, 1.0, 1.0],
                [2.0, -1.0, 0.0, 1.0],
                [0.0, 1.0, -1.0, 2.0],
            ],
        ]
    )
    next_token_distributions = softmax(logits, axis=-1)

    print("logits shape：", logits.shape)
    print("概率 shape：  ", next_token_distributions.shape)
    print("每个位置沿词表维的概率和：")
    print(next_token_distributions.sum(axis=-1))

    assert next_token_distributions.shape == logits.shape
    assert np.allclose(
        next_token_distributions.sum(axis=-1),
        1.0,
    )
    print("=> shape 为 (batch, seq_len, vocab_size)，最后一维和为 1")

    # ------------------------------------------------------------------
    section("6) 自回归生成：新 token 会成为下一步条件")

    toy_vocabulary = ["我", "喜欢", "AI", "数学", "。", "<EOS>"]
    generated, selected_probabilities = generate_toy_sequence(
        toy_vocabulary
    )

    prefix: list[str] = []
    for token, probability in zip(
        generated,
        selected_probabilities,
    ):
        print(
            f"P({token} | {prefix or ['<BOS>']})"
            f" = {probability:.6f}"
        )
        prefix.append(token)

    print("生成结果：", generated)
    assert generated[-1] == "<EOS>"
    print("=> 每生成一个 token，就追加到前文并重新计算条件分布")

    section("核心结论")
    print("1. P(A | B) = P(A ∩ B) / P(B)：条件会重新限定样本范围")
    print("2. 固定前文后，对整个词表的条件概率求和等于 1")
    print("3. P(x_1:T) = product_t P(x_t | x_<t)")
    print("4. 训练时输入与目标错开一位，causal mask 阻止偷看未来")
    print("5. 生成时把新 token 追加到前文，再计算下一步条件分布")
    print("6. total NLL = -sum_t ln P(x_t | x_<t)，PPL = exp(mean NLL)")


if __name__ == "__main__":
    main()
