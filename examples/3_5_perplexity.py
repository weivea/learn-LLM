"""
3.5 困惑度（perplexity）—— 验证脚本

配套知识点：数学知识点掌握表.md · 3.5
目标：用数值实验看清：
      1) PPL 如何由平均 NLL 得到
      2) PPL 为什么是几何平均概率的倒数
      3) 为什么必须使用平均 NLL 和有效 token mask
      4) 如何从 logits 稳定计算 PPL
      5) PPL 的比较条件和指标边界

运行：
    .venv/bin/python examples/3_5_perplexity.py
"""

from __future__ import annotations

import numpy as np


def section(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def validate_mask(
    mask: np.ndarray | None,
    expected_shape: tuple[int, ...],
) -> np.ndarray:
    if mask is None:
        return np.ones(expected_shape, dtype=bool)

    values = np.asarray(mask)
    if values.shape != expected_shape:
        raise ValueError(
            f"mask shape 应为 {expected_shape}，实际为 {values.shape}"
        )
    if values.dtype != np.bool_ and not np.all(
        (values == 0) | (values == 1)
    ):
        raise ValueError("mask 只能包含布尔值或 0/1")

    valid = values.astype(bool)
    if not np.any(valid):
        raise ValueError("mask 至少要包含一个有效 token")
    return valid


def mean_nll_and_perplexity(
    token_nll: np.ndarray,
    mask: np.ndarray | None = None,
) -> tuple[float, float]:
    losses = np.asarray(token_nll, dtype=np.float64)
    if losses.ndim == 0 or losses.size == 0:
        raise ValueError("token_nll 至少需要一个元素")
    if not np.all(np.isfinite(losses)):
        raise ValueError("token_nll 不能包含 NaN 或 Inf")
    if np.any(losses < 0.0):
        raise ValueError("NLL 不能为负数")

    valid = validate_mask(mask, losses.shape)
    mean_nll = float(losses[valid].mean())
    perplexity = float(np.exp(mean_nll))
    if not np.isfinite(perplexity):
        raise OverflowError("平均 NLL 过大，PPL 超出 float64 范围")
    return mean_nll, perplexity


def perplexity_from_probabilities(
    probabilities: np.ndarray,
    mask: np.ndarray | None = None,
) -> tuple[float, float]:
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim == 0 or values.size == 0:
        raise ValueError("probabilities 至少需要一个元素")
    if not np.all(np.isfinite(values)):
        raise ValueError("probabilities 不能包含 NaN 或 Inf")
    if np.any(values <= 0.0) or np.any(values > 1.0):
        raise ValueError("真实 token 概率必须满足 0 < p <= 1")

    return mean_nll_and_perplexity(-np.log(values), mask)


def log_softmax(
    logits: np.ndarray,
    axis: int = -1,
) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim == 0 or values.size == 0:
        raise ValueError("logits 至少需要一维非空输入")
    if not np.all(np.isfinite(values)):
        raise ValueError("logits 不能包含 NaN 或 Inf")

    maximum = np.max(values, axis=axis, keepdims=True)
    shifted = values - maximum
    return shifted - np.log(
        np.exp(shifted).sum(axis=axis, keepdims=True)
    )


def perplexity_from_logits(
    logits: np.ndarray,
    targets: np.ndarray,
    mask: np.ndarray | None = None,
) -> tuple[float, float]:
    values = np.asarray(logits, dtype=np.float64)
    target_ids = np.asarray(targets)

    if values.ndim < 2:
        raise ValueError("logits shape 至少应为 (..., vocab_size)")
    if values.shape[-1] == 0:
        raise ValueError("vocab_size 不能为 0")
    if target_ids.shape != values.shape[:-1]:
        raise ValueError(
            "targets shape 应等于 logits.shape[:-1]："
            f"{values.shape[:-1]}"
        )
    if not np.issubdtype(target_ids.dtype, np.integer):
        raise TypeError("targets 必须是整数 token ID")
    if np.any(target_ids < 0) or np.any(
        target_ids >= values.shape[-1]
    ):
        raise ValueError("targets 包含超出词表范围的 token ID")

    log_probabilities = log_softmax(values, axis=-1)
    target_log_probabilities = np.take_along_axis(
        log_probabilities,
        target_ids[..., np.newaxis],
        axis=-1,
    )[..., 0]
    token_nll = -target_log_probabilities
    return mean_nll_and_perplexity(token_nll, mask)


def main() -> None:
    np.set_printoptions(precision=6, suppress=True)

    # ------------------------------------------------------------------
    section("1) 从真实 token 概率计算平均 NLL 和 PPL")

    token_probabilities = np.array([0.5, 0.25, 0.125])
    token_nll = -np.log(token_probabilities)
    mean_nll, perplexity = perplexity_from_probabilities(
        token_probabilities
    )

    print("真实 token 概率：", token_probabilities)
    print("逐 token NLL：    ", token_nll)
    print(f"平均 NLL：         {mean_nll:.6f}")
    print(f"PPL：              {perplexity:.6f}")

    assert np.allclose(
        token_nll,
        np.log(np.array([2.0, 4.0, 8.0])),
    )
    assert np.isclose(mean_nll, np.log(4.0))
    assert np.isclose(perplexity, 4.0)
    print("=> PPL = exp(平均 NLL) = 4")

    # ------------------------------------------------------------------
    section("2) PPL 是几何平均概率的倒数")

    geometric_mean_probability = float(
        np.exp(np.log(token_probabilities).mean())
    )
    reciprocal = 1.0 / geometric_mean_probability

    print(f"概率的算术平均：       {token_probabilities.mean():.6f}")
    print(f"概率的几何平均：       {geometric_mean_probability:.6f}")
    print(f"几何平均的倒数：       {reciprocal:.6f}")
    print(f"由平均 NLL 得到的 PPL：{perplexity:.6f}")

    assert not np.isclose(
        token_probabilities.mean(),
        geometric_mean_probability,
    )
    assert np.isclose(geometric_mean_probability, 0.25)
    assert np.isclose(reciprocal, perplexity)

    uniform_target_probabilities = np.full(6, 1.0 / 5.0)
    _, uniform_perplexity = perplexity_from_probabilities(
        uniform_target_probabilities
    )
    assert np.isclose(uniform_perplexity, 5.0)
    print("=> 若真实 token 概率始终为 1/K，则 PPL = K")

    # ------------------------------------------------------------------
    section("3) 必须使用平均 NLL，而不是总 NLL")

    short_nll = np.full(2, 0.3)
    long_nll = np.full(20, 0.3)
    short_mean, short_perplexity = mean_nll_and_perplexity(short_nll)
    long_mean, long_perplexity = mean_nll_and_perplexity(long_nll)

    print(f"短序列总 NLL： {short_nll.sum():.6f}")
    print(f"长序列总 NLL： {long_nll.sum():.6f}")
    print(f"短序列平均 NLL：{short_mean:.6f}")
    print(f"长序列平均 NLL：{long_mean:.6f}")
    print(f"短序列 PPL：    {short_perplexity:.6f}")
    print(f"长序列 PPL：    {long_perplexity:.6f}")

    assert np.isclose(
        long_nll.sum(),
        10.0 * short_nll.sum(),
    )
    assert np.isclose(short_mean, long_mean)
    assert np.isclose(short_perplexity, long_perplexity)
    print("=> 总 NLL 随长度增长；平均 NLL 和 PPL 衡量每 token 水平")

    # ------------------------------------------------------------------
    section("4) mask 只保留需要评分的有效 token")

    batched_probabilities = np.array(
        [
            [0.8, 0.5, 0.01],
            [0.4, 0.2, 0.01],
        ]
    )
    mask = np.array(
        [
            [True, True, False],
            [True, False, False],
        ]
    )

    masked_mean_nll, masked_perplexity = (
        perplexity_from_probabilities(
            batched_probabilities,
            mask,
        )
    )
    unmasked_mean_nll, unmasked_perplexity = (
        perplexity_from_probabilities(batched_probabilities)
    )

    changed_padding = batched_probabilities.copy()
    changed_padding[~mask] = 0.9
    _, changed_padding_perplexity = perplexity_from_probabilities(
        changed_padding,
        mask,
    )

    print("概率：")
    print(batched_probabilities)
    print("mask：")
    print(mask)
    print(f"使用 mask 的平均 NLL：{masked_mean_nll:.6f}")
    print(f"使用 mask 的 PPL：    {masked_perplexity:.6f}")
    print(f"错误包含 padding 的 PPL：{unmasked_perplexity:.6f}")

    assert unmasked_mean_nll > masked_mean_nll
    assert unmasked_perplexity > masked_perplexity
    assert np.isclose(
        changed_padding_perplexity,
        masked_perplexity,
    )
    print("=> 被 mask 的数值不会影响结果，分母是有效 token 数")

    # ------------------------------------------------------------------
    section("5) 从 logits 稳定计算 PPL")

    logits = np.array(
        [
            [
                [1000.0, 1001.0, 1002.0, 999.0],
                [2.0, 1.0, 0.0, -1.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            [
                [1.0, 3.0, 2.0, 0.0],
                [-1.0, 0.0, 1.0, 2.0],
                [5.0, 4.0, 3.0, 2.0],
            ],
        ]
    )
    targets = np.array(
        [
            [2, 0, 1],
            [1, 3, 0],
        ]
    )
    score_mask = np.array(
        [
            [True, True, False],
            [True, True, True],
        ]
    )

    logits_mean_nll, logits_perplexity = perplexity_from_logits(
        logits,
        targets,
        score_mask,
    )
    log_probabilities = log_softmax(logits, axis=-1)
    target_probabilities = np.exp(
        np.take_along_axis(
            log_probabilities,
            targets[..., np.newaxis],
            axis=-1,
        )[..., 0]
    )
    probability_mean_nll, probability_perplexity = (
        perplexity_from_probabilities(
            target_probabilities,
            score_mask,
        )
    )

    print("目标 token 概率：")
    print(target_probabilities)
    print(f"从 logits 得到的平均 NLL：{logits_mean_nll:.6f}")
    print(f"从 logits 得到的 PPL：    {logits_perplexity:.6f}")

    assert np.allclose(
        np.exp(log_probabilities).sum(axis=-1),
        1.0,
    )
    assert np.isclose(logits_mean_nll, probability_mean_nll)
    assert np.isclose(logits_perplexity, probability_perplexity)
    print("=> stable log_softmax → 目标 log-prob → mask 平均 → exp")

    # ------------------------------------------------------------------
    section("6) 对数底数不同，匹配逆运算后 PPL 相同")

    mean_nll_nats = -np.log(token_probabilities).mean()
    mean_nll_bits = -np.log2(token_probabilities).mean()
    perplexity_nats = np.exp(mean_nll_nats)
    perplexity_bits = 2.0**mean_nll_bits

    print(f"平均 NLL（nat）：{mean_nll_nats:.6f}")
    print(f"exp(nat NLL)：   {perplexity_nats:.6f}")
    print(f"平均 NLL（bit）：{mean_nll_bits:.6f}")
    print(f"2^(bit NLL)：    {perplexity_bits:.6f}")

    assert np.isclose(perplexity_nats, perplexity_bits)
    print("=> np.log 配 np.exp；np.log2 配 2**x")

    section("核心结论")
    print("1. PPL = exp(有效 token 的平均 NLL)，相同条件下越低越好")
    print("2. PPL = 1 / 真实 token 概率的几何平均，且 PPL >= 1")
    print("3. 必须使用 mask 和平均 NLL，不能直接对总 NLL 取指数")
    print("4. 从 logits 计算时使用稳定 log_softmax，不做序列概率连乘")
    print("5. 不同 tokenizer、数据集或上下文策略的 PPL 不能直接横比")
    print("6. PPL 不等于正确率，也不完整代表推理、事实性或安全性")


if __name__ == "__main__":
    main()
