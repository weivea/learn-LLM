"""
3.3 交叉熵 / 负对数似然（NLL）—— 验证脚本

配套知识点：数学知识点掌握表.md · 3.3
目标：用数值实验看清：
      1) 正确 token 概率越高，NLL 越小
      2) one-hot 分类中，交叉熵等于正确类别的 NLL
      3) 如何从 logits 稳定计算交叉熵
      4) LLM token 损失的 shape、reduction 和 padding mask
      5) 序列概率的乘积如何变成 NLL 之和
      6) Softmax + 交叉熵的梯度为什么是 probabilities - targets

运行：
    .venv/bin/python examples/3_3_cross_entropy_nll.py
"""

import numpy as np


def section(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def log_softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)

    if values.ndim == 0:
        raise ValueError("log_softmax 至少需要一维输入")
    if values.size == 0:
        raise ValueError("log_softmax 不能处理空输入")
    if not np.all(np.isfinite(values)):
        raise ValueError("logits 不能包含 NaN 或 Inf")

    maximum = np.max(values, axis=axis, keepdims=True)
    shifted = values - maximum
    log_denominator = np.log(
        np.exp(shifted).sum(axis=axis, keepdims=True)
    )
    return shifted - log_denominator


def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    return np.exp(log_softmax(logits, axis=axis))


def cross_entropy_from_logits(
    logits: np.ndarray,
    targets: np.ndarray,
    reduction: str = "mean",
):
    values = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(targets)

    if values.ndim == 0:
        raise ValueError("logits 至少需要一维输入")
    if values.size == 0 or values.shape[-1] == 0:
        raise ValueError("logits 不能包含空的词表维")
    if not np.all(np.isfinite(values)):
        raise ValueError("logits 不能包含 NaN 或 Inf")
    if not np.issubdtype(labels.dtype, np.integer):
        raise TypeError("targets 必须是整数类别索引")
    if labels.shape != values.shape[:-1]:
        raise ValueError(
            "targets.shape 必须等于 logits.shape[:-1]"
        )
    if labels.size == 0:
        raise ValueError("targets 不能为空")
    if labels.min() < 0 or labels.max() >= values.shape[-1]:
        raise ValueError("targets 包含超出词表范围的类别索引")

    log_probabilities = log_softmax(values, axis=-1)
    selected = np.take_along_axis(
        log_probabilities,
        labels[..., np.newaxis],
        axis=-1,
    )[..., 0]
    losses = -selected

    if reduction == "none":
        return losses
    if reduction == "sum":
        return losses.sum()
    if reduction == "mean":
        return losses.mean()
    raise ValueError("reduction 必须是 none、sum 或 mean")


def masked_mean(values: np.ndarray, mask: np.ndarray) -> np.float64:
    losses = np.asarray(values, dtype=np.float64)
    valid_mask = np.asarray(mask, dtype=bool)

    if valid_mask.shape != losses.shape:
        raise ValueError("mask.shape 必须等于 losses.shape")

    valid_count = valid_mask.sum()
    if valid_count == 0:
        raise ValueError("mask 中至少需要一个有效位置")

    return np.sum(losses * valid_mask) / valid_count


def main() -> None:
    np.set_printoptions(precision=4, suppress=True)

    # ------------------------------------------------------------------
    section("1) 正确 token 概率越高，NLL 越小")

    correct_probabilities = np.array([1.0, 0.8, 0.5, 0.1, 0.01])
    nll_values = -np.log(correct_probabilities)

    print("正确 token 概率：", correct_probabilities)
    print("对应 NLL：       ", nll_values)

    assert np.isclose(nll_values[0], 0.0)
    assert np.all(np.diff(nll_values) > 0.0)
    assert nll_values[-1] > 1.0
    print("=> 概率越接近 1，损失越接近 0；概率越小，惩罚增长越快")

    # ------------------------------------------------------------------
    section("2) one-hot 交叉熵等于正确类别的 NLL")

    target_distribution = np.array([0.0, 1.0, 0.0])
    predicted_distribution = np.array([0.1, 0.7, 0.2])
    cross_entropy = -np.sum(
        target_distribution * np.log(predicted_distribution)
    )
    nll = -np.log(predicted_distribution[1])

    print(f"目标分布 q：       {target_distribution}")
    print(f"预测分布 p：       {predicted_distribution}")
    print(f"交叉熵 -Σq·ln(p)：{cross_entropy:.6f}")
    print(f"NLL -ln(p_y)：    {nll:.6f}")

    assert np.isclose(cross_entropy, nll)
    assert np.isclose(cross_entropy, 0.35667494393873245)
    print("=> one-hot 目标只保留正确类别的一项，因此交叉熵 = NLL")

    # ------------------------------------------------------------------
    section("3) 从 logits 稳定计算，避免大指数溢出")

    logits = np.array([2.0, 1.0, 0.0])
    target = np.array(0, dtype=np.int64)
    probabilities = softmax(logits)
    loss = cross_entropy_from_logits(logits, target)

    large_logits = np.array([1000.0, 1001.0, 1002.0])
    shifted_logits = np.array([-2.0, -1.0, 0.0])
    large_target = np.array(2, dtype=np.int64)
    stable_large_loss = cross_entropy_from_logits(
        large_logits,
        large_target,
    )
    equivalent_small_loss = cross_entropy_from_logits(
        shifted_logits,
        large_target,
    )

    print(f"logits：             {logits}")
    print(f"softmax：            {probabilities}")
    print(f"target=0 的损失：    {loss:.6f}")
    print(f"大 logits 的稳定损失：{stable_large_loss:.6f}")
    print(f"平移后 logits 的损失：{equivalent_small_loss:.6f}")

    assert np.isclose(loss, -np.log(probabilities[0]))
    assert np.isclose(loss, 0.40760596444438046)
    assert np.isfinite(stable_large_loss)
    assert np.isclose(stable_large_loss, equivalent_small_loss)
    print("=> 稳定 log_softmax 直接在 logits 上工作，不需要先算巨大指数")

    # ------------------------------------------------------------------
    section("4) LLM token 损失、reduction 与 padding mask")

    batch_logits = np.array(
        [
            [
                [3.0, 1.0, 0.2, -2.0],
                [0.1, 0.4, 0.3, 0.2],
                [1.0, 0.0, -1.0, -2.0],
            ],
            [
                [-1.0, 0.0, 1.0, 2.0],
                [2.0, 1.5, 0.0, -1.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
        ]
    )
    batch_targets = np.array(
        [
            [0, 1, 2],
            [3, 0, 0],
        ],
        dtype=np.int64,
    )
    valid_mask = np.array(
        [
            [True, True, True],
            [True, True, False],
        ]
    )

    token_losses = cross_entropy_from_logits(
        batch_logits,
        batch_targets,
        reduction="none",
    )
    total_loss = cross_entropy_from_logits(
        batch_logits,
        batch_targets,
        reduction="sum",
    )
    all_token_mean = cross_entropy_from_logits(
        batch_logits,
        batch_targets,
        reduction="mean",
    )
    valid_token_mean = masked_mean(token_losses, valid_mask)

    print(f"logits shape：        {batch_logits.shape}")
    print(f"targets shape：       {batch_targets.shape}")
    print(f"token losses shape：  {token_losses.shape}")
    print("每个位置的损失：")
    print(token_losses)
    print(f"全部 token 损失和：  {total_loss:.6f}")
    print(f"包含 padding 的均值：{all_token_mean:.6f}")
    print(f"仅有效 token 的均值：{valid_token_mean:.6f}")

    assert batch_logits.shape == (2, 3, 4)
    assert batch_targets.shape == (2, 3)
    assert token_losses.shape == (2, 3)
    assert valid_mask.sum() == 5
    assert np.isclose(total_loss, token_losses.sum())
    assert np.isclose(all_token_mean, token_losses.mean())
    assert np.isclose(
        valid_token_mean,
        token_losses[valid_mask].mean(),
    )
    print("=> 先保留逐 token 损失，再只对 5 个有效位置求平均")

    # ------------------------------------------------------------------
    section("5) 概率乘积经过负对数后变成损失之和")

    sequence_probabilities = np.array([0.8, 0.5, 0.25])
    sequence_likelihood = sequence_probabilities.prod()
    token_nll_values = -np.log(sequence_probabilities)
    sequence_nll_from_product = -np.log(sequence_likelihood)
    sequence_nll_from_sum = token_nll_values.sum()

    print(f"正确 token 概率： {sequence_probabilities}")
    print(f"序列似然乘积：    {sequence_likelihood:.6f}")
    print(f"逐 token NLL：   {token_nll_values}")
    print(f"序列总 NLL：     {sequence_nll_from_sum:.6f}")
    print(f"平均 token NLL： {token_nll_values.mean():.6f}")

    assert np.isclose(sequence_likelihood, 0.1)
    assert np.isclose(
        sequence_nll_from_product,
        sequence_nll_from_sum,
    )
    print("=> -ln(p1·p2·p3) = -ln(p1)-ln(p2)-ln(p3)")

    # ------------------------------------------------------------------
    section("6) 梯度：probabilities - one_hot_targets")

    gradient_logits = np.array([0.2, 1.4, -0.7])
    gradient_target = np.array(1, dtype=np.int64)
    gradient_probabilities = softmax(gradient_logits)
    target_one_hot = np.array([0.0, 1.0, 0.0])
    analytic_gradient = gradient_probabilities - target_one_hot

    epsilon = 1e-6
    numerical_gradient = np.empty_like(gradient_logits)
    for index in range(gradient_logits.size):
        positive = gradient_logits.copy()
        negative = gradient_logits.copy()
        positive[index] += epsilon
        negative[index] -= epsilon
        positive_loss = cross_entropy_from_logits(
            positive,
            gradient_target,
        )
        negative_loss = cross_entropy_from_logits(
            negative,
            gradient_target,
        )
        numerical_gradient[index] = (
            positive_loss - negative_loss
        ) / (2.0 * epsilon)

    print(f"probabilities： {gradient_probabilities}")
    print(f"target one-hot：{target_one_hot}")
    print(f"解析梯度 p-q：  {analytic_gradient}")
    print(f"数值梯度：      {numerical_gradient}")

    assert np.allclose(
        analytic_gradient,
        numerical_gradient,
        atol=1e-8,
    )
    assert analytic_gradient[1] < 0.0
    assert analytic_gradient[0] > 0.0
    assert analytic_gradient[2] > 0.0
    print("=> 梯度下降会提高正确类别 logit，压低错误类别 logits")

    section("核心结论")
    print("1. 单个 token 的 NLL 是 -ln(p_correct)")
    print("2. one-hot 分类中，交叉熵等于 NLL")
    print("3. 实现时从 logits 使用稳定 log_softmax")
    print("4. LLM 损失通常只对有效 token 求平均")
    print("5. Softmax + 交叉熵的梯度是 probabilities - targets")


if __name__ == "__main__":
    main()
