"""
2.6 学习率(learning rate η) —— 验证脚本

配套知识点：数学知识点掌握表.md · 2.6
目标：用代码把「学习率缩放梯度」跑一遍，看清：
      1) η、梯度和实际更新量不是同一个量
      2) 不同 η 为什么会单调收敛、振荡收敛、不收敛或发散
      3) 曲率越大，稳定学习率上限为什么越小
      4) 多维损失为什么由最陡方向限制整体学习率
      5) 太小、合适、接近上限和超过上限的学习率有何差别
      6) warmup + cosine decay 怎样随训练 step 改变 η

运行：
    .venv/bin/python examples/2_6_learning_rate.py
"""

import math

import numpy as np


def section(title: str) -> None:
    print("\n" + "=" * 76)
    print(title)
    print("=" * 76)


def quadratic_loss(w: float) -> float:
    """L(w)=(w-3)^2，最低点在 w=3。"""
    return (w - 3.0) ** 2


def quadratic_grad(w: float) -> float:
    return 2.0 * (w - 3.0)


def run_quadratic(rate: float, steps: int = 8) -> tuple[list[float], list[float]]:
    w = 0.0
    positions = [w]
    losses = [quadratic_loss(w)]

    for _ in range(steps):
        w -= rate * quadratic_grad(w)
        positions.append(w)
        losses.append(quadratic_loss(w))

    return positions, losses


def curved_loss(w: float, curvature: float, target: float = 0.0) -> float:
    """L(w)=a/2*(w-target)^2。"""
    return 0.5 * curvature * (w - target) ** 2


def run_curved_quadratic(
    rate: float,
    curvature: float,
    steps: int,
    initial_w: float = 5.0,
) -> tuple[float, float]:
    w = initial_w
    initial_loss = curved_loss(w, curvature)

    for _ in range(steps):
        grad = curvature * w
        w -= rate * grad

    return w, curved_loss(w, curvature) / initial_loss


def warmup_cosine_rate(
    step: int,
    total_steps: int,
    warmup_steps: int,
    peak_rate: float,
    min_rate: float,
) -> float:
    if not 0 <= step < total_steps:
        raise ValueError("step 必须满足 0 <= step < total_steps")
    if not 0 < warmup_steps < total_steps:
        raise ValueError("warmup_steps 必须在 0 和 total_steps 之间")
    if min_rate > peak_rate:
        raise ValueError("min_rate 不能大于 peak_rate")

    if step < warmup_steps:
        return peak_rate * (step + 1) / warmup_steps

    decay_steps = total_steps - warmup_steps
    progress = (step - warmup_steps) / (decay_steps - 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_rate + (peak_rate - min_rate) * cosine


def main() -> None:
    # ------------------------------------------------------------------
    section("1) η 不是实际移动距离：Δw = -ηg")
    w = 0.0
    grad = quadratic_grad(w)

    print(f"当前位置 w={w:.1f}，梯度 g={grad:.1f}")
    print("η       更新量 Δw=-ηg      新位置 w+Δw")
    for rate in [0.01, 0.1, 0.5, 0.9]:
        delta_w = -rate * grad
        new_w = w + delta_w
        print(f"{rate:<5.2f}   {delta_w:>10.4f}        {new_w:>10.4f}")

    assert np.isclose(-0.1 * grad, 0.6)
    print("=> 梯度决定方向与原始尺度，学习率只负责把它整体缩放")

    # ------------------------------------------------------------------
    section("2) 误差递推：e_(k+1) = (1-2η)e_k")
    cases = [
        (0.1, "单调收敛"),
        (0.5, "一步到达"),
        (0.9, "振荡收敛"),
        (1.0, "等幅振荡"),
        (1.1, "振荡发散"),
    ]
    final_losses: dict[float, float] = {}

    print("η      误差系数    w 路径（前 6 个位置）                    结果")
    for rate, result in cases:
        positions, losses = run_quadratic(rate)
        factor = 1.0 - 2.0 * rate
        final_losses[rate] = losses[-1]
        path = " → ".join(f"{value:>5.2f}" for value in positions[:6])
        print(f"{rate:<4.1f}   {factor:>8.2f}    {path}    {result}")

    assert final_losses[0.1] < 9.0
    assert np.isclose(final_losses[0.5], 0.0)
    assert final_losses[0.9] < 9.0
    assert np.isclose(final_losses[1.0], 9.0)
    assert final_losses[1.1] > 9.0
    print("=> 是否收敛看 |1-2η| 是否小于 1；符号为负只表示每步越过最低点")

    # ------------------------------------------------------------------
    section("3) 曲率决定稳定上限：0 < η < 2/a")
    rate = 0.3

    print("固定 η=0.3，比较两个不同曲率的二次函数：")
    print("曲率 a    稳定上限 2/a    误差系数 1-aη    6 步后损失/初始损失")
    ratios: dict[float, float] = {}
    for curvature in [1.0, 10.0]:
        _, ratio = run_curved_quadratic(rate, curvature, steps=6)
        ratios[curvature] = ratio
        limit = 2.0 / curvature
        factor = 1.0 - curvature * rate
        print(f"{curvature:>6.1f}       {limit:>8.3f}          {factor:>8.3f}          {ratio:>12.6f}")

    assert ratios[1.0] < 1.0
    assert ratios[10.0] > 1.0
    print("=> 同一个 η 在平缓曲面上能收敛，在陡峭曲面上可能直接发散")

    # ------------------------------------------------------------------
    section("4) 多维损失：最陡方向限制整个学习率")

    def anisotropic_loss(theta: np.ndarray) -> float:
        x, y = theta
        return 0.5 * (x**2 + 10.0 * y**2)

    def anisotropic_grad(theta: np.ndarray) -> np.ndarray:
        x, y = theta
        return np.array([x, 10.0 * y])

    def run_2d(rate: float, steps: int = 12) -> tuple[np.ndarray, list[float], list[float]]:
        theta = np.array([2.0, 2.0])
        y_path = [float(theta[1])]
        losses = [anisotropic_loss(theta)]

        for _ in range(steps):
            theta -= rate * anisotropic_grad(theta)
            y_path.append(float(theta[1]))
            losses.append(anisotropic_loss(theta))

        return theta, y_path, losses

    stable_theta, stable_y_path, stable_losses = run_2d(0.18)
    unstable_theta, unstable_y_path, unstable_losses = run_2d(0.21)

    print("L(x,y)=1/2(x²+10y²)，两个方向的曲率分别为 1 和 10")
    print("由最大曲率 10 得到整体稳定条件：η < 2/10 = 0.2")
    print(
        "η=0.18 的 y 路径:",
        " → ".join(f"{value:.3f}" for value in stable_y_path[:7]),
    )
    print(
        "η=0.21 的 y 路径:",
        " → ".join(f"{value:.3f}" for value in unstable_y_path[:7]),
    )
    print(f"η=0.18 最终 θ={stable_theta.round(4)}，loss={stable_losses[-1]:.6f}")
    print(f"η=0.21 最终 θ={unstable_theta.round(4)}，loss={unstable_losses[-1]:.6f}")

    assert stable_losses[-1] < stable_losses[0]
    assert unstable_losses[-1] > unstable_losses[0]
    print("=> x 方向较平缓也救不了失控的 y 方向；学习率必须照顾最大曲率")

    # ------------------------------------------------------------------
    section("5) 在稳定范围内，也不是越接近上限越快")
    curvature = 10.0
    steps = 20
    candidate_rates = [0.001, 0.05, 0.1, 0.19, 0.21]
    candidate_ratios: dict[float, float] = {}

    print(f"L(w)=10/2*w²，从 w=5 出发训练 {steps} 步，稳定上限为 0.2")
    print("η        误差系数    最终 w        最终损失/初始损失")
    for rate in candidate_rates:
        final_w, ratio = run_curved_quadratic(rate, curvature, steps)
        candidate_ratios[rate] = ratio
        factor = 1.0 - curvature * rate
        print(f"{rate:<6.3f}   {factor:>8.3f}     {final_w:>10.6f}          {ratio:>12.6e}")

    assert candidate_ratios[0.001] < 1.0
    assert candidate_ratios[0.05] < candidate_ratios[0.001]
    assert np.isclose(candidate_ratios[0.1], 0.0)
    assert candidate_ratios[0.19] < 1.0
    assert candidate_ratios[0.21] > 1.0
    print("=> 太小很慢；接近上限会强烈振荡；越过上限则发散")

    # ------------------------------------------------------------------
    section("6) LLM 常用 warmup + cosine decay")
    total_steps = 40
    warmup_steps = 5
    peak_rate = 0.1
    min_rate = 0.01
    selected_steps = [0, 1, 2, 3, 4, 5, 10, 20, 30, 39]
    schedule = [
        warmup_cosine_rate(
            step,
            total_steps,
            warmup_steps,
            peak_rate,
            min_rate,
        )
        for step in range(total_steps)
    ]

    print("step    学习率 η_t       阶段")
    for step in selected_steps:
        phase = "warmup" if step < warmup_steps else "cosine decay"
        print(f"{step:>3}     {schedule[step]:>10.6f}    {phase}")

    assert np.all(np.diff(schedule[:warmup_steps]) > 0)
    assert np.all(np.diff(schedule[warmup_steps:]) <= 0)
    assert np.isclose(schedule[warmup_steps - 1], peak_rate)
    assert np.isclose(schedule[warmup_steps], peak_rate)
    assert np.isclose(schedule[-1], min_rate)
    print("=> 训练初期逐步升高以保持稳定，训练后期逐步降低以便小步细调")

    print("\n✅ 全部跑通！学习率控制更新尺度；最佳范围取决于梯度、曲率和训练配置。")


if __name__ == "__main__":
    main()
