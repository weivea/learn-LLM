"""
2.5 梯度下降(gradient descent) —— 验证脚本

配套知识点：数学知识点掌握表.md · 2.5
目标：用代码把「算损失 → 算梯度 → 沿负梯度更新 → 重复」跑一遍，看清：
      1) 一维二次函数怎样逐步靠近最低点
      2) 小步沿负梯度走时，损失变化约为 -η||∇L||²
      3) 学习率不同时会平稳下降、振荡或发散
      4) 多个参数怎样在同一步同时更新
      5) 线性回归怎样通过完整训练循环学到参数
      6) batch、SGD、mini-batch 的更新轨迹有何不同

运行：
    .venv/bin/python examples/2_5_gradient_descent.py
"""

import numpy as np


def section(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def quadratic_loss(w: float) -> float:
    """一维碗形损失，最低点在 w=3。"""
    return (w - 3.0) ** 2


def quadratic_grad(w: float) -> float:
    return 2.0 * (w - 3.0)


def linear_loss_and_grad(
    w: float,
    b: float,
    x: np.ndarray,
    target: np.ndarray,
) -> tuple[float, float, float]:
    """线性回归的 MSE，以及损失对 w、b 的解析梯度。"""
    prediction = w * x + b
    error = prediction - target
    loss = float(np.mean(error ** 2))
    dloss_dw = float(2.0 * np.mean(error * x))
    dloss_db = float(2.0 * np.mean(error))
    return loss, dloss_dw, dloss_db


def main() -> None:
    # ------------------------------------------------------------------
    section("1) 一维梯度下降：L(w)=(w-3)^2")
    w = 0.0
    learning_rate = 0.1
    initial_loss = quadratic_loss(w)

    print("step   当前w      梯度       新w       新损失")
    for step in range(8):
        grad = quadratic_grad(w)
        new_w = w - learning_rate * grad
        new_loss = quadratic_loss(new_w)
        print(f"{step:>3}  {w:>8.4f}  {grad:>8.4f}  {new_w:>8.4f}  {new_loss:>10.6f}")
        w = new_w

    assert quadratic_loss(w) < initial_loss
    print("=> 梯度为负时，减去负数使 w 增大；越接近 w=3，梯度和步子越小")

    # ------------------------------------------------------------------
    section("2) 为什么用减号：ΔL ≈ -η||∇L||²")
    theta = np.array([2.0, -3.0])
    grad = 2.0 * theta
    learning_rate = 1e-3
    delta = -learning_rate * grad

    old_loss = float(theta @ theta)
    new_loss = float((theta + delta) @ (theta + delta))
    predicted_change = float(grad @ delta)
    actual_change = new_loss - old_loss

    print(f"参数 θ={theta}，梯度 ∇L={grad}")
    print(f"更新量 Δθ=-η∇L={delta}")
    print(f"一阶近似的损失变化: ∇L·Δθ = {predicted_change:.8f}")
    print(f"真实损失变化:       L(θ+Δθ)-L(θ) = {actual_change:.8f}")
    print(f"两者差异:           {abs(predicted_change-actual_change):.2e}")

    assert predicted_change < 0
    assert actual_change < 0
    print("=> 步子足够小时，真实变化接近负数 -η||∇L||²，所以损失下降")

    # ------------------------------------------------------------------
    section("3) 同一个函数，不同学习率：平稳、振荡、不收敛、发散")
    rates = [0.1, 0.9, 1.0, 1.1]
    final_losses: dict[float, float] = {}

    for rate in rates:
        w = 0.0
        losses = [quadratic_loss(w)]
        positions = [w]
        for _ in range(8):
            w -= rate * quadratic_grad(w)
            positions.append(w)
            losses.append(quadratic_loss(w))
        final_losses[rate] = losses[-1]
        path = " → ".join(f"{value:.2f}" for value in positions[:6])
        print(f"η={rate:>3}: w路径 {path} ...  最终L={losses[-1]:.4f}")

    assert final_losses[0.1] < 9.0
    assert final_losses[0.9] < 9.0
    assert np.isclose(final_losses[1.0], 9.0)
    assert final_losses[1.1] > 9.0
    print("=> 负梯度决定方向，学习率决定步长；方向对了，步子太大仍会发散")

    # ------------------------------------------------------------------
    section("4) 二维梯度下降：w、b 基于同一个旧位置同时更新")
    theta = np.array([0.0, 0.0])
    target = np.array([2.0, -1.0])
    learning_rate = 0.1

    def bowl_loss(value: np.ndarray) -> float:
        return float(np.sum((value - target) ** 2))

    initial_loss = bowl_loss(theta)
    print("step        (w, b)          梯度              损失")
    for step in range(6):
        grad = 2.0 * (theta - target)
        print(
            f"{step:>3}   ({theta[0]:>6.3f},{theta[1]:>6.3f})"
            f"   ({grad[0]:>7.3f},{grad[1]:>7.3f})   {bowl_loss(theta):>9.5f}"
        )
        theta = theta - learning_rate * grad

    assert bowl_loss(theta) < initial_loss
    print(f"=> 两个参数一起逼近最低点 (2,-1)，当前 θ={theta.round(4)}")

    # ------------------------------------------------------------------
    section("5) 完整训练循环：线性回归从数据学出 y=2x+1")
    x = np.array([1.0, 2.0, 3.0, 4.0])
    target = 2.0 * x + 1.0
    w, b = 0.0, 0.0
    learning_rate = 0.05
    checkpoints = {0, 1, 2, 5, 20, 100, 500}

    print("step      w         b          loss")
    for step in range(501):
        loss, dloss_dw, dloss_db = linear_loss_and_grad(w, b, x, target)
        if step in checkpoints:
            print(f"{step:>3}   {w:>8.4f}  {b:>8.4f}  {loss:>11.8f}")
        if step == 500:
            break

        # 同一步的两个梯度都在旧 w、b 处算好，再同时更新。
        w -= learning_rate * dloss_dw
        b -= learning_rate * dloss_db

    assert loss < 1e-6
    assert np.allclose([w, b], [2.0, 1.0], atol=2e-3)
    print("=> 每一步都执行：前向算 loss → 解析梯度 → 同时更新 w,b → 重复")

    # ------------------------------------------------------------------
    section("6) batch、SGD、mini-batch：每一步使用的数据量不同")
    x = np.arange(1.0, 9.0)
    target = 1.5 * x - 0.5
    initial_loss, _, _ = linear_loss_and_grad(0.0, 0.0, x, target)

    def train(batch_size: int, epochs: int = 60) -> tuple[float, float, float, int]:
        rng = np.random.default_rng(42)
        w, b = 0.0, 0.0
        updates = 0

        for _ in range(epochs):
            indices = rng.permutation(len(x))
            for start in range(0, len(x), batch_size):
                batch_indices = indices[start : start + batch_size]
                _, dloss_dw, dloss_db = linear_loss_and_grad(
                    w,
                    b,
                    x[batch_indices],
                    target[batch_indices],
                )
                w -= 0.01 * dloss_dw
                b -= 0.01 * dloss_db
                updates += 1

        loss, _, _ = linear_loss_and_grad(w, b, x, target)
        return w, b, loss, updates

    methods = [
        ("batch", len(x)),
        ("SGD", 1),
        ("mini-batch", 2),
    ]
    for name, batch_size in methods:
        w, b, loss, updates = train(batch_size)
        print(
            f"{name:<10} batch_size={batch_size:<2}  更新{updates:>3}次"
            f"  w={w:>7.4f}  b={b:>7.4f}  全量loss={loss:.6f}"
        )
        assert loss < initial_loss

    print("=> 三者都用负梯度；区别只是每一步用全量、1 个还是一小批样本估计梯度")

    print("\n✅ 全部跑通！梯度下降 = 反复执行「算损失 → 算梯度 → 沿负梯度更新」。")


if __name__ == "__main__":
    main()
