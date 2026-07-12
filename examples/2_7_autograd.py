"""
2.7 计算图与自动求导 —— 验证脚本

配套知识点：数学知识点掌握表.md · 2.7
目标：从零实现一个最小标量自动求导引擎，看清：
      1) 前向运算怎样建立计算图并保存本地反向规则
      2) backward 怎样按逆拓扑顺序应用链式法则
      3) 多条路径怎样通过 += 累加梯度
      4) 参数梯度为什么会跨计算图累积，以及为什么要 zero_grad
      5) 自动求导怎样驱动一个线性模型训练
      6) 张量矩阵乘法的本地反向规则怎样通过 gradient check

运行：
    .venv/bin/python examples/2_7_autograd.py
"""

from typing import Callable, Iterable, Union

import numpy as np


Number = Union[int, float]


def section(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


class Value:
    """保存标量数值、梯度和计算图依赖的最小自动求导节点。"""

    def __init__(
        self,
        data: Number,
        parents: tuple["Value", ...] = (),
        operation: str = "",
        label: str = "",
    ) -> None:
        self.data = float(data)
        self.grad = 0.0
        self.parents = parents
        self.operation = operation
        self.label = label
        self._backward: Callable[[], None] = lambda: None

    @staticmethod
    def _as_value(value: Union["Value", Number]) -> "Value":
        return value if isinstance(value, Value) else Value(value)

    def __add__(self, other: Union["Value", Number]) -> "Value":
        other = self._as_value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def backward() -> None:
            self.grad += out.grad
            other.grad += out.grad

        out._backward = backward
        return out

    def __radd__(self, other: Union["Value", Number]) -> "Value":
        return self + other

    def __mul__(self, other: Union["Value", Number]) -> "Value":
        other = self._as_value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def backward() -> None:
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = backward
        return out

    def __rmul__(self, other: Union["Value", Number]) -> "Value":
        return self * other

    def __pow__(self, exponent: Number) -> "Value":
        out = Value(self.data**exponent, (self,), f"**{exponent}")

        def backward() -> None:
            local_derivative = exponent * self.data ** (exponent - 1)
            self.grad += local_derivative * out.grad

        out._backward = backward
        return out

    def __neg__(self) -> "Value":
        return self * -1

    def __sub__(self, other: Union["Value", Number]) -> "Value":
        other = self._as_value(other)
        out = Value(self.data - other.data, (self, other), "-")

        def backward() -> None:
            self.grad += out.grad
            other.grad -= out.grad

        out._backward = backward
        return out

    def __rsub__(self, other: Union["Value", Number]) -> "Value":
        return self._as_value(other) - self

    def relu(self) -> "Value":
        out = Value(max(0.0, self.data), (self,), "ReLU")

        def backward() -> None:
            self.grad += (1.0 if self.data > 0 else 0.0) * out.grad

        out._backward = backward
        return out

    def topological_order(self) -> list["Value"]:
        order: list[Value] = []
        visited: set[Value] = set()

        def visit(node: Value) -> None:
            if node in visited:
                return
            visited.add(node)
            for parent in node.parents:
                visit(parent)
            order.append(node)

        visit(self)
        return order

    def backward(self) -> None:
        order = self.topological_order()
        self.grad = 1.0
        for node in reversed(order):
            node._backward()

    def __repr__(self) -> str:
        name = self.label or "unnamed"
        return f"Value(label={name!r}, data={self.data:.6f}, grad={self.grad:.6f})"


def zero_grad(parameters: Iterable[Value]) -> None:
    for parameter in parameters:
        parameter.grad = 0.0


def numerical_derivative(
    function: Callable[[float], float],
    x: float,
    h: float = 1e-5,
) -> float:
    return (function(x + h) - function(x - h)) / (2 * h)


def main() -> None:
    # ------------------------------------------------------------------
    section("1) 前向传播：运算结果自动记录父节点和运算类型")

    w = Value(1.5, label="w")
    x = Value(2.0, label="x")
    b = Value(0.5, label="b")
    target = Value(3.0, label="target")

    product = w * x
    product.label = "product"
    prediction = product + b
    prediction.label = "prediction"
    error = prediction - target
    error.label = "error"
    loss = error**2
    loss.label = "loss"

    print("前向数值:")
    for node in [w, x, b, target, product, prediction, error, loss]:
        parent_names = [parent.label or "constant" for parent in node.parents]
        print(
            f"  {node.label:<10} data={node.data:>6.3f}  "
            f"operation={node.operation or 'leaf':>5}  parents={parent_names}"
        )

    assert np.isclose(loss.data, 0.25)
    print("=> 前向不仅得到 loss=0.25，也保留了 loss 是怎样一步步算出来的")

    # ------------------------------------------------------------------
    section("2) backward：逆拓扑执行“上游梯度 × 本地导数”")

    order = loss.topological_order()
    print("前向拓扑顺序:", " → ".join(node.label or "constant" for node in order))
    print("反向执行顺序:", " → ".join(node.label or "constant" for node in reversed(order)))

    loss.backward()

    print("\n自动求导结果:")
    for node in [loss, error, prediction, product, w, b, x, target]:
        print(f"  dloss/d{node.label:<10} = {node.grad:>7.3f}")

    expected = {"w": 2.0, "b": 1.0, "x": 1.5, "target": -1.0}
    assert np.isclose(w.grad, expected["w"])
    assert np.isclose(b.grad, expected["b"])
    assert np.isclose(x.grad, expected["x"])
    assert np.isclose(target.grad, expected["target"])
    print("=> 自动梯度与手推一致：dw=2, db=1, dx=1.5, dtarget=-1")

    # ------------------------------------------------------------------
    section("3) 共享节点：一条路径内相乘，多条路径之间相加")

    shared_x = Value(3.0, label="shared_x")
    square = shared_x * shared_x
    square.label = "square"
    shared_y = square + shared_x
    shared_y.label = "shared_y"
    shared_y.backward()

    print("y = x*x + x，在 x=3 时：")
    print(f"  左乘法输入贡献 = 3")
    print(f"  右乘法输入贡献 = 3")
    print(f"  直接相加路径贡献 = 1")
    print(f"  自动累加得到 x.grad = {shared_x.grad:.1f}")

    assert np.isclose(shared_x.grad, 7.0)
    print("=> 必须使用 grad += contribution；若用 =，后到的路径会覆盖前面的贡献")

    # ------------------------------------------------------------------
    section("4) gradient check：自动求导 ≈ 中心差分")

    auto_w = w.grad
    auto_b = b.grad

    def scalar_loss_w(candidate_w: float) -> float:
        return (candidate_w * 2.0 + 0.5 - 3.0) ** 2

    def scalar_loss_b(candidate_b: float) -> float:
        return (1.5 * 2.0 + candidate_b - 3.0) ** 2

    numeric_w = numerical_derivative(scalar_loss_w, 1.5)
    numeric_b = numerical_derivative(scalar_loss_b, 0.5)

    print("参数    自动求导      数值梯度      差")
    print(f"w       {auto_w:>8.5f}    {numeric_w:>8.5f}    {abs(auto_w-numeric_w):.1e}")
    print(f"b       {auto_b:>8.5f}    {numeric_b:>8.5f}    {abs(auto_b-numeric_b):.1e}")

    assert np.isclose(auto_w, numeric_w, atol=1e-8)
    assert np.isclose(auto_b, numeric_b, atol=1e-8)
    print("=> 数值差分适合检查实现；实际训练使用自动求导")

    # ------------------------------------------------------------------
    section("5) 参数梯度默认累积：两个新计算图的贡献相加")

    parameter = Value(2.0, label="parameter")

    first_loss = parameter * 3
    first_loss.label = "first_loss"
    first_loss.backward()
    first_grad = parameter.grad

    second_loss = parameter**2
    second_loss.label = "second_loss"
    second_loss.backward()
    accumulated_grad = parameter.grad

    print(f"第一次：d(3p)/dp = {first_grad:.1f}")
    print(f"第二次：d(p²)/dp = 4.0")
    print(f"未清零后的 parameter.grad = {accumulated_grad:.1f}")

    assert np.isclose(first_grad, 3.0)
    assert np.isclose(accumulated_grad, 7.0)

    zero_grad([parameter])
    third_loss = parameter**2
    third_loss.backward()
    print(f"zero_grad 后重新反传 p²，parameter.grad = {parameter.grad:.1f}")

    assert np.isclose(parameter.grad, 4.0)
    print("=> 累积可用于 gradient accumulation；普通训练 step 之间则必须按需清零")

    # ------------------------------------------------------------------
    section("6) 自动求导驱动训练：拟合 y = 2x + 1")

    inputs = [1.0, 2.0, 3.0, 4.0]
    targets = [3.0, 5.0, 7.0, 9.0]
    train_w = Value(-1.0, label="train_w")
    train_b = Value(0.0, label="train_b")
    learning_rate = 0.05

    for step in range(101):
        losses: list[Value] = []
        for input_value, target_value in zip(inputs, targets):
            prediction_value = train_w * input_value + train_b
            sample_error = prediction_value - target_value
            losses.append(sample_error**2)

        mean_loss = sum(losses, Value(0.0)) * (1.0 / len(losses))

        zero_grad([train_w, train_b])
        mean_loss.backward()

        train_w.data -= learning_rate * train_w.grad
        train_b.data -= learning_rate * train_b.grad

        if step in [0, 1, 2, 5, 20, 50, 100]:
            print(
                f"step={step:>3}  loss={mean_loss.data:>10.6f}  "
                f"w={train_w.data:>8.4f}  b={train_b.data:>8.4f}"
            )

    assert np.isclose(train_w.data, 2.0, atol=2e-2)
    assert np.isclose(train_b.data, 1.0, atol=5e-2)
    print("=> 每步都是：新前向图 → backward 算梯度 → 手动更新参数 → 图被丢弃")

    # ------------------------------------------------------------------
    section("7) 张量节点：矩阵乘法本地反向规则")

    matrix_x = np.array([[1.0, 2.0], [-1.0, 3.0]])
    matrix_w = np.array([[0.5, -2.0], [1.5, 1.0]])
    upstream = np.array([[2.0, -1.0], [0.5, 3.0]])

    # 若 Y=XW，且上游梯度为 G=dL/dY，则 dL/dX=GW^T，dL/dW=X^TG。
    grad_x = upstream @ matrix_w.T
    grad_w = matrix_x.T @ upstream

    def tensor_loss(candidate_w: np.ndarray) -> float:
        return float(np.sum((matrix_x @ candidate_w) * upstream))

    numeric_grad_w = np.zeros_like(matrix_w)
    h = 1e-5
    for row in range(matrix_w.shape[0]):
        for column in range(matrix_w.shape[1]):
            plus = matrix_w.copy()
            minus = matrix_w.copy()
            plus[row, column] += h
            minus[row, column] -= h
            numeric_grad_w[row, column] = (
                tensor_loss(plus) - tensor_loss(minus)
            ) / (2 * h)

    print("X shape:", matrix_x.shape)
    print("W shape:", matrix_w.shape)
    print("上游 G=dL/dY shape:", upstream.shape)
    print("dL/dX = G @ W.T:\n", grad_x)
    print("dL/dW = X.T @ G:\n", grad_w)
    print("数值检查 dL/dW:\n", numeric_grad_w)

    assert grad_x.shape == matrix_x.shape
    assert grad_w.shape == matrix_w.shape
    assert np.allclose(grad_w, numeric_grad_w, atol=1e-8)
    print("=> 张量自动求导仍是同一原理，只是本地导数规则变成矩阵运算")

    print("\n✅ 全部跑通！自动求导 = 记录计算图 + 逆拓扑链式法则 + 多路径梯度累加。")


if __name__ == "__main__":
    main()
