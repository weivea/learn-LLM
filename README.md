# learn-LLM

学习大语言模型（LLM）的笔记与实验仓库。

## 当前进度

- **阶段 0 · 数学与直觉**：核心门槛已通过，31/37 个知识点可独立复现
- **阶段 1 · 深度学习 + PyTorch**：准备开始，规划为 3 周 / 12 Steps

入口：

- [大模型系统学习路线](大模型学习路线.md)
- [数学知识点掌握表](数学知识点掌握表.md)
- [阶段 1 · 深度学习与 PyTorch 学习计划](阶段1-深度学习与PyTorch学习计划.md)

## 目录结构

```text
learn-LLM/
├── README.md
├── 大模型学习路线.md
├── 数学知识点掌握表.md
├── 阶段1-深度学习与PyTorch学习计划.md
├── notes/          # 分知识点讲义与自测
├── examples/       # 可独立运行的验证脚本
└── requirements.txt
```

## 快速开始

```bash
git clone https://github.com/weivea/learn-LLM.git
cd learn-LLM
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

当前 `requirements.txt` 仍对应阶段 0；PyTorch 环境会在阶段 1 Step 1.1
按当前机器与官方支持版本配置。

## 学习方式

每个 Step 使用一个独立会话，采用：

**前测 → 中文讲义 → 可运行实验 → 5 道综合自测 → 批改 → 更新掌握度**

以能独立解释和复现为完成标准，不以看完课程为完成标准。

## 许可证

本项目采用 [MIT License](LICENSE) 开源。
