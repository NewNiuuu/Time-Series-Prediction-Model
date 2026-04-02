# MY_LSTM - 基于 PyTorch 的 LSTM 时序预测项目

## 项目背景
本项目旨在从 0 到 1 构建一个基于 PyTorch 的 LSTM 时序预测系统。

### 当前阶段
便利店营业额预测

### 未来扩展
股票量化分析

## 目录结构

| 目录 | 说明 |
|------|------|
| `data/` | 存放原始 xlsx 文件和处理后的数据（真实业务数据不提交至版本控制） |
| `models/` | 存放训练好的模型权重 (.pth 文件) |
| `scripts/` | 存放独立的 Python 脚本，包括训练、推理、数据预处理等 |
| `docs/` | 存放学习日志 (learning_journal.md) |

## 快速开始

### 环境检查
```bash
python scripts/check_env.py
```

### 训练模型
```bash
python scripts/train.py
```

## 技术栈
- Python 3.8+
- PyTorch 2.0+
- pandas / numpy
- scikit-learn
