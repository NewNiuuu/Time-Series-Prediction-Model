#!/usr/bin/env python3
"""
Model 模块 - 阶段3
LSTM 时序预测模型定义。

架构：
- nn.LSTM(input_size=1, hidden_size=32, num_layers=2, batch_first=True)
- nn.Linear(hidden_size, 1)

支持 GPU/CPU 自适应。
"""

import torch
import torch.nn as nn


class SalesLSTM(nn.Module):
    """
    便利店销售额预测 LSTM 模型。

    Parameters
    ----------
    input_size  : int   输入特征维度（每日营业额为1），默认1
    hidden_size : int   LSTM 隐藏层维度，默认32
    num_layers  : int   LSTM 层数，默认2
    output_size : int   输出维度（预测下一天营业额），默认1
    """

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 32,
        num_layers: int = 2,
        output_size: int = 1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,  # (batch, seq, feature)
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        Parameters
        ----------
        x : torch.Tensor  shape (batch, seq_len, input_size)

        Returns
        -------
        torch.Tensor  shape (batch, output_size)
        """
        # LSTM 输出 (output, (h_n, c_n))
        # output shape: (batch, seq_len, hidden_size)
        out, _ = self.lstm(x)
        # 只取序列最后一个时间步的输出
        out = out[:, -1, :]  # (batch, hidden_size)
        out = self.fc(out)   # (batch, output_size)
        return out
