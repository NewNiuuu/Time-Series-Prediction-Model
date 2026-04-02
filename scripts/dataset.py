#!/usr/bin/env python3
"""
Dataset 模块 - 阶段3
职责：
1. 从 CSV 文件读取每日归一化营业额数据
2. 执行滑动窗口切分，返回 PyTorch Dataset
3. 按时间顺序切分训练/验证/测试集，返回 DataLoader

本模块不涉及数据清洗、标准化等上游处理，解耦清晰。
"""

import logging
from typing import Tuple

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class SalesDataset(Dataset):
    """
    便利店销售额时序数据集。

    从 CSV 加载 (date, normalized_amount) 数据，
    按 seq_length 滑动窗口切分，返回 LSTM 所需格式。

    Parameters
    ----------
    csv_path   : str   processed_daily_sales.csv 文件路径
    seq_length : int   滑动窗口长度（用多少天预测下一天），默认14
    """

    def __init__(self, csv_path: str, seq_length: int = 14):
        self.seq_length = seq_length

        df = pd.read_csv(csv_path)
        self.data = df["normalized_amount"].values.astype(np.float32)

        logger.info(f"SalesDataset 初始化: 共 {len(self.data)} 天数据, seq_length={seq_length}")
        self._validate()

    def _validate(self) -> None:
        if len(self.data) <= self.seq_length:
            raise ValueError(
                f"数据长度({len(self.data)}) <= seq_length({self.seq_length})，无法生成有效样本"
            )

    def __len__(self) -> int:
        """返回滑动窗口可生成的样本总数"""
        return len(self.data) - self.seq_length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        返回第 idx 个样本。

        Returns
        -------
        X : torch.Tensor  shape (seq_length, 1)
        y : torch.Tensor  shape (1,)
        """
        X = self.data[idx : idx + self.seq_length].reshape(self.seq_length, 1)
        y = self.data[idx + self.seq_length]
        return torch.from_numpy(X), torch.tensor(y, dtype=torch.float32)


def get_dataloaders(
    csv_path: str,
    seq_length: int = 14,
    batch_size: int = 16,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    从 CSV 创建训练/验证/测试 DataLoader。

    切分比例：训练集 70%，验证集 15%，测试集 15%。
    严格按时间顺序切分，shuffle=False。

    Parameters
    ----------
    csv_path   : str   processed_daily_sales.csv 路径
    seq_length : int   滑动窗口长度
    batch_size : int   DataLoader batch 大小

    Returns
    -------
    (train_loader, val_loader, test_loader)
    """
    df = pd.read_csv(csv_path)
    data = df["normalized_amount"].values.astype(np.float32)

    total_samples = len(data) - seq_length
    train_end = int(total_samples * 0.70)
    val_end = int(total_samples * 0.85)

    logger.info(f"总样本数: {total_samples}, 训练: {train_end}, 验证: {val_end - train_end}, 测试: {total_samples - val_end}")

    def _make_tensor(data_arr: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        X = torch.from_numpy(data_arr[:, :-1].reshape(-1, seq_length, 1)).float()
        y = torch.from_numpy(data_arr[:, -1:]).float()
        return X, y

    def _slice_and_loader(start: int, end: int) -> DataLoader:
        """从 start 到 end 切分数组，转为 DataLoader"""
        sliced = data[start : start + seq_length + (end - start)]
        # 构建滑动窗口: X -> (samples, seq_length, 1), y -> (samples, 1)
        X_list, y_list = [], []
        for i in range(len(sliced) - seq_length):
            X = torch.from_numpy(sliced[i : i + seq_length].copy()).float().unsqueeze(-1)  # (seq, 1)
            y_list.append(sliced[i + seq_length])
            X_list.append(X)
        X = torch.stack(X_list).float()   # (samples, seq_length, 1)
        y = torch.tensor(y_list, dtype=torch.float32).unsqueeze(1)  # (samples, 1)
        dataset = TensorDataset(X, y)
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)

    train_loader = _slice_and_loader(0, train_end)
    val_loader = _slice_and_loader(train_end, val_end)
    test_loader = _slice_and_loader(val_end, total_samples)

    # 打印 shape 校验
    for name, loader in [("训练集", train_loader), ("验证集", val_loader), ("测试集", test_loader)]:
        for batch_X, batch_y in loader:
            logger.info(f"{name}: X shape={batch_X.shape}, y shape={batch_y.shape}")
            break

    return train_loader, val_loader, test_loader
