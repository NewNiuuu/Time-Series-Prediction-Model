#!/usr/bin/env python3
"""
推理与增量训练脚本 - 阶段4
职责：
1. predict_next_day()      : 加载已有模型预测下一天营业额
2. incremental_train()     : 利用新数据对模型进行增量微调，防遗忘

典型业务场景：
- 每日营业结束后，调用 predict_next_day() 预测明日营业额辅助备货
- 积累数日后新数据，调用 incremental_train() 微调模型持续迭代
"""

import logging
import os
import pickle
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from model import SalesLSTM

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ========== 全局路径配置 ==========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # scripts/
PROJECT_DIR = os.path.dirname(BASE_DIR)                 # 项目根目录
DATA_CSV = os.path.join(PROJECT_DIR, "data", "processed_daily_sales.csv")
SCALER_PATH = os.path.join(PROJECT_DIR, "models", "scaler.pkl")
MODEL_PATH = os.path.join(PROJECT_DIR, "models", "lstm_baseline.pth")
FINETUNED_MODEL_PATH = os.path.join(PROJECT_DIR, "models", "lstm_finetuned.pth")

# ========== 模型超参数（需与 model.py / train.py 保持一致）==========
SEQ_LENGTH = 14
HIDDEN_SIZE = 32
NUM_LAYERS = 2
INPUT_SIZE = 1
OUTPUT_SIZE = 1


def get_device() -> torch.device:
    """GPU/CPU 自适应设备选择"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"推理设备: {device}")
    return device


def load_scaler(scaler_path: str):
    """加载 sklearn MinMaxScaler"""
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    logger.info(f"Scaler 已加载: {scaler_path}")
    return scaler


def load_model(model_path: str, device: torch.device) -> SalesLSTM:
    """加载训练好的 SalesLSTM 模型权重"""
    model = SalesLSTM(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        output_size=OUTPUT_SIZE,
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    logger.info(f"模型已加载: {model_path}")
    return model


def predict_next_day(model_path: str = MODEL_PATH, scaler_path: str = SCALER_PATH) -> float:
    """
    读取最新历史数据（最后14天），预测下一天营业额。

    Parameters
    ----------
    model_path  : str  模型权重路径
    scaler_path : str  归一化器路径

    Returns
    -------
    float  预测的下一天营业额（元）
    """
    logger.info("=" * 50)
    logger.info("开始执行: predict_next_day()")
    logger.info("=" * 50)

    device = get_device()

    # 1. 加载模型和 scaler
    model = load_model(model_path, device)
    scaler = load_scaler(scaler_path)

    # 2. 读取 CSV，取最后14天归一化数据
    df = pd.read_csv(DATA_CSV)
    if len(df) < SEQ_LENGTH:
        raise ValueError(f"CSV 数据不足 {SEQ_LENGTH} 天，无法进行预测。当前仅 {len(df)} 天。")

    last_14_days_norm = df["normalized_amount"].values[-SEQ_LENGTH:].astype(np.float32)
    logger.info(f"最后14天归一化值: {last_14_days_norm.round(4).tolist()}")

    # 3. 转为 (1, seq_length, 1) Tensor
    X = torch.from_numpy(last_14_days_norm.reshape(1, SEQ_LENGTH, 1)).float().to(device)

    # 4. 推理（单步预测）
    with torch.no_grad():
        pred_norm = model(X).item()  # scalar

    # 5. 反归一化还原为"元"
    pred_yuan = scaler.inverse_transform([[pred_norm]])[0][0]

    logger.info(f"最后14天原始营业额范围: "
                f"{scaler.inverse_transform(last_14_days_norm.reshape(-1,1)).flatten().min():.2f} ~ "
                f"{scaler.inverse_transform(last_14_days_norm.reshape(-1,1)).flatten().max():.2f} 元")

    print("\n" + "=" * 55)
    print(f"  基于过去 14 天的数据，模型预测明天的营业额为：{pred_yuan:.2f} 元")
    print("=" * 55)

    return pred_yuan


def _build_dataloader_from_csv(
    csv_path: str,
    seq_length: int,
    batch_size: int = 16,
) -> DataLoader:
    """
    给定一个 CSV 文件（含 date, normalized_amount 列），
    构建 PyTorch DataLoader 用于增量训练。
    """
    df = pd.read_csv(csv_path)
    data = df["normalized_amount"].values.astype(np.float32)

    if len(data) <= seq_length:
        raise ValueError(f"CSV 数据不足 {seq_length} 天，无法构建滑动窗口。当前 {len(data)} 天。")

    samples = len(data) - seq_length
    logger.info(f"增量数据: 共 {len(data)} 天 → 可生成 {samples} 个训练样本")

    X_list, y_list = [], []
    for i in range(samples):
        X_list.append(data[i : i + seq_length])
        y_list.append(data[i + seq_length])

    X = torch.tensor(np.array(X_list), dtype=torch.float32).unsqueeze(-1)   # (N, seq, 1)
    y = torch.tensor(np.array(y_list), dtype=torch.float32).unsqueeze(1)   # (N, 1)
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def incremental_train(
    new_data_csv_path: str,
    base_model_path: str = MODEL_PATH,
    output_model_path: str = FINETUNED_MODEL_PATH,
    learning_rate: float = 1e-4,
    epochs: int = 5,
    batch_size: int = 16,
) -> None:
    """
    利用新收集的数据对已有模型进行增量微调（Incremental Learning）。

    防遗忘核心策略：
    1. 使用极小的学习率 (lr=1e-4)，确保参数每次更新幅度极小
    2. 仅训练少量 Epoch（5-10），避免过度拟合新数据
    3. 保存为新文件，不覆盖原始 baseline

    Parameters
    ----------
    new_data_csv_path : str   新数据的 CSV 路径（含 date, normalized_amount 列）
    base_model_path    : str   原始模型权重路径
    output_model_path  : str   微调后模型保存路径
    learning_rate      : float 学习率，默认 1e-4（防灾难性遗忘关键）
    epochs             : int   增量训练轮数，默认 5
    batch_size         : int   batch 大小，默认 16
    """
    logger.info("=" * 50)
    logger.info("开始执行: incremental_train()")
    logger.info(f"新数据路径   : {new_data_csv_path}")
    logger.info(f"基础模型路径 : {base_model_path}")
    logger.info(f"输出路径     : {output_model_path}")
    logger.info(f"学习率       : {learning_rate}（防灾难性遗忘关键）")
    logger.info(f"Epochs       : {epochs}")
    logger.info("=" * 50)

    device = get_device()

    # 1. 加载已有模型
    model = SalesLSTM(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        output_size=OUTPUT_SIZE,
    ).to(device)
    model.load_state_dict(torch.load(base_model_path, map_location=device, weights_only=True))
    logger.info("原始模型权重加载完成")

    # 2. 构建新数据的 DataLoader
    loader = _build_dataloader_from_csv(new_data_csv_path, SEQ_LENGTH, batch_size)

    # 3. 配置训练（极小学习率 + 少量 Epoch）
    criterion = nn.MSELoss()
    # 【核心防遗忘】使用极小的学习率，参数每次更新幅度被严格限制
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    logger.info(f"优化器: Adam(lr={learning_rate})")

    # 4. 增量训练循环
    model.train()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for batch_X, batch_y in loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            preds = model(batch_X)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_X.size(0)

        avg_loss = epoch_loss / len(loader.dataset)
        logger.info(f"增量 Epoch [{epoch:2d}/{epochs}] - Loss: {avg_loss:.6f}")

    # 5. 保存微调后的模型
    os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
    torch.save(model.state_dict(), output_model_path)
    logger.info(f"增量训练完成，模型已保存至: {output_model_path}")


if __name__ == "__main__":
    # ========== 步骤1: 推理预测明日营业额 ==========
    predict_next_day()

    # ========== 步骤2: 模拟增量训练 ==========
    # 将现有 CSV 的最后20天切出来作为"新数据"，模拟现实中新数据到达后的增量训练
    print("\n" + "-" * 50)
    print("开始模拟增量训练...")
    print("-" * 50)

    df_full = pd.read_csv(DATA_CSV)
    if len(df_full) < 20:
        print(f"警告: CSV 仅 {len(df_full)} 天数据，不足20天，跳过增量训练模拟")
    else:
        # 最后20天作为新数据
        split_idx = len(df_full) - 20
        df_new = df_full.iloc[split_idx:].reset_index(drop=True)
        new_data_path = os.path.join(PROJECT_DIR, "data", "simulated_new_data.csv")
        df_new.to_csv(new_data_path, index=False)
        logger.info(f"已将最后 20 天数据保存至: {new_data_path}，作为增量训练新数据")

        # 运行增量训练（5 epochs，极小学习率）
        incremental_train(
            new_data_csv_path=new_data_path,
            base_model_path=MODEL_PATH,
            output_model_path=FINETUNED_MODEL_PATH,
            learning_rate=1e-4,
            epochs=5,
        )

        print("\n" + "=" * 55)
        print("  增量训练完成！使用新数据微调后的模型已保存。")
        print(f"  新模型路径: {FINETUNED_MODEL_PATH}")
        print("=" * 55)

        # 用微调后的模型再次预测，对比结果
        print("\n--- 微调后模型预测结果 ---")
        predict_next_day(model_path=FINETUNED_MODEL_PATH)
