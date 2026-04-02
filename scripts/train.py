#!/usr/bin/env python3
"""
训练脚本 - 阶段3
职责：
1. 加载 dataset.py 的 DataLoader
2. 初始化 model.py 的 SalesLSTM
3. 训练循环（100 epoch，GPU/CPU自适应，MSELoss + Adam）
4. 保存模型至 models/lstm_baseline.pth
5. 推理：inverse_transform 反归一化，绘制真实值 vs 预测值折线图
"""

import logging
import os
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from dataset import get_dataloaders
from model import SalesLSTM

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def train_one_epoch(model: nn.Module, loader: DataLoader, criterion, optimizer, device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    for batch_X, batch_y in loader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)
        optimizer.zero_grad()
        preds = model(batch_X)
        loss = criterion(preds, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch_X.size(0)
    return total_loss / len(loader.dataset)


def evaluate(model: nn.Module, loader: DataLoader, criterion, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch_X, batch_y in loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            preds = model(batch_X)
            loss = criterion(preds, batch_y)
            total_loss += loss.item() * batch_X.size(0)
    return total_loss / len(loader.dataset)


def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple:
    """在给定 DataLoader 上推理，返回拼接后的预测值和真实值（均为归一化尺度）"""
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch_X, batch_y in loader:
            batch_X = batch_X.to(device)
            preds = model(batch_X)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(batch_y.numpy())
    return np.concatenate(all_preds, axis=0), np.concatenate(all_targets, axis=0)


def inverse_transform(scaler_path: str, values: np.ndarray) -> np.ndarray:
    """使用保存的 scaler 将归一化值还原为原始金额（元）"""
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    return scaler.inverse_transform(values.reshape(-1, 1)).flatten()


def main():
    # ========== 配置 ==========
    CSV_PATH = "data/processed_daily_sales.csv"
    SCALER_PATH = "models/scaler.pkl"
    MODEL_SAVE_PATH = "models/lstm_baseline.pth"
    CHART_SAVE_PATH = "docs/test_prediction_chart.png"
    SEQ_LENGTH = 14
    BATCH_SIZE = 16
    EPOCHS = 100
    LEARNING_RATE = 0.001

    os.makedirs("models", exist_ok=True)
    os.makedirs("docs", exist_ok=True)

    # ========== 设备 ==========
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")

    # ========== 数据 ==========
    logger.info("加载数据集...")
    train_loader, val_loader, test_loader = get_dataloaders(
        csv_path=CSV_PATH,
        seq_length=SEQ_LENGTH,
        batch_size=BATCH_SIZE,
    )
    logger.info(f"训练集样本数: {len(train_loader.dataset)}")
    logger.info(f"验证集样本数: {len(val_loader.dataset)}")
    logger.info(f"测试集样本数: {len(test_loader.dataset)}")

    # ========== 模型 ==========
    model = SalesLSTM(input_size=1, hidden_size=32, num_layers=2, output_size=1).to(device)
    logger.info(f"模型结构:\n{model}")

    # ========== 训练配置 ==========
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # ========== 训练循环 ==========
    logger.info(f"开始训练 ({EPOCHS} epochs)...")
    best_val_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = evaluate(model, val_loader, criterion, device)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            marker = " *"
        else:
            marker = ""

        if epoch % 10 == 0 or marker:
            logger.info(
                f"Epoch [{epoch:3d}/{EPOCHS}] "
                f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}{marker}"
            )

    logger.info(f"训练完成，最佳验证 Loss: {best_val_loss:.6f}")
    logger.info(f"模型已保存至: {MODEL_SAVE_PATH}")

    # ========== 测试集推理 ==========
    logger.info("测试集推理...")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device, weights_only=True))
    preds_norm, targets_norm = predict(model, test_loader, device)

    # 反归一化还原为金额（元）
    preds_yuan = inverse_transform(SCALER_PATH, preds_norm)
    targets_yuan = inverse_transform(SCALER_PATH, targets_norm)

    logger.info(f"预测值范围: {preds_yuan.min():.2f} ~ {preds_yuan.max():.2f} 元")
    logger.info(f"真实值范围: {targets_yuan.min():.2f} ~ {targets_yuan.max():.2f} 元")

    # ========== 可视化 ==========
    logger.info("绘制测试集预测对比图...")
    plt.figure(figsize=(12, 6))
    x_axis = np.arange(len(preds_yuan))
    plt.plot(x_axis, targets_yuan, label="Ground Truth", color="steelblue", linewidth=1.5)
    plt.plot(x_axis, preds_yuan, label="Prediction", color="tomato", linewidth=1.5, linestyle="--")
    plt.title("Convenience Store Daily Sales - Test Set: Ground Truth vs Prediction", fontsize=13)
    plt.xlabel("Test Sample Index", fontsize=12)
    plt.ylabel("Sales (Yuan)", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(CHART_SAVE_PATH, dpi=150)
    logger.info(f"图表已保存至: {CHART_SAVE_PATH}")

    # MSE/RMSE 评估
    mse = np.mean((preds_yuan - targets_yuan) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(preds_yuan - targets_yuan))
    logger.info(f"测试集 MSE: {mse:.2f}, RMSE: {rmse:.2f}, MAE: {mae:.2f}")

    print("\n" + "=" * 60)
    print("训练与推理完成")
    print("=" * 60)
    print(f"模型权重: {MODEL_SAVE_PATH}")
    print(f"预测图表: {CHART_SAVE_PATH}")
    print(f"测试 RMSE: {rmse:.2f} 元")


if __name__ == "__main__":
    main()
