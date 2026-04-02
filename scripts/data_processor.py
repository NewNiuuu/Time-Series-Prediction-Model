#!/usr/bin/env python3
"""
时序数据处理模块 - 阶段2（重构版）
职责：
1. 加载并清洗 data/ 下的原始 xlsx 账单（处理乱序表头、异构列名）
2. 按天重采样
3. MinMax 归一化
4. 将处理结果导出为 CSV（date, normalized_amount）
5. 保存 scaler.pkl

滑动窗口切分和 DataLoader 生成已剥离至 dataset.py。
"""

import logging
import os
import pickle
from typing import List

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class TimeSeriesDataProcessor:
    """
    便利店营业额的时序数据处理器。

    职责（仅限数据准备，解耦后不再过问模型侧）：
    1. 鲁棒读取 data/ 目录下的多份 Excel 账单（处理乱序表头、异构列名）
    2. 统一列名，过滤支出，只保留收入
    3. 按天重采样并补齐缺失日期
    4. MinMax 标准化，保存 scaler
    5. 导出为 CSV（date, normalized_amount）
    """

    # 2025 账单列名映射（表头约在第17行）
    COLUMNS_2025 = {
        "交易时间": "date",
        "收支类型": "type",
        "金额": "amount",
    }
    # 2026 账单列名映射（表头约在第18行）
    COLUMNS_2026 = {
        "交易时间": "date",
        "收/支": "type",
        "金额(元)": "amount",
    }

    def __init__(self, data_dir: str = "data", models_dir: str = "models"):
        self.data_dir = data_dir
        self.models_dir = models_dir
        self.scaler: MinMaxScaler | None = None
        self._daily_df: pd.DataFrame | None = None  # 重采样+归一化后的 DataFrame

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def run(self, csv_path: str = "data/processed_daily_sales.csv") -> pd.DataFrame:
        """
        端到端执行：
        加载原始 xlsx → 重采样 → 归一化 → 落盘 CSV → 保存 scaler。

        Parameters
        ----------
        csv_path : str   CSV 输出路径

        Returns
        -------
        pd.DataFrame   含 date, normalized_amount 两列的每日归一化 DataFrame
        """
        # 1. 加载并清洗多文件数据
        df = self._load_and_clean_data(self.data_dir)
        logger.info(f"原始记录数（含支出）: {len(df)} 条")

        # 2. 按天重采样
        daily_df = self._resample_to_daily(df)
        logger.info(f"重采样后天数: {len(daily_df)} 天")

        # 3. 归一化（内部保存 scaler）
        result_df = self._normalize_and_attach(daily_df)
        logger.info(f"归一化完成，shape: {result_df.shape}")

        # 4. 落盘 CSV
        self.export_dataset(result_df, csv_path)

        return result_df

    def export_dataset(self, df: pd.DataFrame, output_path: str = "data/processed_daily_sales.csv") -> None:
        """
        将含 date, normalized_amount 的 DataFrame 导出为 CSV 文件。

        Parameters
        ----------
        df         : pd.DataFrame  含 date, normalized_amount 列
        output_path : str           CSV 文件路径
        """
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        logger.info(f"处理后数据集已导出至: {output_path}")

    def save_scaler(self, path: str | None = None) -> None:
        """
        将 fitted 的 MinMaxScaler 保存为 pickle 文件。
        """
        if self.scaler is None:
            raise RuntimeError("Scaler 未初始化，请先调用 run()")
        if path is None:
            path = os.path.join(self.models_dir, "scaler.pkl")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.scaler, f)
        logger.info(f"Scaler 已保存至: {path}")

    def load_scaler(self, path: str | None = None) -> MinMaxScaler:
        """
        从 pickle 文件加载已保存的 MinMaxScaler。
        """
        if path is None:
            path = os.path.join(self.models_dir, "scaler.pkl")
        with open(path, "rb") as f:
            self.scaler = pickle.load(f)
        logger.info(f"Scaler 已从 {path} 加载")
        return self.scaler

    # ------------------------------------------------------------------
    # 私有方法
    # ------------------------------------------------------------------

    def _load_and_clean_data(self, data_dir: str) -> pd.DataFrame:
        """
        遍历 data_dir，读取所有 Excel 文件：
        - 逐行搜索包含 "交易时间" 的行，将其作为真实表头
        - 统一列名为 date / type / amount
        - 过滤：只保留 type 包含 "收入" 的行
        - 类型转换：amount → float，date → datetime
        - 合并多文件，按时间升序返回
        """
        if not os.path.isdir(data_dir):
            raise FileNotFoundError(f"数据目录不存在: {data_dir}")

        all_dfs: List[pd.DataFrame] = []

        for fname in sorted(os.listdir(data_dir)):
            fpath = os.path.join(data_dir, fname)
            if not fname.endswith((".xlsx", ".xls")):
                continue

            logger.info(f"正在处理文件: {fname}")
            raw = pd.read_excel(fpath, header=None, engine="openpyxl")

            # 找到包含"交易时间"的行索引
            header_idx = None
            for idx, row in raw.iterrows():
                row_str = "".join(str(v) for v in row.values)
                if "交易时间" in row_str:
                    header_idx = idx
                    break

            if header_idx is None:
                logger.warning(f"[{fname}] 未找到包含'交易时间'的表头行，跳过")
                continue

            df_raw = pd.read_excel(fpath, header=header_idx, engine="openpyxl")
            logger.info(f"[{fname}] 检测到表头行索引: {header_idx}, 数据行数: {len(df_raw)}")

            col_map = self._detect_columns(df_raw.columns.tolist(), fname)
            if col_map is None:
                logger.warning(f"[{fname}] 无法识别列名，跳过。列名: {df_raw.columns.tolist()}")
                continue

            df = df_raw[list(col_map.keys())].copy()
            df.rename(columns=col_map, inplace=True)

            # 类型清洗
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df["amount"] = (
                df["amount"]
                .astype(str)
                .str.replace(r"[^\d.\-]", "", regex=True)
                .replace("", np.nan)
                .astype(float)
            )
            df["type"] = df["type"].astype(str)

            before = len(df)
            df.dropna(subset=["date", "amount"], inplace=True)
            after = len(df)
            if before != after:
                logger.warning(f"[{fname}] 因 date/amount 解析失败丢弃 {before - after} 行")

            # 过滤：只保留收入
            df = df[df["type"].str.contains("收入", na=False)].copy()
            logger.info(f"[{fname}] 过滤后（仅收入）剩余: {len(df)} 行")

            all_dfs.append(df[["date", "type", "amount"]])

        if not all_dfs:
            raise ValueError("没有找到任何有效数据文件")

        combined = pd.concat(all_dfs, ignore_index=True)
        combined.sort_values("date", inplace=True)
        combined.reset_index(drop=True, inplace=True)
        logger.info(f"合并后总记录数: {len(combined)}, 时间范围: {combined['date'].min()} ~ {combined['date'].max()}")

        return combined

    def _detect_columns(self, columns: list, fname: str) -> dict | None:
        """
        两阶段列名匹配：先精确匹配 2025 / 2026，再模糊兜底。
        """
        # 精确匹配（2025 文件）
        col_map = {}
        for col in columns:
            s = str(col).strip()
            if s in self.COLUMNS_2025 and s not in col_map:
                col_map[s] = self.COLUMNS_2025[s]
        if len(col_map) == 3:
            logger.info(f"[{fname}] 使用2025列名映射: {col_map}")
            return col_map

        # 精确匹配（2026 文件）
        col_map = {}
        for col in columns:
            s = str(col).strip()
            if s in self.COLUMNS_2026 and s not in col_map:
                col_map[s] = self.COLUMNS_2026[s]
        if len(col_map) == 3:
            logger.info(f"[{fname}] 使用2026列名映射: {col_map}")
            return col_map

        # 模糊匹配（兜底）
        col_map = {}
        for col in columns:
            s = str(col).strip()
            if "交易时间" in s and "date" not in col_map.values():
                col_map[s] = "date"
            elif ("收支类型" in s or "收/支" in s or "收支" in s) and "type" not in col_map.values():
                col_map[s] = "type"
            elif ("金额" in s and ("元" in s or "金额" == s or "金额(元)" == s)) and "amount" not in col_map.values():
                col_map[s] = "amount"
        if len(col_map) == 3:
            logger.info(f"[{fname}] 使用模糊列名映射: {col_map}")
            return col_map

        return None

    def _resample_to_daily(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        按天重采样，将同一天的 amount 累加。
        缺失日期用 0 填充，保证日期连续无空洞。
        返回 DataFrame，index 为 DatetimeIndex，列为 'amount'。
        """
        df = df.set_index("date")
        daily: pd.Series = df["amount"].resample("D").sum()
        daily = daily.fillna(0)

        full_range = pd.date_range(start=daily.index.min(), end=daily.index.max(), freq="D")
        daily = daily.reindex(full_range, fill_value=0)
        daily.index.name = "date"

        logger.info(f"重采样完成，日期范围: {daily.index.min().date()} ~ {daily.index.max().date()}, 共 {len(daily)} 天")
        logger.info(f"日均营业额: {daily.mean():.2f}, 最大: {daily.max():.2f}, 最小: {daily.min():.2f}")

        return daily.to_frame()

    def _normalize_and_attach(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        使用 MinMaxScaler 将每日营业额缩放到 [0, 1]。
        保存 scaler 到 models/scaler.pkl。
        返回含 date, normalized_amount 两列的 DataFrame。
        """
        os.makedirs(self.models_dir, exist_ok=True)

        values = df["amount"].values.reshape(-1, 1)
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        normalized = self.scaler.fit_transform(values).flatten()

        result = pd.DataFrame({
            "date": df.index,
            "normalized_amount": normalized,
        })

        scaler_path = os.path.join(self.models_dir, "scaler.pkl")
        with open(scaler_path, "wb") as f:
            pickle.dump(self.scaler, f)
        logger.info(f"Scaler 已保存至: {scaler_path}")

        return result


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("TimeSeriesDataProcessor - 数据落盘测试")
    print("=" * 60)

    processor = TimeSeriesDataProcessor(data_dir="data", models_dir="models")
    result_df = processor.run(csv_path="data/processed_daily_sales.csv")

    print("\n--- 导出数据集预览 ---")
    print(result_df.head(10))
    print(f"\n总行数: {len(result_df)}")
    print(f"列名: {result_df.columns.tolist()}")

    print("\n" + "=" * 60)
    print("数据落盘完成")
    print("=" * 60)
