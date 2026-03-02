# -*- coding: utf-8 -*-
"""
Advanced Data Preprocessor for Causal Discovery
优化数据预处理，避免重复操作
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional

def advanced_data_preprocessor(
    df: pd.DataFrame, 
    outlier_detection_cols: list[str], 
    standardization_cols: list[str],
    outlier_method: str = 'iqr',
    outlier_threshold: float = 3.0,
    missing_values: list = [-99, -9990]
) -> Tuple[pd.DataFrame, dict[str, list]]:
    """
    高级数据预处理函数

    参数:
        df: 原始数据框
        outlier_detection_cols: 需要进行异常值检测的列名列表
        standardization_cols: 需要进行标准化的列名列表
        outlier_method: 异常值检测方法 ('iqr', 'zscore', 'mad')
        outlier_threshold: 异常值阈值
        missing_values: 需要视为缺失值的特殊数值

    返回:
        processed_df: 处理后的数据框（仅正常值被标准化）
        normal_indices: 每个变量的正常值索引字典 {变量名: [索引列表]}
    """

    processed_df = df.copy()
    normal_indices = {}

    print(f"开始预处理 {len(df.columns)} 个变量...")

    for col in df.columns:
        print(f"处理变量: {col}")

        # 步骤1: 识别非缺失值索引
        non_missing_mask = ~df[col].isna() & ~df[col].isin(missing_values)
        non_missing_indices = df.index[non_missing_mask].tolist()

        if len(non_missing_indices) == 0:
            # 如果该变量没有非缺失值
            normal_indices[col] = []
            print(f"  {col}: 无非缺失值")
            continue

        # 步骤2: 异常值检测（仅对指定列）
        if col in outlier_detection_cols:
            col_data = df[col][non_missing_mask]

            if (sum(col_data == 0) / len(col_data) > 0.5):
                print("The variable {} have high rate of zero values: {}.".format(col, np.round(sum(col_data == 0) / len(col_data),4)))
                outlier_method = 'zscore'
            # 根据方法检测异常值
            if outlier_method == 'iqr':
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 3 * IQR
                upper_bound = Q3 + 3 * IQR

                normal_mask = (col_data >= lower_bound) & (col_data <= upper_bound)

            elif outlier_method == 'zscore':
                mean = col_data.mean()
                std = col_data.std()
                if std != 0:
                    z_scores = np.abs((col_data - mean) / std)
                    normal_mask = z_scores <= outlier_threshold
                else:
                    normal_mask = pd.Series([True] * len(col_data))

            elif outlier_method == 'mad':
                median = col_data.median()
                mad = np.median(np.abs(col_data - median))
                if mad != 0:
                    modified_z_scores = 0.6745 * (col_data - median) / mad
                    normal_mask = np.abs(modified_z_scores) <= outlier_threshold
                else:
                    normal_mask = pd.Series([True] * len(col_data))
            else:
                normal_mask = pd.Series([True] * len(col_data))

            # 获取正常值索引（非缺失值 ∩ 非异常值）
            normal_values_indices = [non_missing_indices[i] for i, is_normal in enumerate(normal_mask) if is_normal]

        else:
            # 不需要异常值检测的列，所有非缺失值都视为正常值
            normal_values_indices = non_missing_indices

        # 记录正常值索引
        normal_indices[col] = normal_values_indices

        # 步骤3: 标准化处理（仅对指定列的正常值）
        if col in standardization_cols and len(normal_values_indices) > 0:
            normal_data = processed_df.loc[normal_values_indices, col]

            if normal_data.std() != 0:  # 防止除以零
                mean = normal_data.mean()
                std = normal_data.std()
                processed_df.loc[normal_values_indices, col] = (normal_data - mean) / std
            else:
                # 如果标准差为0，数据没有变化，保持原值
                pass

        print(f"  {col}: 正常值数量 {len(normal_values_indices)} / {len(non_missing_indices)}")

    print(f"预处理完成！")
    return processed_df, normal_indices


def get_cleaned_data_fast(
    variables: List[str], 
    normal_indices: Dict[str, List], 
    processed_df: pd.DataFrame, 
    min_samples: int = 10
) -> pd.DataFrame:
    """
    快速获取清理后的数据子集

    参数:
        variables: 需要获取的变量列表
        normal_indices: 正常值索引字典
        processed_df: 预处理后的数据框
        min_samples: 最小样本数要求

    返回:
        清理后的数据子集，如果样本数不足则返回空DataFrame
    """
    if not variables:
        return pd.DataFrame()

    # 获取所有变量的正常值索引交集
    common_indices = set(normal_indices[variables[0]])

    for var in variables[1:]:
        if var not in normal_indices:
            return pd.DataFrame()  # 变量不存在

        common_indices &= set(normal_indices[var])

        # 早期终止：如果交集已经太小，直接返回空
        if len(common_indices) < min_samples:
            return pd.DataFrame()

    common_indices = sorted(list(common_indices))

    if len(common_indices) < min_samples:
        return pd.DataFrame()

    # 返回预处理后的数据子集
    return processed_df.loc[common_indices, variables].reset_index(drop=True).copy()


def validate_preprocessing_results(
    original_df: pd.DataFrame,
    processed_df: pd.DataFrame, 
    normal_indices: Dict[str, List],
    sample_vars: List[str] = None
) -> None:
    """
    验证预处理结果的正确性

    参数:
        original_df: 原始数据框
        processed_df: 处理后的数据框
        normal_indices: 正常值索引字典
        sample_vars: 抽检验证的变量列表
    """
    if sample_vars is None:
        sample_vars = list(normal_indices.keys())[:3]  # 默认验证前3个变量

    print("验证预处理结果...")

    for var in sample_vars:
        if var not in normal_indices:
            continue

        normal_idx = normal_indices[var]

        if len(normal_idx) == 0:
            print(f"  {var}: 无正常值")
            continue

        # 检查正常值索引是否有效
        valid_idx = [idx for idx in normal_idx if idx in original_df.index]
        print(f"  {var}: 有效正常值索引 {len(valid_idx)} 个")

        # 检查标准化结果
        if len(valid_idx) > 1:
            original_values = original_df.loc[valid_idx, var]
            processed_values = processed_df.loc[valid_idx, var]

            if not np.allclose(original_values, processed_values):
                # 如果数值不同，应该是标准化了
                mean_check = np.abs(processed_values.mean()) < 1e-10
                std_check = np.abs(processed_values.std() - 1.0) < 1e-10
                print(f"  {var}: 标准化检查 - 均值≈0: {mean_check}, 标准差≈1: {std_check}")
            else:
                print(f"  {var}: 数值未变化（可能未标准化）")

    print("验证完成！")


# 测试函数
def test_preprocessor():
    """测试预处理函数"""
    # 创建测试数据
    np.random.seed(42)
    test_df = pd.DataFrame({
        'var1': np.random.normal(100, 15, 1000),
        'var2': np.random.choice(['A', 'B', 'C'], 1000),
        'var3': np.random.exponential(2, 1000),
        'target': np.random.normal(50, 10, 1000)
    })

    # 添加缺失值
    test_df.loc[np.random.choice(test_df.index, 50), 'var1'] = np.nan
    test_df.loc[np.random.choice(test_df.index, 30), 'var3'] = -99
    test_df.loc[np.random.choice(test_df.index, 20), 'target'] = np.nan

    # 添加异常值
    test_df.loc[np.random.choice(test_df.index, 10), 'var1'] = 200
    test_df.loc[np.random.choice(test_df.index, 15), 'var3'] = 50

    print("测试数据创建完成：")
    print(f"数据形状: {test_df.shape}")
    print(f"数据类型:\n{test_df.dtypes}")

    # 定义变量类型
    var_types = {
        'var1': 'continuous',
        'var2': 'discrete', 
        'var3': 'continuous',
        'target': 'continuous'
    }

    # 运行预处理
    processed_df, normal_indices = advanced_data_preprocessor(
        df=test_df,
        outlier_detection_cols=['var1', 'var3'],  # 连续变量需要异常值检测
        standardization_cols=['var1', 'var3'],   # 连续变量需要标准化
        outlier_method='iqr',
        outlier_threshold=3.0
    )

    # 验证结果
    validate_preprocessing_results(test_df, processed_df, normal_indices)

    # 测试快速数据获取
    print("\n测试快速数据获取:")
    cleaned_data = get_cleaned_data_fast(
        variables=['var1', 'var3'],
        normal_indices=normal_indices,
        processed_df=processed_df,
        min_samples=10
    )

    print(f"获取到的清理数据形状: {cleaned_data.shape}")
    if not cleaned_data.empty:
        print(f"数据摘要:\n{cleaned_data.describe()}")

    return processed_df, normal_indices


if __name__ == "__main__":
    # 运行测试
    processed_df, normal_indices = test_preprocessor()