#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据优化分析程序（简化版）
包含字体优化、Spearman相关性分析和特征筛选功能
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import warnings
import os

warnings.filterwarnings('ignore')

class OptimizedDataFeatureSelector:
    def __init__(self, data_path='./data/lenta_data_sampled_30k_per_group.csv', target_col='response_att', n_features=100):
        """初始化特征选择器"""
        print("正在加载特征数据...")
        self.data = pd.read_csv(data_path)
        self.target_col = target_col
        self.n_features = n_features
        print(f"数据加载完成！形状: {self.data.shape}")

        # 分离特征和标签
        self.feature_cols = [col for col in self.data.columns if col not in
                           [self.target_col, 'group', 'age','gender']]

        # 设置matplotlib参数（解决字体问题）
        self.setup_matplotlib()

    def setup_matplotlib(self):
        """设置matplotlib参数以解决字体问题"""
        try:
            plt.rcParams['font.family'] = 'DejaVu Sans'
            plt.rcParams['axes.unicode_minus'] = False
            sns.set_style("whitegrid")
            print("✓ 字体设置成功 - 使用DejaVu Sans")
        except Exception as e:
            print(f"字体设置警告: {e}")
            plt.rcParams.update(plt.rcParamsDefault)
            plt.rcParams['axes.unicode_minus'] = False

    def quick_preprocess(self):
        """快速数据预处理"""
        print("\n" + "="*60)
        print("数据预处理")
        print("="*60)

        self.processed_data = self.data.copy()

        # 简单缺失值处理 - 中位数填充
        print("处理缺失值...")
        for col in self.feature_cols:
            if self.processed_data[col].isnull().sum() > 0:
                self.processed_data[col].fillna(self.processed_data[col].median(), inplace=True)

        # 标准化特征
        print("标准化特征...")
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(self.processed_data[self.feature_cols])
        self.X_scaled = pd.DataFrame(X_scaled, columns=self.feature_cols)
        self.y = self.processed_data[self.target_col]

        print("✓ 数据预处理完成")

    def calculate_spearman_correlations(self):
        """计算Spearman等级相关系数"""
        print("\n" + "="*60)
        print("计算Spearman等级相关系数")
        print("="*60)

        correlations = []
        total_features = len(self.feature_cols)

        # 分批处理，提高效率
        for i, feature in enumerate(self.feature_cols):
            if i % 100 == 0:
                print(f"处理进度: {i+1}/{total_features}")

            # 获取有效数据
            valid_mask = self.processed_data[feature].notna() & self.y.notna()
            if valid_mask.sum() > 10:
                x_vals = self.processed_data.loc[valid_mask, feature].values
                y_vals = self.y[valid_mask].values

                corr, p_val = spearmanr(x_vals, y_vals)
                correlations.append({
                    'feature': feature,
                    'spearman_corr': corr if not np.isnan(corr) else 0,
                    'abs_spearman_corr': abs(corr) if not np.isnan(corr) else 0,
                    'p_value': p_val if not np.isnan(p_val) else 1.0
                })
            else:
                correlations.append({
                    'feature': feature,
                    'spearman_corr': 0,
                    'abs_spearman_corr': 0,
                    'p_value': 1.0
                })

        self.spearman_results = pd.DataFrame(correlations).sort_values('abs_spearman_corr', ascending=False)

        print(f"计算完成！共分析了 {len(self.spearman_results)} 个特征")
        print(f"最强相关特征: {self.spearman_results.iloc[0]['feature']}")
        print(f"Spearman相关系数: {self.spearman_results.iloc[0]['spearman_corr']:.4f}")

        return self.spearman_results

    def lasso_feature_selection(self):
        """LASSO特征选择"""
        print("\n" + "="*60)
        print("LASSO回归特征选择")
        print("="*60)

        # 使用较小的CV以加快速度
        lasso_cv = LassoCV(cv=3, random_state=42, max_iter=200, n_alphas=50)
        lasso_cv.fit(self.X_scaled, self.y)

        # 获取特征重要性
        lasso_importance = np.abs(lasso_cv.coef_)

        self.lasso_results = pd.DataFrame({
            'feature': self.feature_cols,
            'lasso_coef': lasso_cv.coef_,
            'lasso_importance': lasso_importance
        }).sort_values('lasso_importance', ascending=False)

        # 选择非零系数的特征
        self.lasso_results = self.lasso_results[self.lasso_results['lasso_importance'] > 1e-6]

        print(f"LASSO选择完成！")
        print(f"非零系数特征数: {len(self.lasso_results)}")
        print(f"最佳alpha值: {lasso_cv.alpha_:.4f}")
        print(f"R²得分: {lasso_cv.score(self.X_scaled, self.y):.4f}")

        return self.lasso_results

    def lightgbm_feature_importance(self):
        """LightGBM特征重要性"""
        print("\n" + "="*60)
        print("LightGBM特征重要性")
        print("="*60)

        # LightGBM回归器配置
        lgbm = lgb.LGBMRegressor(
            n_estimators=100,      # 提升轮数
            max_depth=6,          # 树的最大深度
            learning_rate=0.1,    # 学习率
            num_leaves=31,        # 叶子节点数
            random_state=42,
            n_jobs=-1,
            verbose=-1           # 关闭训练日志
        )
        lgbm.fit(self.X_scaled, self.y)

        self.lgbm_results = pd.DataFrame({
            'feature': self.feature_cols,
            'lgbm_importance': lgbm.feature_importances_
        }).sort_values('lgbm_importance', ascending=False)

        print(f"LightGBM特征重要性计算完成！")
        print(f"最重要特征: {self.lgbm_results.iloc[0]['feature']}")
        print(f"重要性得分: {self.lgbm_results.iloc[0]['lgbm_importance']:.4f}")

        return self.lgbm_results

    def combine_feature_importance(self):
        """综合多种方法的特征重要性"""
        print("\n" + "="*60)
        print("综合特征重要性计算")
        print("="*60)

        # 权重配置
        weights = {
            'spearman': 0.4,      # Spearman相关系数权重
            'lasso': 0.3,         # LASSO权重
            'lightgbm': 0.3       # LightGBM权重
        }

        # 合并结果
        combined_df = pd.DataFrame({'feature': self.feature_cols})

        # 添加Spearman结果
        combined_df = combined_df.merge(
            self.spearman_results[['feature', 'abs_spearman_corr']].rename(
                columns={'abs_spearman_corr': 'spearman_score'}),
            on='feature', how='left'
        )
        combined_df['spearman_score'] = combined_df['spearman_score'].fillna(0)

        # 添加LASSO结果
        combined_df = combined_df.merge(
            self.lasso_results[['feature', 'lasso_importance']],
            on='feature', how='left'
        )
        combined_df['lasso_importance'] = combined_df['lasso_importance'].fillna(0)

        # 添加LightGBM结果
        combined_df = combined_df.merge(
            self.lgbm_results[['feature', 'lgbm_importance']],
            on='feature', how='left'
        )
        combined_df['lgbm_importance'] = combined_df['lgbm_importance'].fillna(0)

        # 标准化各方法的得分（0-1范围）
        for method in ['spearman_score', 'lasso_importance', 'lgbm_importance']:
            max_score = combined_df[method].max()
            if max_score > 0:
                combined_df[f'{method}_norm'] = combined_df[method] / max_score
            else:
                combined_df[f'{method}_norm'] = 0

        # 计算综合得分
        combined_df['combined_score'] = 0
        for method, weight in weights.items():
            if method == 'spearman':
                norm_col = 'spearman_score_norm'
            elif method == 'lasso':
                norm_col = 'lasso_importance_norm'
            else:  # lightgbm
                norm_col = 'lgbm_importance_norm'
            combined_df['combined_score'] += weight * combined_df[norm_col]

        self.combined_results = combined_df.sort_values('combined_score', ascending=False)

        print(f"综合特征重要性计算完成！")
        print(f"Top 10 特征:")
        print("排名 | 特征名 | 综合得分 | Spearman | LASSO | LightGBM")
        print("-" * 80)
        for i, (_, row) in enumerate(self.combined_results.head(10).iterrows(), 1):
            print(f"{i:4} | {row['feature'][:30]:30} | {row['combined_score']:8.4f} | "
                  f"{row['spearman_score']:8.4f} | {row['lasso_importance_norm']:8.4f} | "
                  f"{row['lgbm_importance_norm']:8.4f}")

        return self.combined_results

    def select_and_save_features(self, output_dir="./data"):
        """选择特征并保存结果"""
        print("\n" + "="*60)
        print(f"选择Top {self.n_features}个特征")
        print("="*60)

        # 选择Top N特征
        self.selected_features = self.combined_results.head(self.n_features)['feature'].tolist()

        # 创建筛选后的数据集
        self.selected_data = pd.DataFrame()
        self.selected_data[self.selected_features] = self.processed_data[self.selected_features]
        self.selected_data[self.target_col] = self.processed_data[self.target_col]

        # 添加重要的人口统计学特征或其他需要保留的特征
        important_demo_cols = ['age', 'gender', 'group']
        for col in important_demo_cols:
            if col in self.processed_data.columns:
                self.selected_data[col] = self.processed_data[col]

        print(f"✓ 已选择 {self.n_features} 个特征")
        print(f"新数据集形状: {self.selected_data.shape}")
        
        # self.selected_data.loc[self.selected_data['ischaemiaDegree'] != 0,self.target_col] += 5
        
        # 保存结果
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        self.selected_data.to_csv(f'{output_dir}/selected_features_data.csv', index=False)
        self.combined_results.to_csv(f'{output_dir}/feature_importance_ranking.csv', index=False)

        print(f"✓ 结果已保存到 '{output_dir}' 目录")

    def create_summary_report(self):
        """生成筛选报告"""
        print("\n" + "="*60)
        print("生成特征筛选报告")
        print("="*60)

        report = f"""
        # 心磁特征筛选报告

        ## 执行概况
        - 原始特征数量: {len(self.feature_cols)}
        - 筛选后特征数量: {len(self.selected_features)}
        - 目标变量: {self.target_col}
        - 样本数量: {len(self.processed_data)}

        ## 筛选方法权重
        - Spearman相关系数: 40%
        - LASSO回归: 30%
        - LightGBM: 30%

        ## Top 20 特征
        """

        top_20 = self.combined_results.head(20)
        for i, (_, row) in enumerate(top_20.iterrows(), 1):
            report += f"{i:2d}. {row['feature'][:50]} (得分: {row['combined_score']:.4f})\n"

        report += f"""
        ## 特征类别分析（Top {self.n_features}）
        """

        # 分析Top特征的类别分布
        top_features = self.combined_results.head(self.n_features)
        categories = []
        for feature in top_features['feature']:
            if 'current' in feature.lower() or 'curl' in feature.lower():
                categories.append('Current Density')
            elif 'freq' in feature.lower() or 'spectral' in feature.lower():
                categories.append('Frequency')
            elif 'time' in feature.lower() or 'wave' in feature.lower():
                categories.append('Time Domain')
            elif 'magnetic' in feature.lower() or 'pole' in feature.lower():
                categories.append('Magnetic Map')
            else:
                categories.append('Other')

        category_counts = pd.Series(categories).value_counts()
        for cat, count in category_counts.items():
            percentage = count / len(categories) * 100
            report += f"- {cat}: {count} 个特征 ({percentage:.1f}%)\n"

        report += f"""
        ## 筛选结果统计
        - 综合得分范围: {self.combined_results['combined_score'].min():.4f} - {self.combined_results['combined_score'].max():.4f}
        - 平均得分: {self.combined_results['combined_score'].mean():.4f}
        - 得分标准差: {self.combined_results['combined_score'].std():.4f}

        ## 建议
        1. 使用筛选后的特征进行建模，可显著提高模型训练效率
        2. 建议对Top 20-50特征进行重点分析
        3. 可考虑结合领域知识进一步筛选特征
        4. 建议在不同模型上验证特征的有效性
        """

        # 保存报告
        with open(f'./data/feature_selection_report.md', 'w', encoding='utf-8') as f:
            f.write(report)

        print("✓ 特征筛选报告已生成")
        return report

    def create_visualizations(self, output_dir="./data"):
        """创建可视化图表"""
        print("\n" + "="*60)
        print("创建可视化图表")
        print("="*60)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 1. 综合得分分布
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.hist(self.combined_results['combined_score'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title('Combined Feature Importance Distribution')
        plt.xlabel('Combined Score')
        plt.ylabel('Frequency')

        # Top 50特征得分
        plt.subplot(1, 2, 2)
        top_50 = self.combined_results.head(50)
        plt.bar(range(len(top_50)), top_50['combined_score'], alpha=0.7, color='red')
        plt.title('Top 50 Feature Importance Scores')
        plt.xlabel('Feature Rank')
        plt.ylabel('Combined Score')

        plt.tight_layout()
        plt.savefig(f'{output_dir}/feature_importance_overview.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 2. Top特征对比
        plt.figure(figsize=(14, 8))

        methods = [
            ('spearman_score_norm', 'Spearman Correlation'),
            ('lasso_importance_norm', 'LASSO'),
            ('lgbm_importance_norm', 'LightGBM'),
            ('combined_score', 'Combined Score')
        ]

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.ravel()

        for i, (col, title) in enumerate(methods):
            top_15 = self.combined_results.nlargest(15, col)

            bars = axes[i].barh(range(len(top_15)), top_15[col], alpha=0.7)
            axes[i].set_yticks(range(len(top_15)))
            axes[i].set_yticklabels([f[:25] for f in top_15['feature']], fontsize=8)
            axes[i].set_xlabel('Normalized Score')
            axes[i].set_title(f'Top 15 Features - {title}')
            axes[i].invert_yaxis()

        plt.tight_layout()
        plt.savefig(f'{output_dir}/top_features_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✓ 可视化图表已保存到 '{output_dir}' 目录")

def main():
    """主函数"""
    # 数据文件路径
    data_path = "./data/lenta_data_sampled_30k_per_group.csv"

    # 创建特征选择器
    selector = OptimizedDataFeatureSelector(data_path, n_features=50)

    # 运行完整的特征筛选流程
    print("开始执行特征筛选流程...")

    # 1. 数据预处理
    selector.quick_preprocess()

    # 2. 计算各种特征重要性
    selector.calculate_spearman_correlations()
    selector.lasso_feature_selection()
    selector.lightgbm_feature_importance()

    # 3. 综合特征重要性
    selector.combine_feature_importance()

    # 4. 选择并保存特征
    selector.select_and_save_features()

    # 5. 创建可视化
    selector.create_visualizations()

    # 6. 生成报告
    report = selector.create_summary_report()

    print("\n" + "="*60)
    print("特征筛选流程完成！")
    print("="*60)
    print("生成的文件:")
    print("  - 筛选后数据: ./data/selected_features_data.csv")
    print("  - 特征重要性排名: ./data/feature_importance_ranking.csv")
    print("  - 特征筛选报告: ./data/feature_selection_report.md")
    print("  - 可视化图表: ./data/")
    print("\n建议下一步:")
    print("  1. 使用筛选后的特征进行建模")
    print("  2. 分析Top特征与目标变量的关系")
    print("  3. 在不同模型上验证特征有效性")
    print("  4. 比较LightGBM与随机森林的特征选择效果")

if __name__ == "__main__":
    main()