import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class BinaryUpliftEvaluator_RCT:
    def __init__(self, y_true, t_true, uplift_pred, p_scores=None):
        """
        [RCT 专用版] 
        适用于随机对照试验，无需倾向性得分 p_scores。
        自动适应非 1:1 分流。

        Args:
            y_true: 真实结果
            t_true: 真实干预 (0/1)
            uplift_pred: 模型预测的 CATE 分数 (越大代表越敏感)
        """
        self.y = np.array(y_true)
        self.t = np.array(t_true)
        self.pred = np.array(uplift_pred)
        
        # 计算全局权重 (用于替代 1/p_score)
        # w_t = N / N_t
        # w_c = N / N_c
        N = len(self.y)
        N_t = np.sum(self.t == 1)
        N_c = np.sum(self.t == 0)
        
        if N_t == 0 or N_c == 0:
            raise ValueError("RCT 数据中必须同时包含 Treatment 组和 Control 组样本")
            
        self.w_t = N / N_t
        self.w_c = N / N_c

    def get_uplift_metrics(self, bins=10):
        """
        计算十分位统计量 (Decile Metrics) - RCT 版
        """
        df = pd.DataFrame({
            'y': self.y,
            't': self.t,
            'pred': self.pred
        })
        
        # 1. 按预测分排序并分桶 (High Uplift -> Low Uplift)
        df['bucket'] = pd.qcut(df['pred'], bins, labels=False, duplicates='drop')
        df['bucket'] = bins - 1 - df['bucket']
        
        metrics_save = {}
        metrics = []
        for i in range(bins):
            sub = df[df['bucket'] == i]
            n_t = sub[sub['t'] == 1].shape[0]
            n_c = sub[sub['t'] == 0].shape[0]
            
            # 使用 RCT 直接均值差计算 ATE
            # ATE = Mean(Y|T=1) - Mean(Y|T=0)
            if n_t > 0 and n_c > 0:
                mean_y_t = sub[sub['t'] == 1]['y'].mean()
                mean_y_c = sub[sub['t'] == 0]['y'].mean()
                real_uplift = mean_y_t - mean_y_c
            else:
                real_uplift = 0
            
            metrics.append({
                'decile': i + 1,
                'real_uplift': real_uplift,
                'n_treatment': n_t,
                'n_control': n_c,
                'pred_mean': sub['pred'].mean()
            })
            metrics_save[f'decile_{i+1}_real_uplift'] = real_uplift
            
        return pd.DataFrame(metrics), metrics_save

    def plot_decile_chart(self, ax=None):
        """绘制十分位图"""
        metrics, metrics_save = self.get_uplift_metrics(bins=10)
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
            
        sns.barplot(data=metrics, x='decile', y='real_uplift', color='royalblue', ax=ax)
        ax.set_title("Uplift by Decile (RCT Based)")
        ax.set_xlabel("Decile (1=Top, 10=Bottom)")
        ax.set_ylabel("Estimated Real ATE (Mean Diff)")
        ax.axhline(0, color='red', linestyle='--', linewidth=1)
        return ax

    def plot_uplift_curve(self, ax=None, name='Model Uplift'):
        """绘制累积 Uplift 曲线 (RCT 版)"""
        df = pd.DataFrame({'y': self.y, 't': self.t, 'pred': self.pred})
        df = df.sort_values('pred', ascending=False).reset_index(drop=True)
        
        # 计算单样本贡献 (基于全局权重)
        # Treatment Contribution: Y * w_t
        # Control Contribution:   Y * w_c
        score_t = df['y'] * df['t'] * self.w_t
        score_c = df['y'] * (1 - df['t']) * self.w_c
        
        # 累积 Uplift
        cum_uplift = np.cumsum(score_t - score_c)
        n_samples = len(df)
        x_axis = np.arange(1, n_samples + 1) / n_samples
        
        # 随机线
        total_gain = cum_uplift.iloc[-1]
        random_line = total_gain * x_axis # 线性增长
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
            
        ax.plot(x_axis, cum_uplift, label=name, linewidth=2)
        ax.plot(x_axis, random_line, 'r--') 
        ax.set_title("Cumulative Uplift Curve (RCT)")
        ax.set_xlabel("Proportion of Population Targeted")
        ax.set_ylabel("Cumulative Value")
        ax.legend()
        return ax
    
class BinaryUpliftEvaluator:
    def __init__(self, y_true, t_true, uplift_pred, p_scores=None):
        """
        Args:
            y_true: 真实结果
            t_true: 真实干预 (0/1)
            uplift_pred: 模型预测的 CATE 分数 (越大代表越敏感)
            p_scores: 倾向性得分 P(T=1|X). 
                      如果是 RCT 数据，可不传(默认为 mean(T))；
                      如果是观测数据，**必须传**，否则评估有偏。
        """
        self.y = np.array(y_true)
        self.t = np.array(t_true)
        self.pred = np.array(uplift_pred)
        
        # 处理倾向性得分
        if p_scores is None:
            # mean_t = np.mean(self.t)
            # self.p = np.full_like(self.t, mean_t, dtype=float)
            self.p = np.full_like(self.t, 0.5, dtype=float)
        else:
            self.p = np.array(p_scores)
            # Clipping 防止除零
            self.p = np.clip(self.p, 0.01, 0.99)

    def get_uplift_metrics(self, bins=10):
        """
        计算十分位统计量 (Decile Metrics)
        """
        df = pd.DataFrame({
            'y': self.y,
            't': self.t,
            'pred': self.pred,
            'p': self.p
        })
        
        # 1. 按预测分排序并分桶 (High Uplift -> Low Uplift)
        df['bucket'] = pd.qcut(df['pred'], bins, labels=False, duplicates='drop')
        # 反转 bucket，使得 9 是最高分，0 是最低分
        df['bucket'] = bins - 1 - df['bucket']
        
        metrics = []
        for i in range(bins):
            sub = df[df['bucket'] == i]
            n_t = sub[sub['t'] == 1].shape[0]
            n_c = sub[sub['t'] == 0].shape[0]
            
            # 使用 IPW 估计桶内真实的 ATE
            # ATE = Mean(Y*T/e) - Mean(Y*(1-T)/(1-e))
            if len(sub) > 0:
                weighted_y_t = np.sum(sub['y'] * sub['t'] / sub['p'])
                weighted_y_c = np.sum(sub['y'] * (1 - sub['t']) / (1 - sub['p']))
                
                # 归一化权重和 (Hajek Estimator 更稳健)
                weight_sum_t = np.sum(sub['t'] / sub['p'])
                weight_sum_c = np.sum((1 - sub['t']) / (1 - sub['p']))
                
                real_uplift = (weighted_y_t / (weight_sum_t + 1e-9)) - \
                              (weighted_y_c / (weight_sum_c + 1e-9))
            else:
                real_uplift = 0
            
            metrics.append({
                'decile': i + 1,
                'real_uplift': real_uplift,
                'n_treatment': n_t,
                'n_control': n_c,
                'pred_mean': sub['pred'].mean()
            })
            
        return pd.DataFrame(metrics)

    def plot_decile_chart(self, ax=None):
        """绘制十分位图 (业务最看重)"""
        metrics = self.get_uplift_metrics(bins=10)
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
            
        sns.barplot(data=metrics, x='decile', y='real_uplift', color='royalblue', ax=ax)
        ax.set_title("Uplift by Decile (Top 10% = Highest Predicted Gain)")
        ax.set_xlabel("Decile (1=Top, 10=Bottom)")
        ax.set_ylabel("Estimated Real ATE (IPW Adjusted)")
        ax.axhline(0, color='red', linestyle='--', linewidth=1)
        return ax

    def plot_uplift_curve(self, ax = None, name = 'Model Uplift'):
        """绘制累积 Uplift 曲线 (AUUC)"""
        df = pd.DataFrame({'y': self.y, 't': self.t, 'p': self.p, 'pred': self.pred})
        df = df.sort_values('pred', ascending=False).reset_index(drop=True)
        
        # 计算累积增益
        # Gain = cumsum( (Y/P)*T - (Y/(1-P))*(1-T) )
        score_t = (df['y'] * df['t']) / df['p']
        score_c = (df['y'] * (1 - df['t'])) / (1 - df['p'])
        
        # 为了曲线平滑，通常计算 'Cumulative Effect'
        # 这里使用简单版: 累积 Uplift * 人数比例
        cum_uplift = np.cumsum(score_t - score_c)
        n_samples = len(df)
        x_axis = np.arange(1, n_samples + 1) / n_samples
        
        # 随机线 (Random Baseline)
        overall_ate = np.mean(score_t - score_c)
        random_line = overall_ate * x_axis * n_samples # 线性增长
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
            
        ax.plot(x_axis, cum_uplift, label = name , linewidth=2)
        ax.plot(x_axis, random_line, 'r--') # , label='Random Guess'
        ax.set_title("Cumulative Uplift Curve")
        ax.set_xlabel("Proportion of Population Targeted")
        ax.set_ylabel("Cumulative Value (Adjusted)")
        ax.legend()
        return ax

class MultiUpliftEvaluator:
    def __init__(self, y_true, t_true, cate_preds, p_scores_matrix):
        """
        Args:
            cate_preds: (N, K) 矩阵. 第 k 列代表 Treatment k vs Control 的 CATE.
                        注意: 列索引应对应 treatment_id (1, 2, 3...)
            p_scores_matrix: (N, K+1) 矩阵. [P(T=0), P(T=1), P(T=2)...]
        """
        self.y = np.array(y_true)
        self.t = np.array(t_true)
        self.preds = np.array(cate_preds) # Shape (N, K)
        self.p_mat = np.array(p_scores_matrix)
        
        # 获取 Treatment ID 列表 (假设 preds 的列顺序对应 T=1, T=2...)
        self.treatment_ids = np.arange(1, self.preds.shape[1] + 1)

    def evaluate_policy_value(self):
        """
        计算: 如果按照模型推荐去干预，预期的 Y 是多少?
        方法: IPW 策略评估 (Offline Policy Evaluation)
        """
        # 1. 模型推荐策略 pi(x)
        # 比较每个 Treatment 的 CATE，如果都 < 0 (或阈值)，则选 T=0
        # 这里的逻辑是: 选 CATE 最大的那个，如果最大值 <= 0，则不干预
        best_uplift_idx = np.argmax(self.preds, axis=1) # 0->T1, 1->T2...
        max_uplift = np.max(self.preds, axis=1)
        
        # 映射回真实的 T ID
        recommended_t = self.treatment_ids[best_uplift_idx]
        # 如果最大增益 <= 0, 推荐 T=0 (Control)
        recommended_t[max_uplift <= 0] = 0
        
        # 2. 计算 Policy Value (IPW)
        # 只取 "真实干预 == 推荐干预" 的样本进行加权统计
        value_sum = 0.0
        weight_sum = 0.0
        
        n = len(self.y)
        for i in range(n):
            rec = recommended_t[i]
            obs = self.t[i]
            
            if rec == obs:
                # 获取该样本观测 T 的倾向分
                # p_mat 列索引 0->T0, 1->T1...
                propensity = max(self.p_mat[i, obs], 0.01)
                weight = 1.0 / propensity
                
                value_sum += self.y[i] * weight
                weight_sum += weight
                
        policy_value = value_sum / weight_sum if weight_sum > 0 else 0
        
        # 计算 Baseline Value (全随机/平均分配) 用于对比
        # 这里简单返回 IPW 调整后的 Policy Value
        return policy_value, recommended_t

    def plot_policy_distribution(self, recommended_t, ax=None):
        """画图: 模型推荐各个策略的比例"""
        counts = pd.Series(recommended_t).value_counts().sort_index()
        
        if ax is None:
            fig, ax = plt.subplots()
        
        counts.plot(kind='bar', ax=ax, color='teal')
        ax.set_title("Recommended Policy Distribution")
        ax.set_xlabel("Treatment ID (0=Control)")
        ax.set_ylabel("Count of Users")
        return ax

class ContinuousUpliftEvaluator:
    def __init__(self, y_true, t_true, cate_pred):
        """
        cate_pred: 边际效应 (Marginal Effect), 即 dY/dT 的预测值
        """
        self.y = np.array(y_true)
        self.t = np.array(t_true)
        self.cate = np.array(cate_pred)

    def plot_dose_response(self, n_bins=5, ax=None):
        """
        可视化: 敏感人群 vs 不敏感人群 的真实 Y-T 关系
        """
        # 1. 根据预测的敏感度 (CATE) 将人群分为两组
        # Top 30% (High Sensitivity) vs Bottom 30% (Low Sensitivity)
        threshold_high = np.percentile(self.cate, 70)
        threshold_low = np.percentile(self.cate, 30)
        
        mask_high = self.cate >= threshold_high
        mask_low = self.cate <= threshold_low
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
            
        # 2. 绘制 High Group 的真实散点/拟合线
        self._plot_smooth_line(self.t[mask_high], self.y[mask_high], 
                             ax, color='red', label='High Sensitivity Group')
        
        # 3. 绘制 Low Group 的真实散点/拟合线
        self._plot_smooth_line(self.t[mask_low], self.y[mask_low], 
                             ax, color='blue', label='Low Sensitivity Group')
        
        ax.set_title("Dose-Response Check: Real Y vs T by Predicted Group")
        ax.set_xlabel("Treatment Value (e.g. Price/Discount)")
        ax.set_ylabel("Outcome Y")
        ax.legend()
        return ax

    def _plot_smooth_line(self, t, y, ax, color, label):
        """辅助函数: 对 T 进行分箱并计算 Y 的均值，画折线图"""
        # 简单分箱平滑
        df = pd.DataFrame({'t': t, 'y': y})
        # 将 T 分为 10 个桶
        df['t_bin'] = pd.cut(df['t'], bins=10)
        # 计算每个桶的 T 均值和 Y 均值
        agg = df.groupby('t_bin', observed=True).agg({'t': 'mean', 'y': 'mean'}).dropna()
        
        ax.plot(agg['t'], agg['y'], color=color, marker='o', linewidth=2, label=label)
        # 可选: 加上半透明散点
        # ax.scatter(t, y, color=color, alpha=0.1, s=10)

def run_evaluation_demo():
    print("🚀 启动全栈评估模块 Demo...")
    np.random.seed(42)
    n = 3000

    # -------------------------------------------------
    # 1. Binary Evaluation Demo
    # -------------------------------------------------
    print("\n[Demo 1] 二元策略评估")
    # 模拟数据
    y_bin = np.random.normal(0, 1, n)
    t_bin = np.random.binomial(1, 0.5, n)
    pred_bin = np.random.normal(0, 1, n) + t_bin * 0.5 # 只有一点点预测能力
    # 观测数据需要 P Score
    p_bin = np.full(n, 0.5) 
    
    eval_bin = BinaryUpliftEvaluator(y_bin, t_bin, pred_bin, p_bin)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    eval_bin.plot_decile_chart(ax=axes[0])
    eval_bin.plot_uplift_curve(ax=axes[1])
    plt.suptitle("Binary Uplift Evaluation")
    plt.tight_layout()
    plt.show()

    # -------------------------------------------------
    # 2. Multi-Treatment Evaluation Demo
    # -------------------------------------------------
    print("\n[Demo 2] 多元策略评估")
    # 模拟 T in {0, 1, 2}
    t_multi = np.random.choice([0, 1, 2], size=n)
    y_multi = np.random.normal(0, 1, n)
    # 预测矩阵 (N, 2): T1 CATE, T2 CATE
    pred_multi = np.random.normal(0, 0.5, size=(n, 2))
    # Propensity Matrix (N, 3): P(0), P(1), P(2)
    p_mat = np.full((n, 3), 0.33)
    
    eval_multi = MultiUpliftEvaluator(y_multi, t_multi, pred_multi, p_mat)
    val, rec_t = eval_multi.evaluate_policy_value()
    print(f"   Expected Policy Value: {val:.4f}")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    eval_multi.plot_policy_distribution(rec_t, ax=ax)
    plt.title("Multi-Treatment Policy Recommendation")
    plt.show()

    # -------------------------------------------------
    # 3. Continuous Evaluation Demo
    # -------------------------------------------------
    print("\n[Demo 3] 连续策略评估 (Dose-Response)")
    t_cont = np.random.uniform(0, 10, n)
    # 假设 High Group 斜率为 2, Low Group 斜率为 0.5
    cate_true = np.random.choice([0.5, 2.0], size=n) # 真实敏感度
    y_cont = 2 + cate_true * t_cont + np.random.normal(0, 2, n)
    # 我们的预测稍微带点噪声
    pred_cont = cate_true + np.random.normal(0, 0.5, n)
    
    eval_cont = ContinuousUpliftEvaluator(y_cont, t_cont, pred_cont)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    eval_cont.plot_dose_response(ax=ax)
    plt.show()
