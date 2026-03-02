import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, auc, mean_absolute_error
from xgboost import XGBRegressor, XGBClassifier
import copy
import os
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr

# ==========================================
# [新增] 第三方库兼容性补丁 & 导入
# ==========================================
import sklearn.utils
try:
    from sklearn.utils import check_matplotlib_support
except ImportError:
    # 修复 scikit-learn 1.6+ 移除了 check_matplotlib_support 导致 sklift 报错的问题
    def check_matplotlib_support(caller_name):
        try:
            import matplotlib
        except ImportError:
            raise ImportError(f"{caller_name} requires matplotlib")
    sklearn.utils.check_matplotlib_support = check_matplotlib_support

# 尝试导入第三方评估库
try:
    from sklift.metrics import uplift_auc_score, qini_auc_score
    SKLIFT_AVAILABLE = True
except ImportError:
    SKLIFT_AVAILABLE = False

try:
    from causalml.metrics import get_cumgain, get_qini
    CAUSALML_AVAILABLE = True
except ImportError:
    CAUSALML_AVAILABLE = False

# 引入已有的 Uplift 库
from models.meta_learners import SLearner, TLearner, XLearner
from models.class_transform import ClassTransformEstimator
from models.tree import UniversalUpliftForest

from ocu_framework import FirstStageDeconfounder, SecondStageUpliftTrainer
from evaluation import BinaryUpliftEvaluator, BinaryUpliftEvaluator_RCT

from lgb_models import lgb_optuna

import re
from sklearn.preprocessing import LabelEncoder
from advanced_data_preprocessor import advanced_data_preprocessor, get_cleaned_data_fast


# ==========================================
# 1. 辅助功能模块
# ==========================================

class UpliftMetrics:
    @staticmethod
    def get_auuc_qini(y_true, t_true, uplift_score, p_scores=None):
        """
        [修复版] 计算标量指标: AUUC 和 Qini Coefficient
        支持 Observational Data (通过 p_scores 进行 IPW 调整)
        """
        y_true, t_true, uplift_score = np.array(y_true), np.array(t_true), np.array(uplift_score)
        
        if p_scores is None:
            # 如果不传 p_scores，默认假设是 RCT (随机实验)，均值填充
            p_scores = np.full_like(t_true, t_true.mean())
            # p_scores = np.full_like(t_true, 0.5, dtype=float)
        else:
            p_scores = np.array(p_scores)
            # 避免除以 0
            p_scores = np.clip(p_scores, 0.05, 0.95)

        # 构造 DataFrame 方便排序
        df = pd.DataFrame({
            'y': y_true, 't': t_true, 'score': uplift_score, 'p': p_scores
        }).sort_values('score', ascending=False).reset_index(drop=True)
        
        N = len(df)
        
        # --- 核心修复: 使用 IPW 计算累积收益 ---
        # 这种计算方式与 evaluation.py 中的 plot_uplift_curve 逻辑对齐
        
        # 1. 计算每个样本的 IPW 贡献
        # Treatment 组收益贡献: Y / P
        # Control 组收益贡献: Y / (1-P)
        value_t = (df['t'] * df['y']) / df['p']
        value_c = ((1 - df['t']) * df['y']) / (1 - df['p'])
        
        # 2. 累积 Uplift 曲线 (Cumulative Gain)
        # 曲线高度 = 累积 (Treatment贡献 - Control贡献)
        cum_uplift = np.cumsum(value_t - value_c)
        
        # 3. 计算 AUUC (Area Under Uplift Curve)
        # 标准化 x 轴为 [0, 1]
        x_axis = np.arange(1, N + 1) / N
        auuc = auc(x_axis, cum_uplift)
        
        # 4. 计算 Qini 系数 (Qini Coefficient)
        # Qini 曲线通常指未除以总人数的累积增益，但为了鲁棒性，
        # 我们这里使用 AUUC 相对 Random 的提升来计算系数
        
        # Random Line (对角线): 假设不做筛选，随机选择 k% 的人，获得的增益应该是总增益的 k%
        total_gain = cum_uplift.iloc[-1]
        area_random = 0.5 * total_gain # 梯形面积公式简化 (0 + total) * 1 / 2
        
        # Qini 系数定义: (Area_Model - Area_Random) / Area_Random (有时分母会有不同定义，这里取最通用版)
        # 注意：如果 total_gain 接近 0，这个系数会不稳定
        if abs(area_random) < 1e-9:
            qini_coef = 0.0
        else:
            qini_coef = (auuc - area_random) / abs(area_random) # 使用 abs 避免负增益翻转

        return {'auuc': auuc, 'qini': qini_coef}
    @staticmethod
    def get_auuc_qini_rct(y_true, t_true, uplift_score, p_scores=None):
        """
        [RCT 专用版] 计算标量指标: AUUC 和 Qini Coefficient
        适用于随机对照试验 (Randomized Controlled Trial)，无需倾向性得分。
        自动适应非 1:1 的流量分配 (如 80/20 分流)。
        """
        y_true, t_true, uplift_score = np.array(y_true), np.array(t_true), np.array(uplift_score)
        
        # 构造 DataFrame 方便排序
        df = pd.DataFrame({
            'y': y_true, 
            't': t_true, 
            'score': uplift_score
        }).sort_values('score', ascending=False).reset_index(drop=True)
        
        N = len(df)
        N_t = df['t'].sum()
        N_c = N - N_t
        
        # 防止除零错误
        if N_t == 0 or N_c == 0:
            return {'auuc': 0.0, 'qini': 0.0}

        # --- 核心逻辑: RCT 下的无偏估计 ---
        # 权重 = 总样本数 / 该组样本数
        # 这等价于 IPW 中的 1 / P(T)
        w_t = N / N_t
        w_c = N / N_c
        
        # 1. 计算每个样本的贡献
        # 如果样本是 Treatment: 贡献 = Y * (N / N_t)
        # 如果样本是 Control:   贡献 = Y * (N / N_c)
        value_t = df['t'] * df['y'] * w_t
        value_c = (1 - df['t']) * df['y'] * w_c
        
        # 2. 累积 Uplift 曲线 (Cumulative Gain)
        # Curve[k] = (Sum(Y_t)_top_k * w_t) - (Sum(Y_c)_top_k * w_c)
        cum_uplift = np.cumsum(value_t - value_c)
        
        # 3. 计算 AUUC
        x_axis = np.arange(1, N + 1) / N
        auuc = auc(x_axis, cum_uplift)
        
        # 4. 计算 Qini 系数
        total_gain = cum_uplift.iloc[-1]
        area_random = 0.5 * total_gain
        
        if abs(area_random) < 1e-9:
            qini_coef = 0.0
        else:
            qini_coef = (auuc - area_random) / abs(area_random)
            
        return {'auuc': auuc, 'qini': qini_coef}
    @staticmethod
    def get_ground_truth_metrics(true_tau, pred_uplift):
        """
        [新增] 专门针对合成数据的 Ground Truth 评估指标
        """
        # 1. MSE (原有)
        mse = mean_squared_error(true_tau, pred_uplift)
        
        # 2. MAE (新增: 对异常值不敏感的误差)
        mae = mean_absolute_error(true_tau, pred_uplift)
        
        # 3. Spearman Rank Correlation (新增: 衡量排序能力的核心指标)
        # 业务价值极高：决定了是否找对了"那波人"
        rho, _ = spearmanr(true_tau, pred_uplift)
        
        # 4. Mean Bias (新增: 衡量是否系统性高估/低估)
        bias = np.mean(pred_uplift - true_tau)
        
        return {
            'MSE': mse,
            'MAE': mae,
            'Spearman': rho,
            'Bias': bias
        }

def calc_external_metrics(y_true, t_true, uplift_score):
    """
    调用 CausalML 和 scikit-uplift 进行校验。
    包含极强的容错逻辑，处理连续值 Y、索引对齐和 NaN 问题。
    """
    res = {}
    
    # 1. 极度防御性数据清洗
    # .ravel() 展平数组，去除索引
    # np.nan_to_num 填充可能存在的 NaN (防止 CausalML 报错)
    y = np.nan_to_num(np.array(y_true).ravel())
    t = np.nan_to_num(np.array(t_true).astype(int).ravel())
    s = np.nan_to_num(np.array(uplift_score).ravel())
    
    # 2. 判断 Y 是连续值还是二元值 (关键！)
    # 如果 unique 值超过 5 个，认为是连续值（回归问题）
    unique_y_count = len(np.unique(y))
    is_continuous_y = unique_y_count > 5
    
    print(f"  [Info] 目标变量 Y 唯一值数量: {unique_y_count} -> {'连续值' if is_continuous_y else '二元值'} 模式")
    print(f"  [Info] CAUSALML_AVAILABLE={CAUSALML_AVAILABLE}, SKLIFT_AVAILABLE={SKLIFT_AVAILABLE}")
    
    # --- CausalML 校验 ---
    if CAUSALML_AVAILABLE:
        try:
            # 构造纯净的 DataFrame，强制重置索引
            tmp_df = pd.DataFrame({'y': y, 't': t, 'tau': s}).reset_index(drop=True)
            
            # 使用单列字符串调用，避开列表参数 Bug
            c_gain = get_cumgain(tmp_df, outcome_col='y', treatment_col='t', treatment_effect_col='tau')
            
            # CausalML 的 get_cumgain 返回 DataFrame (index=percentile, values=gain)
            if isinstance(c_gain, pd.DataFrame):
                # 简单计算 AUC
                x_ax = np.linspace(0, 1, len(c_gain))
                y_ax = c_gain.iloc[:, 0].values
                res['CausalML_AUUC'] = auc(x_ax, y_ax)
            
            # CausalML 的 Qini (可能对连续值返回 NaN 或报错，加 try-catch)
            try:
                q_val = get_qini(tmp_df, outcome_col='y', treatment_col='t', treatment_effect_col='tau')
                if hasattr(q_val, 'iloc'):
                    res['CausalML_Qini'] = float(q_val.iloc[0])
                elif np.isscalar(q_val):
                    res['CausalML_Qini'] = float(q_val)
                else:
                    res['CausalML_Qini'] = float(q_val[0])
            except Exception:
                res['CausalML_Qini'] = -999.0 # 标记为失败

        except Exception as e:
            # 这里的报错通常是 No objects to concatenate，如果发生了说明数据清洗后为空
            print(f"  [Warning] CausalML 计算失败: {e}")
            pass 

    # --- sklift 校验 ---
    if SKLIFT_AVAILABLE:
        try:
            # 1. Uplift AUC (支持连续值 Y)
            res['sklift_AUUC'] = uplift_auc_score(y_true=y, uplift=s, treatment=t)
            
            # 2. Qini AUC (仅支持二元值 Y)
            if not is_continuous_y:
                res['sklift_Qini'] = qini_auc_score(y_true=y, uplift=s, treatment=t)
            else:
                # 连续值 Y 调用 qini_auc_score 会报 (N,) (2,) 错误，跳过
                res['sklift_Qini'] = "N/A (Cont. Y)"

        except Exception as e:
            print(f"  [Warning] sklift 计算失败: {e}")
            pass
            
    return res

def estimate_propensity(df_train, df_test, feature_cols, t_col='T', n_trials=50, random_state=42):
    """
    [兼容方案] 当没有真实倾向分时，使用分类器估计 P(T|X)
    用于观测数据的 AUUC 评估
    """
    print("  [Info] 正在估算测试集倾向性得分 (用于观测数据评估)...")
    # 1. 使用 LightGBM 估计倾向分

    X_conf = df_train[feature_cols].copy()
    T = df_train[t_col].values
    
    # y_type='discrete' 因为 T 是二元的
    propensity_model, best_params, _, _ = lgb_optuna(
        X=X_conf, 
        y=T, 
        y_type='discrete', 
        is_plot=False, 
        is_optM=(n_trials > 0), 
        n_trials=n_trials,
        random_state=random_state
    )
    
    # 预测测试集的倾向分
    p_pred = propensity_model.predict(df_test[feature_cols])
    # 截断以保证数值稳定
    return np.clip(p_pred, 0.05, 0.95)

def evaluate_model_performance(df_test, pred_uplift, p_scores, method_name, color, axes, is_observational, treat_col='T', target_col='Y', plot_decile=False):
    """
    [独立辅助函数] 评估单个模型的性能并绘图
    """

    # 1. 计算标量指标 (现在是 IPW 调整后的正确指标)
    metrics = UpliftMetrics.get_auuc_qini_rct(
        df_test[target_col], 
        df_test[treat_col], 
        pred_uplift, 
        p_scores=p_scores # <--- 关键修改: 传入 P-Score
    )

    '''
    # ==========================================
    # [新增] 插入第三方库对比校验
    # ==========================================
    # 仅在控制台打印对比，不影响 metrics 返回结构，保持原代码逻辑兼容性
    ext_metrics = calc_external_metrics(df_test[target_col].values, df_test[treat_col].values, pred_uplift)
    
    if ext_metrics:
        # 格式化打印对比信息
        print(f">>> [{method_name}] Metric Check:")
        print(f"My Impl : Qini={metrics['qini']:.4f} | AUUC={metrics['auuc']:.4f}")
        if 'CausalML_Qini' in ext_metrics:
            print(f"CausalML : Qini={ext_metrics['CausalML_Qini']:.4f} | AUUC={ext_metrics.get('CausalML_AUUC', 0):.4f}")
        if 'sklift_Qini' in ext_metrics:
            print(f"sklift : Qini={ext_metrics['sklift_Qini']:.4f}")
    # ==========================================
    '''
    # 只有在合成数据(非观测)模式下，且有 ground truth 时才计算 MSE
    if not is_observational and 'true_tau' in df_test.columns:
        # metrics['mse'] = mean_squared_error(df_test['true_tau'], pred_uplift)
        gt_metrics = UpliftMetrics.get_ground_truth_metrics(df_test['true_tau'], pred_uplift)
        # 合并字典
        metrics.update(gt_metrics)
    
    # 2. 调用 evaluation.py 进行专业绘图 (AUUC)
    # 关键点：传入 p_scores (可能是真实的，也可能是估计的， 绘图使用了 p_scores)
    evaluator = BinaryUpliftEvaluator_RCT(
        y_true=df_test[target_col], 
        t_true=df_test[treat_col], 
        uplift_pred=pred_uplift, 
        p_scores=p_scores
    )
    
    # 在左图绘制 Uplift Curve
    evaluator.plot_uplift_curve(ax=axes[0], name=f"{method_name} (Qini Coefficient = {metrics['qini']:.2f})")
    _, metrics_decile = evaluator.get_uplift_metrics(bins=10)
    metrics.update(metrics_decile)

    # 如果指定需要绘制十分位图 (通常只针对 OCU)
    if plot_decile:
        evaluator.plot_decile_chart(ax=axes[1])
        # 获取 Top 10% 的真实增益数值用于标题
        metrics_dataframe, _ = evaluator.get_uplift_metrics(bins=10)
        top_lift = metrics_dataframe.iloc[0]['real_uplift']
        axes[1].set_title(f"OCU (Ours) - Uplift by Decile\n(Top Decile Lift: {top_lift:.2f})")
        
    return metrics

# ==========================================
# 2. 复杂因果图数据生成器
# ==========================================
def generate_complex_environment_save(n=5000, seed=2025):
    np.random.seed(seed)
    
    # 1. 混杂链
    C_root = np.random.normal(0, 1.5, n)
    C_prox = 0.7 * C_root + np.random.normal(0, 1, n)
    
    # 2. 工具变量
    Z_strong = np.random.normal(0, 2, n)
    
    # 3. 精度变量
    P_strong = np.random.normal(2, 1, n)
    
    # 4. 噪音
    Noise_1 = np.random.normal(0, 1, n)
    
    # 机制生成: Treatment Assignment
    logit = 0.8 * C_prox + 1.2 * Z_strong
    prob_t = 1 / (1 + np.exp(-logit)) # True Propensity Score
    T = np.random.binomial(1, prob_t)
    
    # Outcome
    y0 = 1.5 * C_root + 2.0 * C_prox + 3.0 * P_strong + np.random.normal(0, 1, n)
    true_tau = 2.0 + 1.5 * C_prox
    y1 = y0 + true_tau
    Y = np.where(T==1, y1, y0)
    
    df = pd.DataFrame({
        'C_root': C_root, 'C_prox': C_prox,
        'Z_strong': Z_strong, 'P_strong': P_strong,
        'Noise_1': Noise_1,
        'T': T, 'Y': Y, 'true_tau': true_tau
    })
    
    graph = {
        'C_prox': ['C_root'],
        'T': ['C_prox', 'Z_strong'],
        'Y': ['C_root', 'C_prox', 'P_strong', 'T']
    }
    
    # 返回 df, graph 以及真实的倾向性得分
    return df, graph, prob_t

# import numpy as np
# import pandas as pd

def generate_complex_environment(n=5000, seed=2025):
    """
    [高维根混杂版] 数据生成器
    
    核心设计：
    1. 10个根混杂 (Roots) -> 映射为 -> 3个近端混杂 (Proximals)。
    2. 保留原有的强工具变量 (Z)、精度变量 (P) 和对撞因子 (Colliders)。
    3. 目的：制造 Trad_IPW (使用 Roots+Prox) 的多重共线性和过拟合困境，
       突显 OCU (仅使用 Prox) 的最小充分集优势。
    """
    np.random.seed(seed)
    
    # 辅助函数：生成外生残差
    def resid(n, scale=0.1):
        return np.random.normal(0, scale, n)

    # ==========================================
    # 1. 根节点 (Root Nodes) - 扩大至 10 维
    # ==========================================
    n_roots = 10
    # 生成 10 个相互独立的标准正态根变量
    C_roots = np.random.normal(0, 1, size=(n, n_roots))
    root_cols = [f'C_root_{i}' for i in range(n_roots)]
    
    # 将其放入字典方便后续打包
    data_roots = {col: C_roots[:, i] for i, col in enumerate(root_cols)}

    # ==========================================
    # 2. 保留原有的 工具变量、精度变量、噪音
    # ==========================================
    # 工具变量 (IV) - 只影响 T
    Z_1 = np.random.normal(0, 1, n)
    Z_2 = np.random.uniform(-2, 2, n)
    
    # 精度变量 (Precision) - 只影响 Y
    P_1 = np.random.normal(0, 1, n)
    P_2 = np.random.chisquare(df=2, size=n) - 2
    
    # 纯噪音
    Noise_1 = np.random.normal(0, 1, n)
    Noise_2 = np.random.normal(0, 1, n)

    # ==========================================
    # 3. 中间混杂变量 (Proximal Confounders)
    # ==========================================
    # 核心逻辑：将 10 个 Roots 压缩为 3 个 Proximals
    # 这构成了信息的“瓶颈”，也是 Trad_IPW 容易出错的地方
    
    # --- C_prox_1: 由 Roots[0~3] 决定 (混合线性与非线性) ---
    # 逻辑：Root_0 线性，Root_1 Tanh 非线性，Root_2/3 交互
    raw_prox_1 = (
        1.2 * C_roots[:, 0] + 
        0.8 * np.tanh(C_roots[:, 1]) + 
        0.5 * (C_roots[:, 2] * C_roots[:, 3]) + 
        resid(n)
    )
    # 归一化以保持数值稳定性
    C_prox_1 = (raw_prox_1 - raw_prox_1.mean()) / raw_prox_1.std()

    # --- C_prox_2: 由 Roots[3~6] 决定 (强非线性) ---
    # 注意：Root_3 是与 C_prox_1 的重叠变量，制造 Prox 之间的相关性
    raw_prox_2 = (
        0.5 * (C_roots[:, 3] ** 2) + 
        np.sin(C_roots[:, 4] * 2) + 
        0.7 * np.abs(C_roots[:, 5]) - 
        0.5 * C_roots[:, 6] +
        resid(n)
    )
    C_prox_2 = (raw_prox_2 - raw_prox_2.mean()) / raw_prox_2.std()

    # --- C_prox_3: 由 Roots[6~9] 决定 (阈值/阶跃效应) ---
    # Root_6 是重叠变量
    raw_prox_3 = (
        1.0 * C_roots[:, 7] + 
        0.8 * np.where(C_roots[:, 8] > 0, 1.0, -1.0) + # 阶跃
        0.4 * np.exp(C_roots[:, 9] / 2.0) +
        0.3 * C_roots[:, 6] +
        resid(n)
    )
    C_prox_3 = (raw_prox_3 - raw_prox_3.mean()) / raw_prox_3.std()

    # ==========================================
    # 4. 策略/干预分配 (Treatment Assignment T)
    # ==========================================
    # Parents: C_prox (1,2,3) + Z (1,2)
    # 保持原逻辑，但变量已经是经过高维压缩后的 Prox
    logit = (
        1.0 * C_prox_1 + 
        -0.8 * C_prox_2 + 
        0.6 * C_prox_3 + 
        0.4 * (C_prox_1 * C_prox_2) + # 混杂交互
        1.5 * Z_1 +              # Strong Linear IV
        0.8 * np.sin(Z_2 * 2)    # Non-linear IV
    )
    
    logit = (logit - logit.mean()) / logit.std() * 2.5
    prob_t = 1 / (1 + np.exp(-logit))
    T = np.random.binomial(1, prob_t)

    # ==========================================
    # 5. 结果变量 (Outcome Y)
    # ==========================================
    # Parents: C_prox (1,2,3) + P (1,2) + T
    
    # Baseline Y0
    y0 = (
        0.5 * C_prox_1 + 
        0.5 * (C_prox_2 ** 2) + 
        0.3 * np.abs(C_prox_3) + 
        2.5 * P_1 +                   # Strong Precision
        1.5 * np.log1p(np.abs(P_2)) + # Non-linear Precision
        resid(n, scale=0.5)
    )
    
    # True Uplift
    true_tau = (
        1.0 + 
        1.5 * (C_prox_1 > 0.5).astype(float) + 
        0.5 * C_prox_2 + 
        0.3 * P_1 
    )
    
    y1 = y0 + true_tau
    Y = np.where(T == 1, y1, y0)

    # ==========================================
    # 6. 对撞因子 (Colliders) - 保持不变
    # ==========================================
    Collid_1 = 0.6 * T + 0.4 * Y + resid(n)
    Collid_2 = np.sin(T * Y) + resid(n)

    # ==========================================
    # 7. 数据打包 & 图定义
    # ==========================================
    # 合并所有数据
    data = data_roots.copy()
    data.update({
        'C_prox_1': C_prox_1, 'C_prox_2': C_prox_2, 'C_prox_3': C_prox_3,
        'Z_1': Z_1, 'Z_2': Z_2,
        'P_1': P_1, 'P_2': P_2,
        'Noise_1': Noise_1, 'Noise_2': Noise_2,
        'Collid_1': Collid_1, 'Collid_2': Collid_2,
        'T': T, 'Y': Y, 
        'true_tau': true_tau
    })
    df = pd.DataFrame(data)
    
    # 定义因果图结构
    graph = {}
    
    # 1. Roots 没有父节点
    for col in root_cols:
        graph[col] = []
    
    # 2. 其他外生变量无父节点
    for col in ['Z_1', 'Z_2', 'P_1', 'P_2', 'Noise_1', 'Noise_2']:
        graph[col] = []
        
    # 3. Proximals 的父节点 (映射关系)
    graph['C_prox_1'] = ['C_root_0', 'C_root_1', 'C_root_2', 'C_root_3']
    graph['C_prox_2'] = ['C_root_3', 'C_root_4', 'C_root_5', 'C_root_6'] # C_root_3 重叠
    graph['C_prox_3'] = ['C_root_6', 'C_root_7', 'C_root_8', 'C_root_9'] # C_root_6 重叠
    
    # 4. T 的父节点: Proximals + Z
    # 注意：真实机制中，T 仅由 C_prox 直接决定，Roots 是通过 C_prox 间接影响
    graph['T'] = ['C_prox_1', 'C_prox_2', 'C_prox_3', 'Z_1', 'Z_2']
    
    # 5. Y 的父节点: Proximals + P + T
    graph['Y'] = ['C_prox_1', 'C_prox_2', 'C_prox_3', 'P_1', 'P_2', 'T']
    
    # 6. Colliders
    graph['Collid_1'] = ['T', 'Y']
    graph['Collid_2'] = ['T', 'Y']
    
    return df, graph, prob_t

# ==========================================
# 3. 单个模型消融实验运行器
# ==========================================
def run_ablation_study(base_line, model_name, df_train, df_test, graph, feature_cols, treat_col='T', target_col='Y', p_test_true = None, is_observational=False, verbose=False, deconfound_method='IPW', n_trials=100, reuseKwargs=None):
    if verbose: 
        print(f"\n[{model_name}] 启动消融实验...")
    
    # --- 关键逻辑: 确定评估用的 P-Score ---
    true_ps_features = graph[treat_col]

    if is_observational:
        # [观测数据模式] 假设我们不知道 p_test_true，必须从数据中估算
        # 使用训练集训练一个 Propensity Model，预测测试集
        # p_eval = estimate_propensity(df_train, df_test, true_ps_features, t_col=treat_col)
        p_eval = None
    else:
        # [合成数据模式] 使用上帝视角的真实 PS，评估最准确
        p_eval = p_test_true
        # p_eval = None

    results = {}

    # 初始化绘图画布
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    sns.set_style("whitegrid")

    # --- Exp A: Naive ---
    try:
        print(f"  [{model_name}] 运行 Naive 基线...")
        model = copy.deepcopy(base_line)
        if hasattr(model, 'fit'):
            model.fit(df_train[feature_cols].values, df_train[target_col].values, df_train[treat_col].values)
            pred = model.predict(df_test[feature_cols].values)
            results['Naive'] = evaluate_model_performance(
                df_test, pred, p_eval, 'Naive', 'gray', axes, is_observational, treat_col=treat_col, target_col=target_col, plot_decile=False
            )
    except Exception as e:
        if verbose: print(f"  [Naive] Failed: {e}")

    pre_use_ps = False
    if model_name == 'X-Learner':
        pre_use_ps = True
    
    if reuseKwargs is not None and len(reuseKwargs) > 0:
        try:
            df_stg1_c_total, w_c_total, ps_c_total, c_total, s_prec, ps_model_c_total, df_stg1, w, ps, c_opt, ps_model = reuseKwargs
        except Exception as e:
            print(f"  [ReuseKwargs] Failed to unpack: {e}")
            return
    else:
        print("  [Info] 未提供重用参数，全部重新计算第一阶段数据...")
        df_stg1_c_total, w_c_total, ps_c_total, c_total, s_prec, ps_model_c_total = None, None, None, None, None, None
        df_stg1, w, ps, c_opt, ps_model = None, None, None, None, None
    
    # --- Exp B-1: Trad IPW (No Prec) ---

    try:
        if df_stg1_c_total is None:
            deconfounder = FirstStageDeconfounder(treat_col, target_col, graph, deconfound_method, conf_type='total', is_precise=True)
            df_stg1_c_total, w_c_total, ps_c_total, c_total, s_prec, ps_model_c_total = deconfounder.fit_transform(df_train, n_trials=n_trials)
        else:
            print("  [Info] 重用已有的第一阶段数据 (Trad IPW No Prec)...")
        
        trainer = SecondStageUpliftTrainer(base_line, ps_model_c_total, c_total, s_prec = [])  
        trainer.fit(df_stg1_c_total, treat_col, target_col, ps_c_total, w_c_total, verbose=False)
        pred = trainer.predict(df_test, use_ps = pre_use_ps)
        results['Trad_IPW_NoPrec'] = evaluate_model_performance(
            df_test, pred, p_eval, 'Trad_IPW_NoPrec', 'blue', axes, is_observational, treat_col=treat_col, target_col=target_col, plot_decile=False
        )
    except Exception as e:
        if verbose: print(f"  [Trad_IPW_NoPrec] Failed: {e}")

    # --- Exp B-2: Trad IPW (With Prec) ---
    try:
        if df_stg1_c_total is None:
            deconfounder = FirstStageDeconfounder(treat_col, target_col, graph, deconfound_method, conf_type='total', is_precise=True)
            df_stg1_c_total, w_c_total, ps_c_total, c_total, s_prec, ps_model_c_total = deconfounder.fit_transform(df_train, n_trials=n_trials)
        else:
            print("  [Info] 重用已有的第一阶段数据 (Trad IPW With Prec)...")

        trainer = SecondStageUpliftTrainer(base_line, ps_model_c_total, c_total, s_prec)
        trainer.fit(df_stg1_c_total, treat_col, target_col, ps_c_total, w_c_total, verbose=False)
        pred = trainer.predict(df_test, use_ps = pre_use_ps)
        results['Trad_IPW_WithPrec'] = evaluate_model_performance(
            df_test, pred, p_eval, 'Trad_IPW_WithPrec', 'cyan', axes, is_observational, treat_col=treat_col, target_col=target_col, plot_decile=False
        )
    except Exception as e:
        if verbose: print(f"  [Trad_IPW_WithPrec] Failed: {e}")

    # --- Exp C: OCU-Framework (Ours) ---
    try:
        if df_stg1 is None:
            deconfounder = FirstStageDeconfounder(treat_col, target_col, graph, deconfound_method, conf_type='proximal', is_precise=True)
            df_stg1, w, ps, c_opt, s_prec, ps_model = deconfounder.fit_transform(df_train, n_trials=n_trials)
        else:
            print("  [Info] 重用已有的第一阶段数据 (OCU Ours)...")

        trainer = SecondStageUpliftTrainer(base_line, ps_model, c_opt, s_prec)
        trainer.fit(df_stg1, treat_col, target_col, ps, w, verbose=False)
        pred = trainer.predict(df_test, use_ps = pre_use_ps)
        results['OCU_Ours'] = evaluate_model_performance(
            df_test, pred, p_eval, 'OCU_Ours', 'red', axes, is_observational, treat_col=treat_col, target_col=target_col, plot_decile=True
        )
    except Exception as e:
        if verbose: print(f"  [OCU_Ours] Failed: {e}")
        
    # --- 图表美化与保存 ---
    axes[0].set_title(f"{model_name} - Ablation Study: AUUC Curves")
    axes[0].grid(True, linestyle='--', alpha=0.6)
    
    save_path = f'./results/{model_name}_ablation_study_results.png'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close() # 关闭图形防止内存泄漏
    
    if verbose: print(f"  -> 可视化结果已保存: {save_path}")

    reuseKwargs = (df_stg1_c_total, w_c_total, ps_c_total, c_total, s_prec, ps_model_c_total, df_stg1, w, ps, c_opt, ps_model)
    return results, reuseKwargs

def preprocess_data(Ancestors_Graph, dataPath, target_col, treat_col, treat_space = None, target_space = None, datasetName = 'TTS'):
    """
    数据预处理函数
    """
    # 1. 处理 Ancestors_Graph
    print("步骤1: 处理 Ancestors_Graph...")

    # 收集所有在值中出现过的变量
    all_values = []
    for value_list in Ancestors_Graph.values():
        all_values.extend(value_list)

    # 获取所有当前作为键的变量
    all_keys = set(Ancestors_Graph.keys())

    # 找出在值中出现但未作为键的变量
    missing_keys = set(all_values) - all_keys

    # 将这些变量作为新键添加到Ancestors_Graph中，值为空列表
    for key in missing_keys:
        Ancestors_Graph[key] = []

    print(f"已添加 {len(missing_keys)} 个缺失的键: {missing_keys}")

    # 2. 读取数据
    print(f"\n步骤2: 读取数据 {dataPath}...")
    try:
        df = pd.read_csv(dataPath)
        print(f"数据读取成功，形状: {df.shape}")
    except FileNotFoundError:
        print(f"错误: 未找到文件 {dataPath}")
        return None, None, None
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return None, None, None

    # 3. 提取需要的特征
    print("\n步骤3: 提取所需特征...")

    # 获取所有需要的特征
    all_features = set()
    for key, value_list in Ancestors_Graph.items():
        all_features.add(key)
        all_features.update(value_list)

    # 添加目标列和处理列
    # all_features.add(target_col)
    # all_features.add(treat_col)

    print(f"总共有 {len(all_features)} 个特征需要提取")

    # 检查数据中是否包含所有需要的特征
    missing_features = all_features - set(df.columns)
    if missing_features:
        print(f"警告: 数据中缺失以下 {len(missing_features)} 个特征:")
        for feature in missing_features:
            print(f"  - {feature}")

    # 提取特征
    available_features = all_features.intersection(set(df.columns))
    df = df[list(available_features)]
    print(f"实际提取了 {len(available_features)} 个特征")

    if datasetName == 'TTS':
        # 4. 处理 treat_col
        print(f"\n步骤4: 处理 {treat_col} 列...")

        # 检查 treat_col 是否存在
        if treat_col not in df.columns:
            print(f"错误: 数据中不存在 {treat_col} 列")
            return None, None, None

        # 获取原始值分布
        original_counts = df[treat_col].value_counts()
        print(f"原始 {treat_col} 值分布:")
        for value, count in original_counts.items():
            print(f"  {value}: {count} 行")

        # 方法1: 清洗字符串 - 去除等号和引号
        print(f"\n正在清洗 {treat_col} 列数据...")

        # 将列转换为字符串类型
        df[treat_col] = df[treat_col].astype(str)

        # 清洗数据：去除开头的 '="' 和结尾的 '"'
        df[treat_col] = df[treat_col].str.strip()  # 先去除首尾空格

        # 使用正则表达式去除 '="' 前缀和 '"' 后缀
        pattern = r'^="?(.+?)"?$'  # 匹配 '="xxx"' 或 '"xxx"' 或 'xxx'
        df[treat_col] = df[treat_col].apply(lambda x: re.search(pattern, x).group(1) if re.search(pattern, x) else x)

        # 再次查看清洗后的分布
        cleaned_counts = df[treat_col].value_counts()
        print(f"清洗后 {treat_col} 值分布:")
        for value, count in cleaned_counts.items():
            print(f"  {value}: {count} 行")

        # 映射处理
        # 创建映射条件
        mask = df[treat_col].isin(treat_space.keys())

        # 显示匹配情况
        matched_counts = df[treat_col][mask].value_counts()
        print(f"\n匹配到映射字典的值:")
        for value, count in matched_counts.items():
            print(f"  {value}: {count} 行")

        # 应用映射
        df.loc[mask, treat_col] = df.loc[mask, treat_col].map(treat_space)

        # 删除不匹配的行
        original_rows = len(df)
        df = df[mask].copy()
        df[treat_col] = df[treat_col].astype(int)  # 转换为数值类型
        removed_rows = original_rows - len(df)

        print(f"\n已删除 {removed_rows} 行不匹配的数据")
        print(f"剩余 {len(df)} 行数据")
        print(f"处理后的 {treat_col} 值分布:")
        print(df[treat_col].value_counts().sort_index())

        # 5. 处理 target_col（新增功能）
        if target_space is not None:
            print(f"\n步骤5: 处理 {target_col} 列的区间映射...")

            # 检查 target_col 是否存在
            if target_col not in df.columns:
                print(f"错误: 数据中不存在 {target_col} 列")
                return None, None, None

            # 将target_col转换为数值类型
            print(f"将 {target_col} 转换为数值类型...")
            try:
                df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
            except Exception as e:
                print(f"转换 {target_col} 为数值类型时出错: {e}")
                return None, None, None

            # 检查缺失值
            nan_count = df[target_col].isna().sum()
            if nan_count > 0:
                print(f"警告: {target_col} 列有 {nan_count} 个无法转换的值，已设为NaN")

            # 显示原始值的统计信息
            print(f"\n{target_col} 原始值统计:")
            print(f"最小值: {df[target_col].min():.2f}")
            print(f"最大值: {df[target_col].max():.2f}")
            print(f"平均值: {df[target_col].mean():.2f}")
            print(f"中位数: {df[target_col].median():.2f}")

            # 定义区间映射函数
            def map_interval(value, interval_dict):
                """
                将数值映射到区间标签
                """
                if pd.isna(value):
                    return None

                for label, (lower, upper) in interval_dict.items():
                    # 处理开区间和闭区间
                    if lower is not None and upper is not None:
                        if lower <= value < upper:  # 左闭右开
                            return label
                    elif lower is not None and upper is None:
                        if lower <= value:  # 大于等于下限
                            return label
                    elif lower is None and upper is not None:
                        if value < upper:  # 小于上限
                            return label

                return None  # 不在任何区间内

            # 应用区间映射
            print(f"\n应用区间映射: {target_space}")

            # 创建映射列
            target_mapped = df[target_col].apply(lambda x: map_interval(x, target_space))

            # 统计映射结果
            mapped_counts = target_mapped.value_counts(dropna=False)
            print(f"映射结果分布:")
            for value, count in mapped_counts.items():
                print(f"  {value}: {count} 行")

            # 只保留在区间内的行
            original_rows_target = len(df)
            mask_target = target_mapped.notna()
            df = df[mask_target].copy()

            # 将映射结果赋值给target_col
            df[target_col] = target_mapped[mask_target].astype(int)  # 转换为整数类型

            removed_rows_target = original_rows_target - len(df)
            print(f"\n已删除 {removed_rows_target} 行不在目标区间的数据")
            print(f"剩余 {len(df)} 行数据")
            print(f"处理后的 {target_col} 值分布:")
            print(df[target_col].value_counts().sort_index())

            # 显示各区间的原始值范围
            print(f"\n各区间原始值统计:")
            for label, (lower, upper) in target_space.items():
                if lower is not None and upper is not None:
                    print(f"区间 {label} [{lower}, {upper}):")
                elif lower is not None and upper is None:
                    print(f"区间 {label} [{lower}, ∞):")
                elif lower is None and upper is not None:
                    print(f"区间 {label} (-∞, {upper}):")
        else:
            print(f"\n步骤5: 跳过 {target_col} 的区间映射 (target_space=None)")

    # 6. 返回结果
    print(f"\n步骤6: 数据预处理完成!")
    print(f"最终数据形状: {df.shape}")
    print(f"特征数量: {df.shape[1]}")
    print(f"样本数量: {df.shape[0]}")

    return df, Ancestors_Graph, all_features

def preprocess_non_numeric_columns(dataFrame, missing_values, missFill = -9990):
    """
    对数据框中的非数值列进行LabelEncode处理，缺失值保持缺失状态或使用固定编码填充。

    参数:
    dataFrame - 输入的DataFrame

    返回:
    processed_df - 处理后的DataFrame
    """
    # 获取数值列和非数值列
    numeric_cols = dataFrame.select_dtypes(include=['number']).columns
    non_numeric_cols = dataFrame.select_dtypes(exclude=['number']).columns

    # 初始化LabelEncoder
    le = LabelEncoder()

    # 处理非数值列
    for col in non_numeric_cols:
        # 保存缺失值的索引
        missing_indices = dataFrame[col].isnull() | dataFrame[col].isna() | dataFrame[col].isin(missing_values)

        # 对非缺失值进行LabelEncode
        non_missing_data = dataFrame.loc[~missing_indices, col].astype(str)
        encoded_data = le.fit_transform(non_missing_data)

        # 将编码后的数据放回原DataFrame
        dataFrame.loc[~missing_indices, col] = encoded_data

        # 对缺失值进行填充，例如使用 missFill
        dataFrame.loc[missing_indices, col] = missFill

    # 打印处理信息
    if len(non_numeric_cols) > 0:
        print(f"对非数值列进行了LabelEncode处理：{non_numeric_cols}")
    else:
        print("没有非数值列需要处理")

    return dataFrame

# ==========================================
# 4. 批量实验主引擎
# ==========================================
def main(is_observational=False, deconfound_method='IPW'):
    print(f"🚀 启动批量实验引擎 (Observational Mode: {is_observational})")
    print("="*60)

    base_models = {
        "S-Learner": SLearner(base_estimator=XGBRegressor(n_estimators=100, verbosity=0)),
        "T-Learner": TLearner(estimator_t=XGBRegressor(n_estimators=100, verbosity=0)),
        "X-Learner": XLearner(outcome_learner=XGBRegressor(n_estimators=100, max_depth=5, verbosity=0),
                              effect_learner=XGBRegressor(n_estimators=100, max_depth=5, verbosity=0)),
        "ClassTransform": ClassTransformEstimator(base_estimator=XGBRegressor(n_estimators=100, verbosity=0)),
        "UpliftForest": UniversalUpliftForest(n_estimators=100, max_depth=5, min_samples_treatment=200) 
    }

    if is_observational:
        print("注意: 当前处于观测数据模式，Propensity Score 将通过模型估计。")
        datasetName = 'lenta' # 'TTS' 'lenta' 

        if datasetName == 'TTS':
            # Ancestors_Graph = {'main_talk_dur': ['duotou_3rd_cnt_new', 'is_pay_study_cust', 'age', 'login_count_30d', 'xyl_score', 'voice_role_cd'], 'duotou_3rd_cnt_new': ['login_count_90d', 'last_mon_ext_apply_cnt', 'last_5103_wdraw_to_date'], 'voice_role_cd': ['census_pr_cd','credit_lim_new','age','min_recent_login_days','last_5103_appl_time_to_date','reg_dt_to_date','last_5103_wdraw_to_date','accum_loan_amt','occ_cd','ocr_ethnic'], 'is_pay_study_cust': ['last_5103_appl_time_to_date', 'gender_cd', 'is_internet_busi_cust', 'is_freq_traveler', 'age'], 'age': ['pboc_ext_max_credit_lim', 'last_180d_5103_loan_amt', 'last_5103_wdraw_to_date'], 'login_count_30d': ['login_count_7d', 'login_count_3d'], 'xyl_score': ['last_180d_5103_wdraw_app_amt', 'last_180_day_5103_wdraw_baff_cnt', 'last_180d_5103_wdraw_app_days', 'last_180d_5103_lim_use_rate', 'duotou_3rd_cnt_new', 'login_count_30d'], 'pboc_ext_max_credit_lim': ['card_credit_avg_lvl'], 'last_180_day_5103_wdraw_baff_cnt': ['last_180d_5103_lim_use_rate', 'last_180_day_5103_appl_wdraw_refuse_cnt', 'last_180d_5103_wdraw_app_amt', 'last_180d_5103_wdraw_app_days', 'last_180d_5103_lim_pass_rate'], 'login_count_7d': ['last_180d_5103_wdraw_app_days', 'login_count_3d', 'last_5103_wdraw_to_date'], 'last_180d_5103_lim_use_rate': ['last_180_day_5103_wdraw_baff_cnt', 'last_180d_5103_loan_days', 'last_180_day_5103_appl_wdraw_refuse_cnt', 'last_180d_5103_wdraw_app_amt', 'last_180d_5103_wdraw_app_days'], 'login_count_3d': [], 'gender_cd': ['risk_score'], 'last_180d_5103_loan_amt': ['last_180d_5103_loan_days', 'last_180d_5103_lim_pass_rate'], 'last_180d_5103_wdraw_app_amt': ['last_180d_5103_wdraw_app_days', 'last_180_day_5103_wdraw_baff_cnt', 'last_180d_5103_loan_amt'], 'is_internet_busi_cust': [], 'is_freq_traveler': ['edu_deg_cd', 'house_loan_cnt_lvl', 'is_internet_busi_cust'], 'last_180d_5103_wdraw_app_days': ['last_180d_5103_wdraw_app_amt'], 'last_mon_ext_apply_cnt': [], 'last_5103_wdraw_to_date': ['login_count_7d', 'last_180_day_5103_appl_wdraw_refuse_cnt'], 'edu_deg_cd': ['marital_status_cd'], 'risk_score': ['last_180d_5103_lim_pass_rate', 'user_max_dpd'], 'house_loan_cnt_lvl': ['credit_org_num_lvl', 'income_range_ind', 'zx_job_name', 'marital_status_cd', 'bus_loan_cnt_lvl', 'zx_job_title', 'zx_empl_status', 'zx_occ_th2', 'edu_deg_cd', 'zx_occ_th1', 'zx_occ_th3', 'car_loan_cnt_lvl', 'occ_cd'], 'last_180d_5103_loan_days': ['credit_org_num_lvl', 'income_range_ind', 'marital_status_cd', 'bus_loan_cnt_lvl', 'card_credit_avg_lvl', 'zx_occ_th2', 'edu_deg_cd', 'last_180_day_5103_loan_wdraw_cnt', 'cl_org_num_lvl', 'zx_occ_th1', 'last_30_day_5103_loan_wdraw_cnt', 'car_loan_cnt_lvl'], 'last_180d_5103_lim_pass_rate': ['spl_prin_sum'], 'card_credit_avg_lvl': ['credit_org_num_lvl', 'income_range_ind', 'zx_job_name', 'marital_status_cd', 'bus_loan_cnt_lvl', 'zx_job_title', 'zx_empl_status', 'zx_occ_th2', 'edu_deg_cd', 'house_loan_cnt_lvl', 'last_180_day_5103_loan_wdraw_cnt', 'zx_occ_th1', 'zx_occ_th3', 'car_loan_cnt_lvl', 'occ_cd'], 'spl_prin_sum': ['last_30_day_5103_loan_wdraw_cnt'], 'last_30_day_5103_loan_wdraw_cnt': ['is_tel_sale_pos_answer_user'], 'user_max_dpd': ['residence_pr_cd'], 'marital_status_cd': ['is_student_cust_gp'], 'residence_pr_cd': ['census_pr_cd'], 'is_tel_sale_pos_answer_user': [], 'is_student_cust_gp': ['census_pr_cd', 'residence_pr_cd'], 'census_pr_cd': []}
            Ancestors_Graph = {
                'main_talk_dur': ['duotou_3rd_cnt_new', 'is_pay_study_cust', 'age', 'login_count_30d', 'xyl_score', 'voice_role_cd'], 
                'duotou_3rd_cnt_new': ['login_count_90d', 'last_mon_ext_apply_cnt', 'last_5103_wdraw_to_date'], 
                'voice_role_cd': ['census_pr_cd','credit_lim_new','age','min_recent_login_days','last_5103_appl_time_to_date','reg_dt_to_date','last_5103_wdraw_to_date','accum_loan_amt','occ_cd','ocr_ethnic'], 
                'is_pay_study_cust': ['last_5103_appl_time_to_date', 'gender_cd', 'is_internet_busi_cust', 'is_freq_traveler', 'age'], 
                'age': [], 
                'login_count_30d': ['login_count_7d', 'login_count_3d'], 
                'xyl_score': ['last_180d_5103_wdraw_app_amt', 'last_180_day_5103_wdraw_baff_cnt', 'last_180d_5103_wdraw_app_days', 'last_180d_5103_lim_use_rate', 'duotou_3rd_cnt_new', 'login_count_30d'], 
                'last_180_day_5103_wdraw_baff_cnt': ['last_180d_5103_lim_use_rate', 'last_180_day_5103_appl_wdraw_refuse_cnt', 'last_180d_5103_wdraw_app_amt', 'last_180d_5103_wdraw_app_days', 'last_180d_5103_lim_pass_rate'], 
                'login_count_7d': ['last_180d_5103_wdraw_app_days', 'login_count_3d', 'last_5103_wdraw_to_date'], 
                'last_180d_5103_lim_use_rate': ['last_180_day_5103_wdraw_baff_cnt', 'last_180d_5103_loan_days', 'last_180_day_5103_appl_wdraw_refuse_cnt', 'last_180d_5103_wdraw_app_amt', 'last_180d_5103_wdraw_app_days'], 
                'login_count_3d': [], 
                'gender_cd': [], 
                'last_180d_5103_loan_amt': ['last_180d_5103_loan_days', 'last_180d_5103_lim_pass_rate'], 
                'last_180d_5103_wdraw_app_amt': ['last_180d_5103_wdraw_app_days', 'last_180_day_5103_wdraw_baff_cnt', 'last_180d_5103_loan_amt'], 
                'is_internet_busi_cust': [], 
                'is_freq_traveler': ['edu_deg_cd', 'house_loan_cnt_lvl', 'is_internet_busi_cust'], 
                'last_180d_5103_wdraw_app_days': ['last_180d_5103_wdraw_app_amt'], 
                'last_mon_ext_apply_cnt': [], 
                'last_5103_wdraw_to_date': ['login_count_7d', 'last_180_day_5103_appl_wdraw_refuse_cnt'], 
                'edu_deg_cd': ['marital_status_cd'], 
                'house_loan_cnt_lvl': ['credit_org_num_lvl', 'income_range_ind', 'zx_job_name', 'marital_status_cd', 'bus_loan_cnt_lvl', 'zx_job_title', 'zx_empl_status', 'zx_occ_th2', 'edu_deg_cd', 'zx_occ_th1', 'zx_occ_th3', 'car_loan_cnt_lvl', 'occ_cd'], 
                'last_180d_5103_loan_days': ['credit_org_num_lvl', 'income_range_ind', 'marital_status_cd', 'bus_loan_cnt_lvl', 'card_credit_avg_lvl', 'zx_occ_th2', 'edu_deg_cd', 'last_180_day_5103_loan_wdraw_cnt', 'cl_org_num_lvl', 'zx_occ_th1', 'last_30_day_5103_loan_wdraw_cnt', 'car_loan_cnt_lvl'], 
                'last_180d_5103_lim_pass_rate': ['spl_prin_sum'], 
                'card_credit_avg_lvl': ['credit_org_num_lvl', 'income_range_ind', 'zx_job_name', 'marital_status_cd', 'bus_loan_cnt_lvl', 'zx_job_title', 'zx_empl_status', 'zx_occ_th2', 'edu_deg_cd', 'house_loan_cnt_lvl', 'last_180_day_5103_loan_wdraw_cnt', 'zx_occ_th1', 'zx_occ_th3', 'car_loan_cnt_lvl', 'occ_cd'], 
                'spl_prin_sum': ['last_30_day_5103_loan_wdraw_cnt'], 
                'last_30_day_5103_loan_wdraw_cnt': ['is_tel_sale_pos_answer_user'], 
                'marital_status_cd': ['is_student_cust_gp'], 
                'residence_pr_cd': ['census_pr_cd'], 
                'is_tel_sale_pos_answer_user': [], 
                'is_student_cust_gp': ['census_pr_cd', 'residence_pr_cd'], 
                'census_pr_cd': []
            }

            dataPath = './data/Total8W_EDU_MARITAL_Notnull.csv'  # Total8W_EDU_MARITAL_Notnull Random3W_EDU_MARITAL_Notnull
            target_col = 'main_talk_dur'
            treat_col = 'voice_role_cd'

            treat_space = {'maxiaoxi': 1, 'maxiaoze': 0}
            # target_space = {'0': [0,10], '1': [30, 9999999]}
            target_space = None

            df_processed, graph, feature_set = preprocess_data(Ancestors_Graph, dataPath, target_col, treat_col, treat_space, target_space = target_space, datasetName = 'TTS')
            print(f'UnionAncestorsGraph: {graph}')
            feature_set = list(feature_set)

            p_test_true = None
            feature_cols = [c for c in feature_set if c not in [target_col, treat_col]]
            dataFrame = preprocess_non_numeric_columns(df_processed, missing_values = ['01.缺失','09.缺失'], missFill = -9990)

            feature_setType = {}
            for col_ in feature_set:
                # 自动检测变量类型
                unique_ratio = dataFrame[col_].nunique() / len(dataFrame)
                if (unique_ratio < 0.01 and dataFrame[col_].nunique() <= 20):
                    feature_setType[col_] = 'discrete'
                    # dataFrame[col_] = dataFrame[col_].fillna(-9990).astype(np.int32)
                    dataFrame[col_] = dataFrame[col_].astype(np.int32)
                    # dataFrame[col_] = dataFrame[col_].fillna(-9990).infer_objects(copy=False).astype(np.int32)
                else:
                    feature_setType[col_] = 'continuous'
                    dataFrame[col_] = dataFrame[col_].astype(np.float32)

            if dataFrame[target_col].nunique() > 10:
                feature_setType[target_col] = 'continuous'
                dataFrame[target_col] = dataFrame[target_col].astype(np.float32)

            print("The type of {} is {}.".format(target_col, feature_setType[target_col]))
            print("变量类型检测结果:")
            for col, vtype in feature_setType.items():
                print(f"  {col}: {vtype} (唯一值数: {dataFrame[col].nunique()})")

            print("修改后列名:", feature_set)  # Modified: 打印列名
            print("数据类型:\n", dataFrame.dtypes)  # Modified: 打印数据类型

            '''
            ################数据预处理优化################
            print("开始高级数据预处理...")

            # 确定需要异常值检测的列（连续变量）
            outlier_detection_cols = [col for col in feature_set if feature_setType.get(col, 'continuous') == 'continuous']

            # 确定需要标准化的列（连续变量，排除目标变量）
            # standardization_cols = [col for col in outlier_detection_cols if col != targetCol]
            standardization_cols = outlier_detection_cols

            print(f"异常值检测变量: {len(outlier_detection_cols)} 个")
            print(f"标准化变量: {len(standardization_cols)} 个")

            # 运行高级预处理
            processed_dataFrame, normal_indices = advanced_data_preprocessor(
                df=dataFrame,
                outlier_detection_cols=outlier_detection_cols,
                standardization_cols=standardization_cols,
                outlier_method='iqr', # choices=['iqr', 'zscore', 'mad']
                outlier_threshold=5.0,
                missing_values=[-99, -9990]
            )

            print("高级数据预处理完成！")
            print(f"预处理后的数据形状: {processed_dataFrame.shape}")

            cleaned_data = get_cleaned_data_fast(variables=feature_set,
                                                normal_indices=normal_indices,
                                                processed_df=processed_dataFrame,
                                                min_samples=5
                                                )
            '''
            cleaned_data = dataFrame.copy()
        elif datasetName == 'lenta':
            Ancestors_Graph = {'response_att': ['k_var_cheque_category_width_15d', 'k_var_days_between_visits_1m', 'k_var_disc_per_cheque_15d', 'k_var_cheque_15d', 'mean_discount_depth_15d', 'group'], 'group': ['crazy_purchases_cheque_count_12m', 'sale_count_12m_g32', 'sale_sum_12m_g26', 'cheque_count_12m_g56', 'disc_sum_6m_g34', 'cheque_count_12m_g32', 'promo_share_15d'], 'k_var_cheque_category_width_15d': ['food_share_15d', 'k_var_disc_per_cheque_15d', 'promo_share_15d', 'k_var_cheque_group_width_15d', 'mean_discount_depth_15d'], 'k_var_days_between_visits_1m': ['stdev_days_between_visits_15d', 'k_var_days_between_visits_15d', 'cheque_count_6m_g40', 'food_share_1m', 'k_var_disc_per_cheque_15d', 'perdelta_days_between_visits_15_30d', 'cheque_count_6m_g32', 'cheque_count_12m_g41', 'k_var_cheque_category_width_15d'], 'k_var_disc_per_cheque_15d': ['cheque_count_12m_g41', 'mean_discount_depth_15d'], 'k_var_cheque_15d': ['k_var_cheque_category_width_15d', 'stdev_days_between_visits_15d', 'k_var_days_between_visits_1m', 'k_var_days_between_visits_15d', 'k_var_sku_per_cheque_15d', 'food_share_15d', 'k_var_disc_per_cheque_15d', 'promo_share_15d', 'k_var_cheque_group_width_15d', 'mean_discount_depth_15d'], 'mean_discount_depth_15d': [], 'cheque_count_6m_g32': ['disc_sum_6m_g34'], 'cheque_count_6m_g40': ['disc_sum_6m_g34', 'cheque_count_6m_g32', 'cheque_count_12m_g46', 'cheque_count_6m_g33', 'sale_count_12m_g54', 'cheque_count_12m_g32', 'cheque_count_6m_g41', 'cheque_count_12m_g41'], 'cheque_count_12m_g41': ['cheque_count_12m_g33', 'cheque_count_6m_g41', 'cheque_count_12m_g42'], 'food_share_1m': ['disc_sum_6m_g34', 'cheque_count_6m_g40', 'cheque_count_6m_g46', 'cheque_count_6m_g32', 'cheque_count_6m_g41', 'crazy_purchases_cheque_count_12m', 'cheque_count_6m_g33'], 'cheque_count_12m_g33': ['cheque_count_6m_g33'], 'disc_sum_6m_g34': [], 'cheque_count_6m_g33': ['disc_sum_6m_g34', 'sale_sum_6m_g54'], 'cheque_count_12m_g42': ['cheque_count_12m_g25', 'disc_sum_6m_g34', 'cheque_count_12m_g33'], 'cheque_count_12m_g25': [], 'sale_count_12m_g32': [], 'stdev_days_between_visits_15d': [], 'cheque_count_12m_g32': [], 'sale_sum_12m_g26': [], 'cheque_count_6m_g41': [], 'sale_sum_6m_g54': [], 'food_share_15d': [], 'k_var_cheque_group_width_15d': [], 'cheque_count_12m_g46': [], 'perdelta_days_between_visits_15_30d': [], 'sale_count_12m_g54': [], 'promo_share_15d': [], 'cheque_count_6m_g46': [], 'cheque_count_12m_g56': [], 'k_var_days_between_visits_15d': [], 'crazy_purchases_cheque_count_12m': [], 'k_var_sku_per_cheque_15d': []}
            dataPath = './data/lenta_data_processed.csv' # lenta_data_processed, selected_features_data
            target_col = 'response_att'
            treat_col = 'group'

            cleaned_data, graph, feature_set = preprocess_data(Ancestors_Graph, dataPath, target_col, treat_col, datasetName = 'lenta')
            # print(f'UnionAncestorsGraph: {graph}')
            feature_set = list(feature_set)

            p_test_true = None
            feature_cols = [c for c in feature_set if c not in [target_col, treat_col]]

        else:
            print(f"错误: 未知的数据集名称 {datasetName}")
            return
        
        # 对处理后的数据进行训练集和测试集划分
        print("正在进行训练集和测试集划分...")
        # 按照 8:2 的比例划分
        df_train, df_test = train_test_split(
            cleaned_data, 
            test_size = 0.2, 
            random_state = 2026,
            stratify = cleaned_data[treat_col]  # 按照treat_col进行分层抽样
        )

        '''
        # 生成数据 (Train & Test)
        df_train, graph, _ = generate_complex_environment(n=3000, seed=1)
        df_test, _, p_test_true = generate_complex_environment(n=1000, seed=2)
        feature_cols = [c for c in df_train.columns if c not in ['T', 'Y', 'true_tau']]
        '''

        print(f"数据划分完成: Train {df_train.shape}, Test {df_test.shape}")
        print(f"Train Treatment Rate: {df_train[treat_col].mean():.3f}")
        print(f"Test  Treatment Rate: {df_test[treat_col].mean():.3f}")
    else:
        # 1. 生成全量数据 (比如 N=4000)
        df_full, graph, prob_t_full = generate_complex_environment(n=50000, seed=2026)

        # 2. 特征列定义
        treat_col, target_col = 'T', 'Y'
        feature_cols = [c for c in df_full.columns if c not in [treat_col, target_col, 'true_tau']]

        # 3. 执行分层切分 (Stratified Split)
        # 关键点：按照 'T' 进行分层，确保训练集和测试集的干预比例一致
        df_train, df_test = train_test_split(
            df_full, 
            test_size = 0.2, 
            random_state = 2026, 
            stratify = df_full[treat_col] 
        )

        # 4. 如果是合成数据，需要把上帝视角的真实 PS 也切分出来传给评估器
        # (注意：prob_t_full 是 numpy array，需对应切分)
        train_idx = df_train.index
        test_idx = df_test.index
        p_test_true = prob_t_full[test_idx] # 获取测试集对应的真实 PS

        print(f"数据划分完成: Train {df_train.shape}, Test {df_test.shape}")
        print(f"Train Treatment Rate: {df_train[treat_col].mean():.3f}")
        print(f"Test  Treatment Rate: {df_test[treat_col].mean():.3f}")

    all_results = []
    df_stg1_c_total, w_c_total, ps_c_total, c_total, s_prec_1, ps_model_c_total, df_stg1, w, ps, c_opt, ps_model = None, None, None, None, None, None, None, None, None, None, None
    reuseKwargs = (df_stg1_c_total, w_c_total, ps_c_total, c_total, s_prec_1, ps_model_c_total, df_stg1, w, ps, c_opt, ps_model)
    
    for name, model in base_models.items():
        print(f"Running Experiment for: {name} ...")
        # 传入 is_observational 参数
        res, reuseKwargs = run_ablation_study(model, name, df_train, df_test, graph, feature_cols, treat_col=treat_col, target_col=target_col, p_test_true=p_test_true, is_observational=is_observational, verbose=True, deconfound_method=deconfound_method, reuseKwargs=reuseKwargs)

        for method, metrics in res.items():
            # print(f"  Method: {method} | Metrics: {metrics}")
            row = {
                'Model': name,
                'Method': method,
                'AUUC': metrics.get('auuc'),
                'Qini': metrics.get('qini'),
                'Decile_1_Real_Uplift': metrics.get('decile_1_real_uplift'),
                'Decile_2_Real_Uplift': metrics.get('decile_2_real_uplift'),
                'Decile_3_Real_Uplift': metrics.get('decile_3_real_uplift'),
                'Decile_4_Real_Uplift': metrics.get('decile_4_real_uplift'),
                'Decile_5_Real_Uplift': metrics.get('decile_5_real_uplift'),
                'Decile_6_Real_Uplift': metrics.get('decile_6_real_uplift'),
                'Decile_7_Real_Uplift': metrics.get('decile_7_real_uplift'),
                'Decile_8_Real_Uplift': metrics.get('decile_8_real_uplift'),
                'Decile_9_Real_Uplift': metrics.get('decile_9_real_uplift'),
                'Decile_10_Real_Uplift': metrics.get('decile_10_real_uplift')
            }
            if not is_observational:
                row['MSE'] = metrics.get('MSE')
                row['MAE'] = metrics.get('MAE')
                row['Spearman'] = metrics.get('Spearman')
            all_results.append(row)

    print("\n" + "="*60)
    print("🏆 FINAL RESULTS SUMMARY")
    print("="*60)

    df_res = pd.DataFrame(all_results)
    pd.set_option('display.float_format', lambda x: '%.4f' % x)

    # 动态选择排序列
    sort_metric = 'MSE' if not is_observational else 'Qini'
    ascending = True if sort_metric in ['MSE', 'MAE'] else False # MSE/MAE 越小越好，Spearman/Qini 越大越好

    # 更加丰富的 Pivot Table
    if not is_observational:
        # 展示 MSE, MAE, Spearman
        pivot_df = df_res.pivot(index='Model', columns='Method', values=['MSE', 'MAE', 'Spearman'])
    else:
        pivot_df = df_res.pivot(index='Model', columns='Method', values=['Qini', 'AUUC'])

    print(f"\nDetailed Comparison (Pivot Table):")
    print(pivot_df)

    df_res.to_csv(f'./results/{deconfound_method}_batch_experiment_summary.csv', index=False)
    print(f"\n所有结果已保存至 ./results/{deconfound_method}_batch_experiment_summary.csv")

if __name__ == "__main__":
    # 如果是合成数据实验 (有 Ground Truth)，设为 False
    # 如果是真实业务数据 (无 Ground Truth)，设为 True -> 代码会自动估算 P-Score
    # main(is_observational=False, deconfound_method='IPW') # PSM IPW
    main(is_observational=True, deconfound_method='IPW') # PSM IPW