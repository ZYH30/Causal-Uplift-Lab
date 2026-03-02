import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils.validation import check_X_y, check_is_fitted, check_array
# 在 models/meta_learners.py 头部添加
from xgboost import XGBRegressor, XGBClassifier
# 如果想用 LightGBM: from lightgbm import LGBMRegressor, LGBMClassifier

from .baseClass import  BaseUpliftEstimator

class SLearner(BaseUpliftEstimator):
    def __init__(self, base_estimator=None):
        """
        base_estimator: 任意 sklearn 回归器 (如 XGBoost, LR). 默认为 XGBRegressor.
        """
        self.base_estimator = base_estimator if base_estimator is not None else XGBRegressor(
            n_estimators=100, 
            max_depth=10, 
            learning_rate=0.05, 
            verbosity=0
        )
        self.estimator_ = None

    def fit(self, X, y, T, sample_weight=None, p_scores=None):
        # 1. 检查输入数据
        X, y = check_X_y(X, y, accept_sparse=True)
        T = check_array(T, ensure_2d=False)
        
        # 2. 拼接特征: 将 T 作为最后一列拼接到 X 中 -> [X, T]
        # 注意：这里需要将 T 变形成 (N, 1)
        X_with_T = np.hstack((X, T.reshape(-1, 1)))
        
        # 3. 训练单一模型 mu(X, T)
        self.estimator_ = clone(self.base_estimator)
        self.estimator_.fit(X_with_T, y, sample_weight=sample_weight)
        
        return self

    def predict(self, X):
        check_is_fitted(self, 'estimator_')
        X = check_array(X, accept_sparse=True)
        
        # 4. 构建反事实数据
        # 构造全为 0 的 T 列 (对照组)
        T0 = np.zeros((X.shape[0], 1))
        # 构造全为 1 的 T 列 (处理组)
        T1 = np.ones((X.shape[0], 1))
        
        # 5. 拼接并预测
        X_0 = np.hstack((X, T0))
        X_1 = np.hstack((X, T1))
        
        mu_0 = self.estimator_.predict(X_0)
        mu_1 = self.estimator_.predict(X_1)
        
        # 6. 计算 CATE: mu(x, 1) - mu(x, 0)
        return mu_1 - mu_0

class TLearner(BaseUpliftEstimator):
    def __init__(self, estimator_t=None, estimator_c=None):
        """
        允许为处理组 (Treatment) 和对照组 (Control) 指定不同的模型。
        如果 estimator_c 为空，则默认使用 estimator_t 的克隆。
        """
        self.estimator_t = estimator_t if estimator_t is not None else XGBRegressor(
            n_estimators=100, 
            max_depth=10, 
            learning_rate=0.05, 
            verbosity=0
        )
        self.estimator_c = estimator_c if estimator_c is not None else clone(self.estimator_t)
        
        self.model_t_ = None
        self.model_c_ = None

    def fit(self, X, y, T, sample_weight=None, p_scores=None):
        X, y = check_X_y(X, y)
        T = check_array(T, ensure_2d=False)
        
        # 1. 根据 T 拆分数据
        # boolean mask
        mask_t = (T == 1)
        mask_c = (T == 0)
        
        X_t, y_t = X[mask_t], y[mask_t]
        X_c, y_c = X[mask_c], y[mask_c]
        
        # 切分权重
        w0 = sample_weight[mask_c] if sample_weight is not None else None
        w1 = sample_weight[mask_t] if sample_weight is not None else None

        # 2. 独立训练两个模型
        # 注意：实际工程中要检查 X_t 或 X_c 是否为空，这里省略以保持核心逻辑清晰
        self.model_t_ = clone(self.estimator_t).fit(X_t, y_t, sample_weight=w1)
        self.model_c_ = clone(self.estimator_c).fit(X_c, y_c, sample_weight=w0)
        
        return self

    def predict(self, X):
        check_is_fitted(self, ['model_t_', 'model_c_'])
        X = check_array(X)
        
        # 3. 分别预测并相减
        mu_1 = self.model_t_.predict(X)
        mu_0 = self.model_c_.predict(X)
        
        return mu_1 - mu_0

class XLearner(BaseUpliftEstimator):
    def __init__(self, outcome_learner=None, effect_learner=None, propensity_learner=None):
        """
        outcome_learner: 第一阶段用于拟合 Y 的基模型 (同 T-Learner)
        effect_learner: 第二阶段用于拟合 D (伪效应) 的模型
        # propensity_learner: 用于估计倾向性得分 e(x) 的分类器
        """
        self.outcome_learner = outcome_learner if outcome_learner is not None else XGBRegressor(
            n_estimators=100, max_depth=10, learning_rate=0.05, verbosity=0
        )
        self.effect_learner = effect_learner if effect_learner is not None else XGBRegressor(
            n_estimators=100, max_depth=10, learning_rate=0.05, verbosity=0
        )
        self.propensity_learner = propensity_learner if propensity_learner is not None else XGBClassifier(
            n_estimators=100, 
            max_depth=10, 
            learning_rate=0.05, 
            use_label_encoder=False, 
            eval_metric='logloss', # 防止警告
            verbosity=0
        )
        
        # 内部存储的模型
        self.model_mu_t_ = None
        self.model_mu_c_ = None
        self.model_tau_t_ = None
        self.model_tau_c_ = None
        self.model_e_ = None

    def fit(self, X, y, T, sample_weight=None, p_scores=None):
        X, y = check_X_y(X, y)
        T = check_array(T, ensure_2d=False)
        
        # --- Stage 1: 训练基模型 mu_0, mu_1 (同 T-Learner) ---
        mask_t = (T == 1)
        mask_c = (T == 0)
        X_t, y_t = X[mask_t], y[mask_t]
        X_c, y_c = X[mask_c], y[mask_c]
        
        w1 = sample_weight[T==1] if sample_weight is not None else None
        w0 = sample_weight[T==0] if sample_weight is not None else None
        
        self.model_mu_t_ = clone(self.outcome_learner).fit(X_t, y_t, sample_weight=w1)
        self.model_mu_c_ = clone(self.outcome_learner).fit(X_c, y_c, sample_weight=w0)
        
        # --- Stage 2: 计算伪效应 (Imputed Treatment Effects) ---
        # D_1: 对于处理组，原本的 Y 减去 "如果他是对照组会怎样(mu_c预测)"
        # D_1 = Y_1 - mu_0(X_1)
        imputed_treatment_effects_t = y_t - self.model_mu_c_.predict(X_t)
        
        # D_0: 对于对照组，"如果他是处理组会怎样(mu_t预测)" 减去 原本的 Y
        # D_0 = mu_1(X_0) - Y_0
        imputed_treatment_effects_c = self.model_mu_t_.predict(X_c) - y_c
        
        # --- Stage 3: 训练效应模型 tau_0, tau_1 ---
        # 注意：model_tau_t 使用的是 处理组数据 拟合 D_1
        self.model_tau_t_ = clone(self.effect_learner).fit(X_t, imputed_treatment_effects_t, sample_weight=w1)
        # 注意：model_tau_c 使用的是 对照组数据 拟合 D_0
        self.model_tau_c_ = clone(self.effect_learner).fit(X_c, imputed_treatment_effects_c, sample_weight=w0)
        
        # 如果没传，作为兜底，我们自己训练一个
        if p_scores is None:
            self.model_e_ = clone(self.propensity_learner).fit(X, T)
        else:
            self.model_e_ = None

        return self

    def predict(self, X, p_scores = None):
        check_is_fitted(self, ['model_tau_t_', 'model_tau_c_'])
        X = check_array(X)
        
        # 1. 预测两个潜在的 uplift
        tau_1 = self.model_tau_t_.predict(X)
        tau_0 = self.model_tau_c_.predict(X)
        
        if p_scores is not None:
            # [优先] 使用外部传入的精准得分 (来自 OCU Stage 1)
            g_x = p_scores
        elif self.model_e_ is not None:
            # [兜底] 使用内部训练的模型
            g_x = self.model_e_.predict_proba(X)[:, 1]
        else:
            # 既没有传 p_scores，fit 时也没训练 model_e_ (理论上不应发生，除非用法错误)
            # 为了稳健，如果真走到这一步，尝试用 tau 的平均值或者报错
            # 这里抛出明确错误
            raise ValueError("XLearner 需要倾向性得分进行预测，但既未传入 p_scores 也未在 fit 阶段训练内部模型。")
        
        # 3. 加权融合 (Kunzel et al. 公式)
        # 如果 g(x) 接近 1 (像处理组)，则 (1-g(x)) 小，主要用 g(x)*tau_0 (由对照组模型预测的)
        # 这里的逻辑是：如果某区域全是处理组，那么 tau_1 (基于处理组数据) 很准，但我们需要反事实。
        # 标准 X-Learner 加权公式： tau = g(x)*tau_0 + (1-g(x))*tau_1
        return g_x * tau_0 + (1 - g_x) * tau_1



        
