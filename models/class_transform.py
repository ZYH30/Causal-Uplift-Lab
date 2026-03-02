import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from xgboost import XGBRegressor, XGBClassifier
from .baseClass import BaseUpliftEstimator

class ClassTransformEstimator(BaseUpliftEstimator):
    def __init__(self, base_estimator=None, propensity_estimator=None, clip_propensity=0.05):
        """
        Args:
            base_estimator: 用于拟合变换后 Y* 的回归模型 (如 LinearRegression, XGBRegressor)
            propensity_estimator: 用于估计 e(x) 的分类模型 (如 LogisticRegression)
            clip_propensity: 截断阈值 (Control Point!). 
                             防止 e(x) 过于接近 0 或 1 导致数值爆炸.
                             范围通常在 [0.01, 0.1] 之间.
        """
        self.base_estimator = base_estimator if base_estimator is not None else XGBRegressor(
            n_estimators=100, max_depth=10, learning_rate=0.05, verbosity=0
        )
        self.propensity_estimator = propensity_estimator if propensity_estimator is not None else XGBClassifier(
            n_estimators=100, 
            max_depth=10, 
            learning_rate=0.05, 
            use_label_encoder=False, 
            eval_metric='logloss', # 防止警告
            verbosity=0
        )
        self.clip_propensity = clip_propensity
        
        self.estimator_ = None
        self.propensity_model_ = None

    def fit(self, X, y, T, sample_weight=None, p_scores=None):
        """
        训练流程优化:
        1. 优先使用传入的 p_scores (来自 OCU Stage 1)
        2. 如果没传，才自己训练 propensity_model
        3. 计算 Y* (Class Transformation)
        4. 训练 Base Estimator 拟合 Y* (支持 sample_weight)
        """
        X, y = check_X_y(X, y)
        T = check_array(T, ensure_2d=False)
        
        # --- Step 1: 获取倾向性得分 e(x) ---
        if p_scores is not None:
            # [核心优化] 直接使用传入的高质量 P-Score (Stage 1)
            # 避免了在 fit 内部重新训练导致的收敛问题和偏差
            e_x = p_scores
            self.propensity_model_ = None # 标记为外部依赖
        else:
            print("No p_scores provided, training propensity model internally.")
            # Fallback: 自己训练
            self.propensity_model_ = clone(self.propensity_estimator)
            # 尝试传入 sample_weight (如果模型支持)
            try:
                self.propensity_model_.fit(X, T, sample_weight=sample_weight)
            except TypeError:
                self.propensity_model_.fit(X, T)
            
            e_x = self.propensity_model_.predict_proba(X)[:, 1]
        
        # --- Step 2: 手动截断 (Clipping) - 核心掌控点 ---
        # 如果 e_x 接近 0，1/e_x 会爆炸；如果 e_x 接近 1，1/(1-e_x) 会爆炸
        # 我们强制将 e_x 限制在 [delta, 1-delta] 之间
        e_x_clipped = np.clip(e_x, self.clip_propensity, 1 - self.clip_propensity)
        
        # --- Step 3: 变换目标变量 Y* ---
        # 公式: Y* = Y * (T - e(x)) / (e(x) * (1 - e(x)))
        # 使用截断后的 e_x_clipped 进行计算
        numerator = T - e_x_clipped
        denominator = e_x_clipped * (1 - e_x_clipped)
        
        # 计算权重 w
        weights = numerator / denominator

        # [新增] 权重归一化，防止数值爆炸
        weights = weights / np.mean(np.abs(weights)) 

        # 得到变换后的目标
        y_transformed = y * weights
        
        # --- Step 4: 拟合回归模型 ---
        self.estimator_ = clone(self.base_estimator)
        # [关键修正] 
        # ClassTransform 自身已经包含了去偏逻辑 (Z-Transformation)。
        # 再次应用 IPW sample_weight 会导致 "双重逆概率加权" (Double Inverse Probability Weighting)，
        # 这会导致方差爆炸 (1/e^2)，因此这里必须忽略 IPW 权重。
        # 除非 sample_weight 是外部的抽样权重 (Sampling Weights)，否则不应传入。
        # 在 OCU 框架下，weights 是 IPW 权重，所以这里强制不传。
        
        self.estimator_.fit(X, y_transformed) 
        # 注意：这里去掉了 sample_weight=sample_weight
        
        return self

    def predict(self, X):
        """
        预测 CATE
        因为 E[Y*|X] = tau(X)，所以直接预测即可
        """
        check_is_fitted(self, 'estimator_')
        X = check_array(X)
        return self.estimator_.predict(X)
