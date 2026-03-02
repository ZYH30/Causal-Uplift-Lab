import numpy as np
from sklearn.base import clone
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

# 复用父类
from .baseClass import  BaseUpliftEstimator

class DML_RLearner(BaseUpliftEstimator):
    def __init__(self, 
                 model_y=None, 
                 model_t=None, 
                 model_final=None, 
                 n_splits=5, 
                 random_state=42,
                 discrete_treatment=False):
        """
        Args:
            model_y: 预测 Y 的模型 (Nuisance Model 1)
            model_t: 预测 T 的模型 (Nuisance Model 2). 
                     如果是连续变量，用 Regressor; 如果是二元变量，内部会处理.
            model_final: 最终估计 CATE 的模型 (Effect Model)
            n_splits: Cross-Fitting 的折数 (K-Fold)
            discrete_treatment: T 是否为离散分类变量 (二元). 
                                True -> 使用 predict_proba; False -> 使用 predict
        """
        self.model_y = model_y if model_y is not None else RandomForestRegressor(max_depth=5)
        self.model_t = model_t if model_t is not None else RandomForestRegressor(max_depth=5)
        self.model_final = model_final if model_final is not None else LinearRegression() # Lasso() 常用
        
        self.n_splits = n_splits
        self.random_state = random_state
        self.discrete_treatment = discrete_treatment
        
        self.final_estimator_ = None

    def fit(self, X, y, T):
        X, y = check_X_y(X, y)
        T = check_array(T, ensure_2d=False)
        
        # 准备存储全量的残差
        # y_res = Y - E[Y|X]
        # t_res = T - E[T|X]
        y_res = np.zeros_like(y, dtype=float)
        t_res = np.zeros_like(T, dtype=float)
        
        # --- 核心掌控点 1: 手动实现 Cross-Fitting (K-Fold) ---
        # 避免同一数据既用来训练 Nuisance Model 又用来预测残差 (Overfitting Bias)
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        
        for train_idx, test_idx in kf.split(X):
            # 切分数据
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            T_train, T_test = T[train_idx], T[test_idx]
            
            # 1. 训练 model_y (Outcome Model)
            est_y = clone(self.model_y).fit(X_train, y_train)
            y_pred = est_y.predict(X_test)
            y_res[test_idx] = y_test - y_pred
            
            # 2. 训练 model_t (Treatment Model)
            est_t = clone(self.model_t).fit(X_train, T_train)
            
            if self.discrete_treatment:
                # 如果是二元变量，我们关注的是 P(T=1|X)
                # 这里的残差是 T - P(T=1|X)
                t_pred = est_t.predict_proba(X_test)[:, 1]
            else:
                # 如果是连续变量，直接预测数值 E[T|X]
                t_pred = est_t.predict(X_test)
                
            t_res[test_idx] = T_test - t_pred
            
        # --- 核心掌控点 2: 构造加权回归目标 (R-Learner Transformation) ---
        # 目标: min sum (y_res - tau(x) * t_res)^2
        # 变换为: min sum t_res^2 * (y_res/t_res - tau(x))^2
        
        # 这里的 sample_weight 对应 t_res^2
        weights = t_res ** 2
        
        # 这里的 target 对应 y_res / t_res
        # 注意数值稳定性: 如果 t_res 极小，除法会爆炸。
        # 实践技巧: 过滤掉 t_res 极小的样本，或者加一个 epsilon
        mask = np.abs(t_res) > 1e-2 # 简单的 Trimming
        
        X_final = X[mask]
        target_final = y_res[mask] / t_res[mask]
        weights_final = weights[mask]
        
        # 3. 训练最终的 CATE 模型
        # 这个模型直接学习 X -> Y_target 的映射，本质上就是在学习 CATE
        self.final_estimator_ = clone(self.model_final)
        
        # 检查 final_model 是否支持 sample_weight
        # 大多数 sklearn 模型 (LinearRegression, RF, XGB) 都支持
        try:
            self.final_estimator_.fit(X_final, target_final, sample_weight=weights_final)
        except TypeError:
            # 如果不支持权重 (如 KNN), 则不传权重 (效果会打折)
            print("Warning: model_final does not support sample_weight. Ignoring weights.")
            self.final_estimator_.fit(X_final, target_final)
            
        return self

    def predict(self, X):
        check_is_fitted(self, 'final_estimator_')
        X = check_array(X)
        # 最终模型直接输出 CATE
        return self.final_estimator_.predict(X)
