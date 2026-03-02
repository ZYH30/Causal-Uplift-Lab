import optuna
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import make_scorer, mean_squared_error, log_loss
from sklearn.base import clone

# 支持的模型类 (需用户环境中安装了 xgboost/lightgbm)
try:
    from xgboost import XGBRegressor, XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMRegressor, LGBMClassifier
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

class BaseLearnerTuner:
    """
    通用基模型调优器
    用于为 Uplift 模型中的 outcome_learner 或 propensity_learner 寻找最佳超参数
    """
    def __init__(self, model_class, task='regression', n_trials=50, cv=3, random_state=42):
        """
        Args:
            model_class: 模型类 (e.g., XGBRegressor, RandomForestRegressor)
            task: 'regression' 或 'classification'
            n_trials: Optuna 尝试次数
            cv: 交叉验证折数
        """
        self.model_class = model_class
        self.task = task
        self.n_trials = n_trials
        self.cv = cv
        self.random_state = random_state
        self.best_params_ = None
        self.best_score_ = None
        self.best_estimator_ = None

    def _get_search_space(self, trial):
        """
        定义常用模型的搜索空间
        """
        model_name = self.model_class.__name__
        
        # --- XGBoost ---
        if 'XGB' in model_name:
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'n_jobs': -1,
                'random_state': self.random_state,
                'verbosity': 0
            }
        
        # --- LightGBM ---
        elif 'LGBM' in model_name:
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'n_jobs': -1,
                'random_state': self.random_state,
                'verbose': -1
            }

        # --- RandomForest ---
        elif 'RandomForest' in model_name:
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_float('max_features', 0.1, 1.0),
                'n_jobs': -1,
                'random_state': self.random_state
            }
        
        else:
            raise ValueError(f"暂不支持自动调优该模型: {model_name}. 请手动传入 params.")

    def tune(self, X, y):
        """执行调优"""
        def objective(trial):
            params = self._get_search_space(trial)
            model = self.model_class(**params)
            
            # 定义评分标准
            if self.task == 'regression':
                # 负均方误差 (越大越好)
                scoring = 'neg_mean_squared_error'
            else:
                # 负 LogLoss (越大越好)
                scoring = 'neg_log_loss'
                
            scores = cross_val_score(model, X, y, cv=self.cv, scoring=scoring, n_jobs=-1)
            return scores.mean()

        # 创建 study (方向是 maximize, 因为 sklearn scoring 返回的是负值)
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)
        
        self.best_params_ = study.best_params
        self.best_score_ = study.best_value
        
        # 使用最佳参数实例化并训练最终模型
        self.best_estimator_ = self.model_class(**self.best_params_)
        self.best_estimator_.fit(X, y)
        
        print(f"[{self.model_class.__name__}] 调优完成. Best CV Score: {self.best_score_:.4f}")
        return self.best_estimator_

# 假设这是您的业务数据
# X_train, y_train, T_train = load_data()
X_train, y_train, T_train = 1, 1, 1

# --- 场景演示：优化 DML 模型 ---
# DML 需要两个核心基模型：
# 1. model_y: 预测 E[Y|X] (回归)
# 2. model_t: 预测 E[T|X] (如果是连续定价，回归；如果是发券，分类)

# 1. 调优 Outcome Model (Y ~ X)
print(">>> Tuning Outcome Model (Y ~ X)...")
tuner_y = BaseLearnerTuner(model_class=RandomForestRegressor, task='regression', n_trials=20)
best_model_y = tuner_y.tune(X_train, y_train)

# 2. 调优 Treatment Model (T ~ X)
# 假设 T 是连续的折扣金额
print("\n>>> Tuning Treatment Model (T ~ X)...")
tuner_t = BaseLearnerTuner(model_class=RandomForestRegressor, task='regression', n_trials=20)
best_model_t = tuner_t.tune(X_train, T_train)

# 3. 调优 Final Effect Model (Res_Y ~ Res_T)
# 这一步比较难直接 tune，因为没有 ground truth label。
# 通常做法是：使用更简单的线性模型（Lasso）来保持鲁棒性，或者复用 best_model_y 的超参数结构。
# 这里我们复用 RandomForest，但使用默认或轻量级参数
best_model_final = RandomForestRegressor(max_depth=5, min_samples_leaf=10) 

# 4. 组装 DML
# 此时传入最佳基模型了
from dml import DML_RLearner # 假设上面的代码保存在 phase3_code.py

dml_optimized = DML_RLearner(
    model_y=best_model_y,      # 注入调优后的模型
    model_t=best_model_t,      # 注入调优后的模型
    model_final=best_model_final,
    discrete_treatment=False,
    n_splits=5
)

dml_optimized.fit(X_train, y_train, T_train)
print("\n>>> DML Model Fitted with Optimized Base Learners!")