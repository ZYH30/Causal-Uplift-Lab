from sklearn.base import BaseEstimator, RegressorMixin, clone

class BaseUpliftEstimator(BaseEstimator, RegressorMixin):
    """所有 Uplift 模型的基础父类，用于规范化接口"""
    def fit(self, X, y, T):
        """
        X: 特征矩阵
        y: 结果变量 (Outcome)
        T: 策略变量 (Treatment), 必须是二元 0/1
        """
        raise NotImplementedError
    
    def predict(self, X):
        """预测 CATE (Conditional Average Treatment Effect)"""
        raise NotImplementedError