"""
LightGBM模型优化拟合并返回残差
包含lgb_train和lgb_optuna函数
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import mean_squared_error, roc_auc_score, accuracy_score, log_loss
from sklearn.model_selection import train_test_split
import optuna
import logging
import utilseed

# 设置日志级别
optuna.logging.set_verbosity(optuna.logging.WARNING)

def lgb_train(X, y, y_type, num_boost_round=512, learning_rate=0.005, 
              max_depth=3, num_leaves=40, min_data_in_leaf=5, min_data_in_bin=5, 
              subsample=0.7, opt=False, is_plot=True, verbose=False, random_state=None,
              residual_type='default'):
    """
    LightGBM训练函数
    """
    utilseed.set_seed(random_state)

    n_feature = X.shape[1]
    
    if y_type == 'discrete':
        y_encoded = LabelEncoder().fit_transform(y)
        objective = 'multiclass' if len(set(y)) > 2 else 'binary'
        eval_metric = 'multi_logloss' if objective == 'multiclass' else 'binary_logloss'
        num_class = len(set(y)) if objective == 'multiclass' else 1
        
    elif y_type == 'continuous':
        y_encoded = y
        objective = 'regression'
        eval_metric = 'rmse'
        num_class = 1
        
    else:
        raise ValueError("Invalid y_type. Supported values are 'discrete' and 'continuous'.")

    # 固定内部数据划分的随机状态
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42
    )

    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test)
    
    params = {
        'learning_rate': learning_rate,
        'max_depth': max_depth,
        'num_leaves': num_leaves,
        'min_data_in_leaf': min_data_in_leaf,
        'min_data_in_bin': min_data_in_bin,
        'subsample': subsample,
        'random_state': random_state if random_state is not None else 2025,
        'deterministic': True,
        'force_row_wise': True
    }

    paramsTotal = {
        'objective': objective,
        'metric': eval_metric,
        'num_class': num_class,
        'verbosity': -1,
        **params
    }
    
    '''
    if opt:
        num_boost_round = num_boost_round
    else:
        num_boost_round = num_boost_round * 2
    '''
    
    model = lgb.train(paramsTotal, train_data, valid_sets=[test_data],
                      num_boost_round=num_boost_round,
                      callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=verbose)])

    # 预测和评估
    if opt:
        y_pred = model.predict(X_test)
        if objective == 'regression':
            loss = mean_squared_error(y_test, y_pred)
        elif objective == 'binary':
            loss = log_loss(y_test, y_pred, labels=[0, 1])
        else:
            loss = log_loss(y_test, y_pred, labels=list(range(num_class)))
        return loss
    else:
        total_pred = model.predict(X)
        
        def calculate_lgb_residual(y_true, predictions, y_type, objective_type, residual_type, num_classes=None):
            """为LightGBM计算不同类型的残差"""
            if y_type == 'continuous':
                return y_true - predictions
            
            elif y_type == 'discrete':
                if objective_type == 'binary':
                    if residual_type == 'prob' or residual_type == 'default':
                        return y_true - predictions
                    elif residual_type == 'logit':
                        epsilon = 1e-7
                        predictions = np.clip(predictions, epsilon, 1 - epsilon)
                        y_logit = np.log(y_true + epsilon) - np.log(1 - y_true + epsilon)
                        logit_pred = np.log(predictions) - np.log(1 - predictions)
                        return y_logit - logit_pred
                    elif residual_type == 'cross_entropy':
                        epsilon = 1e-7
                        predictions = np.clip(predictions, epsilon, 1 - epsilon)
                        return -(y_true * np.log(predictions) + (1 - y_true) * np.log(1 - predictions))
                        
                else:  # 多分类
                    if residual_type == 'weighted_prob' or residual_type == 'default':
                        y_onehot = np.zeros((len(y_true), num_classes))
                        y_onehot[np.arange(len(y_true)), y_true] = 1
                        prob_residual = predictions - y_onehot
                        return np.linalg.norm(prob_residual, axis=1)
                    elif residual_type == 'cross_entropy':
                        num_samples = len(y_true)
                        cross_entropy = np.zeros(num_samples)
                        for i in range(num_samples):
                            true_class = int(y_true[i])
                            if predictions.ndim == 1:
                                prob = predictions[i] if true_class == 1 else 1 - predictions[i]
                                cross_entropy[i] = -np.log(max(prob, 1e-7))
                            else:
                                prob = predictions[i, true_class]
                                cross_entropy[i] = -np.log(max(prob, 1e-7))
                        return cross_entropy
                    elif residual_type == 'normalized_cross_entropy':
                        cross_entropy = calculate_lgb_residual(y_true, predictions, y_type, objective_type, 'cross_entropy', num_classes)
                        max_entropy = -np.log(1.0/num_classes)
                        return cross_entropy / max_entropy
            return None

        num_classes = len(set(y_encoded)) if y_type == 'discrete' else None
        residual = calculate_lgb_residual(y_encoded, total_pred, y_type, objective, residual_type, num_classes)
        
        loss = mean_squared_error(y_encoded, total_pred) if objective == 'regression' else \
            roc_auc_score(y_encoded, total_pred) if objective == 'binary' else \
            accuracy_score(y_encoded, total_pred.argmax(axis=1))
            
        return model, 1, residual, loss

def lgb_optuna(X, y, y_type, is_plot=True, is_optM=True, n_trials=64, random_state=2025, residual_type='default'):
    """
    LightGBM的Optuna优化
    
    参数:
        X: 特征矩阵
        y: 目标变量
        y_type: 变量类型 ('continuous' 或 'discrete')
        is_plot: 是否绘图
        is_optM: 是否使用Optuna优化 (True: 使用optuna寻优, False: 使用固定参数)
        n_trials: optuna优化试验次数
        random_state: 随机种子
        residual_type: 残差类型
    
    返回:
        model: 训练好的模型
        best_params: 最佳参数 (或固定参数)
        residual: 残差
        loss: 损失值
    """
    
    utilseed.set_seed(random_state)
    
    if is_optM:
        # 使用Optuna进行参数优化
        def objectiveFun(trial):
            nn = X.shape[0]
            params = {'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.4),
                      'max_depth': trial.suggest_int('max_depth', 5, 25),
                      'num_leaves': trial.suggest_int('num_leaves', 40, 200),
                      'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 50, 200),
                      'min_data_in_bin': trial.suggest_int('min_data_in_bin', 50, 200),
                      'subsample': trial.suggest_float('subsample', 0.7, 0.9)
                     }
            
            loss = lgb_train(
                X, y, y_type, num_boost_round=256, learning_rate=params['learning_rate'],
                max_depth=params['max_depth'], num_leaves=params['num_leaves'], 
                min_data_in_leaf=params['min_data_in_leaf'], min_data_in_bin=params['min_data_in_bin'], 
                subsample=params['subsample'], opt=True, random_state=random_state,
                residual_type=residual_type
            )
            return loss

        study = optuna.create_study(direction='minimize')
        study.optimize(objectiveFun, n_trials=n_trials)
        best_params = study.best_params
        print(f"Optuna优化完成！最佳参数: {best_params}")
        print(f"最佳目标值: {study.best_value}")
        
    else:
        # 使用固定参数，跳过Optuna优化以节省时间
        best_params = {
            'learning_rate': 0.1,
            'max_depth': 10,
            'num_leaves': 200,
            'min_data_in_leaf': 200,
            'min_data_in_bin': 200,
            'subsample': 0.9
        }
        print(f"使用固定参数（跳过Optuna优化）: {best_params}")
    
    # 使用最佳参数（或固定参数）训练最终模型
    model, _, residual, loss = lgb_train(
        X, y, y_type, num_boost_round=256, opt=False, is_plot=is_plot, random_state=random_state,
            residual_type=residual_type, **best_params
    )

    return model, best_params, residual, loss