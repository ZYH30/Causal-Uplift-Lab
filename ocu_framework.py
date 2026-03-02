import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import warnings

from copy import deepcopy

# 引入您提供的模块
# 假设这些文件在同一目录下，或者在 Python 路径中
try:
    from getProxConAndRegVars import find_confounder_sets, get_parents, get_ancestors, get_descendants, invert_graph
    from lgb_models import lgb_optuna
except ImportError as e:
    print(f"警告: 依赖模块导入失败，请确保 getProxConAndRegVars.py 和 lgb_models.py 在路径中. Error: {e}")

class FirstStageDeconfounder:
    def __init__(self, treatment_var='T', target_var='Y', ancestors_graph=None, method='IPW', conf_type = 'proximal', is_precise = True):
        """
        第一阶段：去混杂处理器

        参数:
        - method: 'IPW' (稳定权重) 或 'PSM' (倾向分匹配)
        - treatment_var: 策略变量名 (必须为二元 0/1)
        - target_var: 目标变量名
        - ancestors_graph: 祖先图字典 (键为子变量, 值为父变量列表)
        """
        assert method in ['IPW', 'PSM'], "method 必须是 'IPW' 或 'PSM'"
        self.method = method
        self.t_var = treatment_var
        self.y_var = target_var
        self.graph = ancestors_graph
        self.conf_type = conf_type
        self.is_precise = is_precise
        
        self.c_opt = []   # 最优混杂集
        self.s_prec = []  # 精度变量集
        self.propensity_model = None
        self.best_params = None
        
    def _extract_causal_features(self):
        """
        基于因果图提取 C_opt 和 S_Prec
        """
        if self.graph is None:
            raise ValueError("必须提供因果图 (ancestors_graph)")

        print(f"\n[Causal Feature Selection] 正在基于因果图提取特征...")

        # 1. 提取最优混杂集 (C_opt / Proximal Set)
        # 使用提供的 find_confounder_sets 函数
        # 注意：我们需要的是 T 和 Y 之间的混杂
        conf_sets = find_confounder_sets(self.graph, self.t_var, self.y_var)
        self.c_opt = conf_sets[self.conf_type]
        print(f"  -> 最优混杂集 (C_opt): {self.c_opt}")

        # 2. 提取精度变量集 (S_Prec)
        if self.is_precise:
            # 定义: S_Prec = Parents(Y) - Descendants(T) - {T} - C_opt = ** Adj(Y) | C_opt ** # 在 控制 C_opt 后，Y 的剩余直接父变量（排除中介变量）
            # 即直接影响 Y，但不影响 T (非混杂)，且不是 T 本身
            
            parents_y = get_parents(self.graph, self.y_var)
            candidates = set(parents_y)
            # print(f"  -> 直接父变量: {parents_y}")
            
            graph_invert = invert_graph(self.graph)
            descendants_t = get_descendants(graph_invert, self.t_var)
            exclude_set = set(descendants_t) | {self.t_var} | set(self.c_opt)
            
            # print(f"  -> descendants_t: {descendants_t}")
            # print(f"  -> exclude_set: {exclude_set}")
            # 集合运算
            self.s_prec = list(candidates - exclude_set)
            
            '''
            ancestors_t = get_ancestors(self.graph, self.t_var)
            # 集合运算
            exclude_set = set(ancestors_t) | {self.t_var} | set(self.c_opt)
            self.s_prec = list(candidates - exclude_set)
            '''
            
            # 排序以保证确定性
            self.s_prec.sort()
        else:
            self.s_prec = []
            
        print(f"  -> 精度变量集 (S_Prec): {self.s_prec}")
        
        if not self.c_opt:
            warnings.warn("警告: 未检测到最优混杂变量，倾向分模型将无法有效去偏！")

    def fit_transform(self, df, n_trials=32, random_state=42):
        """
        执行去混杂流程
        
        参数:
        - df: 输入数据框
        - n_trials: Optuna 优化次数
        
        返回:
        - df_out: 处理后的数据 (IPW模式为原数据，PSM模式为匹配后数据)
        - weights: 样本权重 (IPW模式为计算出的权重，PSM模式为 None)
        - c_opt: 最优混杂列表
        - s_prec: 精度变量列表
        """
        # 1. 特征提取
        self._extract_causal_features()
        
        # 数据完整性检查
        required_cols = self.c_opt + [self.t_var, self.y_var] + self.s_prec
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"输入数据缺失列: {missing}")

        X_conf = df[self.c_opt].copy()
        T = df[self.t_var].values
        
        # 2. 训练倾向性得分模型 P(T|C_opt)
        print(f"\n[Propensity Modeling] 开始训练倾向性得分模型 (Method: LightGBM + Optuna)...")
        
        # 调用 lgb_models.py 中的 lgb_optuna
        # y_type='discrete' 因为 T 是二元的
        self.propensity_model, self.best_params, _, _ = lgb_optuna(
            X=X_conf, 
            y=T, 
            y_type='discrete', 
            is_plot=False, 
            is_optM=(n_trials > 0), 
            n_trials=n_trials,
            random_state=random_state
        )
        
        # 预测倾向性得分 (lgb.train 返回的是 Booster, predict 直接返回正类概率)
        ps_score = self.propensity_model.predict(X_conf)
        
        # 截断以满足 Positivity (Overlap)
        ps_score = np.clip(ps_score, 0.05, 0.95)
        
        # 3. 根据方法执行去混杂 (获取 4 个基础返回值)
        if self.method == 'IPW':
            # 解包 _apply_ipw 返回的 4 个元素
            df_out, weights, c_out, s_out = self._apply_ipw(df, T, ps_score)
        elif self.method == 'PSM':
            # 解包 _apply_psm 返回的 4 个元素
            df_out, weights, c_out, s_out = self._apply_psm(df, T, ps_score)
        
        # 4. [修改] 拼接 propensity_model 并返回扁平化的 6 个元素
        # 这样调用方就可以使用: df, w, ps, c, s, model = ... 进行解包了
        return df_out, weights, ps_score, c_out, s_out, self.propensity_model
        
    def _apply_ipw(self, df, T, ps_score):
        print("\n[De-confounding] 计算稳定逆倾向得分权重 (Stabilized IPW)...")
        
        # 计算 P(T) - 边缘概率
        p_t1 = np.mean(T)
        p_t0 = 1 - p_t1
        
        # 计算稳定权重
        # w = P(T=1)/e(x) if T=1
        # w = P(T=0)/(1-e(x)) if T=0
        weights = np.where(T == 1, p_t1 / ps_score, p_t0 / (1 - ps_score))
        
        # 在 ocu_framework.py 的 _apply_ipw 结尾处添加
        # 分组归一化逻辑
        weights[T == 1] *= (np.sum(T == 1) / np.sum(weights[T == 1]))
        weights[T == 0] *= (np.sum(T == 0) / np.sum(weights[T == 0]))

        # 2. [新增] 权重截尾 (Winsorizing) - 更加稳健的做法
        # 将权重的上下 1% 极端值，用 1% 分位点的值代替，防止个别离群点主导
        # lower = np.percentile(weights, 1)
        # upper = np.percentile(weights, 99)
        # weights = np.clip(weights, lower, upper)

        # 返回原始数据（未删减）和权重
        return df, weights, self.c_opt, self.s_prec

    def _apply_psm(self, df, T, ps_score):
        # 这个方法最后再优化
        print("\n[De-confounding] 执行倾向性得分匹配 (PSM - 1:1 Nearest Neighbor)...")
        
        df_matched = df.copy()
        df_matched['ps_score'] = ps_score
        # 使用 Logit 变换后的 PS 进行匹配 (线性化，效果更好)
        df_matched['ps_logit'] = np.log(ps_score / (1 - ps_score))
        
        # 分离处理组和对照组
        treated = df_matched[df_matched[self.t_var] == 1]
        control = df_matched[df_matched[self.t_var] == 0]
        
        if len(treated) == 0 or len(control) == 0:
            raise ValueError("处理组或对照组样本为空，无法进行匹配")

        # 寻找匹配
        # 我们通常为数量较少的一方寻找匹配，或者固定为 Treatment 找 Control
        # 这里假设为每个 Treatment 样本找一个 Control
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(control[['ps_logit']])
        distances, indices = nbrs.kneighbors(treated[['ps_logit']])
        
        # 获取匹配到的 Control 索引
        matched_control_indices = control.iloc[indices.flatten()].index
        
        # 构建匹配后的数据集 (Treated + Matched Control)
        # 注意：这里可能存在同一个 Control 匹配多个 Treated 的情况 (With Replacement)
        # 为了方便后续 Uplift 建模，我们可以选择不重复匹配 (Without Replacement) 或者保留重复
        # 这里采用简单的索引合并，如果 indices 有重复，Control 样本会重复出现 (这是允许的)
        
        matched_controls = control.iloc[indices.flatten()]
        final_df = pd.concat([treated, matched_controls], axis=0).reset_index(drop=True)
        
        print(f"  -> 原始样本数: {len(df)}, 匹配后样本数: {len(final_df)}")
        
        # PSM 不返回权重 (或者视权重为 1)
        return final_df, None, self.c_opt, self.s_prec

class SecondStageUpliftTrainer:
    def __init__(self, base_uplift_model, propensity_model, c_opt, s_prec, random_state=42):
        """
        第二阶段：Uplift 建模训练器 (OCU-Framework)
        
        参数:
        - base_uplift_model: 初始化的 Uplift 模型实例 (如 XLearner, UniversalUpliftForest)
                             必须实现 .fit(X, y, T) 和 .predict(X) 接口
        - random_state: 随机种子，用于重采样
        """
        self.base_model = base_uplift_model
        self.random_state = random_state

        # 构建第二阶段特征集 X_fit = C_opt U S_Prec
        # 理论依据: Henckel et al. (2019) - 包含精度变量可最小化 CATE 估计的方差
        self.x_fit_cols_ = list(set(c_opt + s_prec)) # 存储第二阶段使用的特征列名
        self.c_opt = c_opt
        self.propensity_model = propensity_model
        
    def fit(self, df, t_var, y_var, ps_score=None, weights=None, verbose=True):
        """
        训练入口
        
        参数:
        - df: 第一阶段处理后的数据框 (IPW模式为原数据，PSM模式为匹配后数据)
        - c_opt: 最优混杂变量列表 (来自第一阶段)
        - s_prec: 精度调整变量列表 (来自第一阶段)
        - t_var: 策略变量名
        - y_var: 目标变量名
        - weights: 样本权重 (IPW模式必传，PSM模式为 None)
        """

        # 排序以保证特征顺序一致性
        self.x_fit_cols_.sort()
        
        if verbose:
            print(f"\n[Stage 2 Modeling] 开始训练 Uplift 模型...")
            print(f"  -> 使用模型: {self.base_model.__class__.__name__}")
            print(f"  -> 建模特征集 (X_fit): {self.x_fit_cols_} (C_opt + S_Prec)")
        
        # 提取数据矩阵
        try:
            X_data = df[self.x_fit_cols_].values
            y_data = df[y_var].values
            T_data = df[t_var].values
        except KeyError as e:
            raise ValueError(f"数据框缺失必要的列: {e}")
        '''
        # 2. 根据去混杂策略分支处理
        if weights is not None:
            # === 分支 A: IPW 模式 (执行重采样) ===
            if verbose:
                print("  -> 检测到样本权重 (IPW Mode). 执行加权重采样 (Weighted Resampling)...")
            
            # 归一化权重以作为概率
            w_sum = np.sum(weights)
            if w_sum == 0:
                raise ValueError("样本权重和为0，无法重采样")
            p_probs = weights / w_sum
            
            # 执行重采样
            n_samples = len(df)
            rng = np.random.default_rng(self.random_state)
            
            # 关键：根据权重 p 进行有放回抽样
            indices = rng.choice(n_samples, size=n_samples, replace=True, p=p_probs)
            
            X_train = X_data[indices]
            y_train = y_data[indices]
            T_train = T_data[indices]
            
            if verbose:
                print(f"  -> 重采样完成. 样本分布已重构以反映因果权重.")
                
        else:
            # === 分支 B: PSM 模式 (直接训练) ===
            if verbose:
                print("  -> 未检测到样本权重 (PSM Mode). 直接基于匹配样本训练...")
            
            X_train = X_data
            y_train = y_data
            T_train = T_data
        '''
        # 统一处理：无论 IPW 还是 PSM，均直接使用提供的数据进行训练
        # 3. 训练模型
        # 由于数据已经处理（重采样或匹配），模型只需将其视为普通的 RCT 数据
        # self.base_model.fit(X_train, y_train, T_train)
        
        # 直接调用 base_model，传入权重和得分
        try:
            self.base_model.fit(
                X_data, 
                y_data, 
                T_data, 
                sample_weight=weights,  # IPW 权重用于加权训练
                p_scores=ps_score       # 倾向得分用于模型内部计算
            )
        except TypeError as e:
            # 捕获因 Base Model 未实现 sample_weight/p_scores 参数而引发的错误
            error_msg = (
                f"当前使用的 Uplift 模型 ({self.base_model.__class__.__name__}) "
                f"的 .fit() 方法不支持 sample_weight 或 p_scores 参数。\n"
                f"请确保所有 Base Model (S/T/X-Learner, ClassTransform) 都已更新接口签名。\n"
                f"原始错误: {e}"
            )
            raise TypeError(error_msg) from e
        
        if verbose:
            print(f"  -> 模型训练完成.\n")
            
        return self

    def predict(self, df, use_ps = False):
        """
        预测 CATE
        """
        
        # 确保预测时使用相同的特征集
        if not all(col in df.columns for col in self.x_fit_cols_):
            missing = [c for c in self.x_fit_cols_ if c not in df.columns]
            raise ValueError(f"预测数据缺失特征: {missing}")
            
        X_pred = df[self.x_fit_cols_].values

        test_p_scores = None
        if use_ps:
            X_prop = df[self.c_opt]
            
            # 使用 Stage 1 模型预测测试集倾向分
            # 注意：LightGBM predict 返回的是概率
            test_p_scores = self.propensity_model.predict(X_prop)
            # 保持一致的截断逻辑
            test_p_scores = np.clip(test_p_scores, 0.05, 0.95)
        
        # 3. 调用 Base Model 的 predict
        try:
            # 尝试传入 p_scores
            if test_p_scores is not None:
                return self.base_model.predict(X_pred, p_scores=test_p_scores)
            else:
                return self.base_model.predict(X_pred)
        except TypeError:
            # 如果模型 (如 S-Learner) 不接受 p_scores，则回退到普通预测
            return self.base_model.predict(X_pred)