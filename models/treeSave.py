import numpy as np
from joblib import Parallel, delayed
import time

class UniversalUpliftTreeNode:
    def __init__(self, depth, is_leaf=False, value=None):
        self.depth = depth
        self.is_leaf = is_leaf
        # value 存储格式: 字典 {treatment_id: cate_value}
        self.value = value  
        self.feature_idx = None
        self.threshold = None
        self.left = None
        self.right = None

class UniversalUpliftTree:
    def __init__(self, max_depth=3, min_samples_leaf=10, min_samples_treatment=5):
        """
        全能型 Uplift Tree
        Args:
            max_depth: 树的最大深度
            min_samples_leaf: 叶子节点最小总样本数
            min_samples_treatment: 每个组别（对照组+所有干预组）在叶子节点所需的最小样本数
        """
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_treatment = min_samples_treatment
        
        self.root = None
        self.treatment_classes_ = None # 存储所有非0的 Treatment ID
        self.is_binary_ = False        # 标记是否为二元干预

    def fit(self, X, y, T):
        """
        训练入口
        """
        # 1. 识别 Treatment 类别 (假设 0 为对照组)
        unique_t = np.unique(T)
        # 排序确保输出顺序稳定
        self.treatment_classes_ = sorted([t for t in unique_t if t != 0])
        
        if len(self.treatment_classes_) == 0:
            raise ValueError("数据中只包含 T=0，无法进行 Uplift 建模。")
            
        # 2. 判断是否为二元模式
        self.is_binary_ = (len(self.treatment_classes_) == 1)
        
        # 3. 递归构建树
        self.root = self._grow_tree(X, y, T, depth=0)
        return self

    def predict(self, X):
        """
        预测入口: 自动适配输出形状
        """
        predictions = []
        for i in range(len(X)):
            cates_dict = self._predict_single(X[i], self.root)
            # 按 self.treatment_classes_ 的顺序提取结果
            pred_vec = [cates_dict.get(k, 0.0) for k in self.treatment_classes_]
            predictions.append(pred_vec)
        
        result = np.array(predictions)
        
        # --- 智能 Flatten 逻辑 ---
        # 如果是二元模式，将 (N, 1) 展平为 (N, )
        if self.is_binary_:
            return result.flatten()
        else:
            return result

    def _predict_single(self, x, node):
        if node.is_leaf:
            return node.value
        
        if x[node.feature_idx] <= node.threshold:
            return self._predict_single(x, node.left)
        else:
            return self._predict_single(x, node.right)

    def _calculate_cate_vector(self, y, T):
        """
        计算 CATE 向量。返回字典 {k: tau_k}
        """
        cates = {}
        
        # 获取对照组均值
        y_c = y[T == 0]
        if len(y_c) == 0: return None
        mu_c = np.mean(y_c)
        
        # 遍历每一个 Treatment 计算增益
        for k in self.treatment_classes_:
            y_k = y[T == k]
            if len(y_k) == 0: return None
            cates[k] = np.mean(y_k) - mu_c
            
        return cates

    def _check_constraints(self, T):
        """
        严格约束: 确保对照组和 *所有* Treatment 组样本量充足
        """
        # 检查对照组
        if np.sum(T == 0) < self.min_samples_treatment:
            return False
        
        # 检查所有干预组
        for k in self.treatment_classes_:
            if np.sum(T == k) < self.min_samples_treatment:
                return False
        return True

    def _find_best_split_lower(self, X, y, T):
        best_gain = -np.inf
        best_feature_idx = None
        best_threshold = None
        
        n_samples, n_features = X.shape
        
        for feat_idx in range(n_features):
            unique_values = np.unique(X[:, feat_idx])
            
            # 百分位加速策略
            if len(unique_values) > 100:
                 thresholds = np.percentile(unique_values, np.linspace(5, 95, 20))
            else:
                 thresholds = unique_values
            
            for threshold in thresholds:
                mask_left = X[:, feat_idx] <= threshold
                mask_right = ~mask_left
                
                # 1. 基础总样本量检查
                if (np.sum(mask_left) < self.min_samples_leaf) or \
                   (np.sum(mask_right) < self.min_samples_leaf):
                    continue
                
                y_L, T_L = y[mask_left], T[mask_left]
                y_R, T_R = y[mask_right], T[mask_right]
                
                # 2. 严格的组别分布检查 (Positivity)
                if not (self._check_constraints(T_L) and self._check_constraints(T_R)):
                    continue
                
                # 3. 计算 CATE 向量
                cates_L = self._calculate_cate_vector(y_L, T_L)
                cates_R = self._calculate_cate_vector(y_R, T_R)
                
                if cates_L is None or cates_R is None:
                    continue
                
                # 4. 计算通用 Gain (欧氏范数平方和)
                # Binary Case: vector长度为1，退化为 scalar^2
                # Multi Case:  vector长度为K，为 sum(scalar^2)
                sum_sq_L = sum([v**2 for v in cates_L.values()])
                sum_sq_R = sum([v**2 for v in cates_R.values()])
                
                gain = len(y_L) * sum_sq_L + len(y_R) * sum_sq_R
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature_idx = feat_idx
                    best_threshold = threshold
                    
        return best_feature_idx, best_threshold
    
    def _find_best_split(self, X, y, T):
        best_gain = -np.inf
        best_feature_idx = None
        best_threshold = None
        
        n_samples, n_features = X.shape
        
        # 0. 预处理：构建统计矩阵 (Pre-computation)
        # 我们需要快速获取每个组的 sum_y 和 count
        # 找出所有的 unique classes (包括 Control 0)
        all_classes = [0] + self.treatment_classes_
        n_classes = len(all_classes)
        
        # 映射 T 到 0..K 的索引，以便通过数组索引访问
        # 假设 T 的值就是 0, 1, 2... 如果不是，需要一个 mapper，但在你的 synthetic data 里是的。
        # 为了通用性，建立一个 mapper
        class_to_idx = {t: i for i, t in enumerate(all_classes)}
        
        # 构造 One-Hot 风格的矩阵
        # y_mat[i, k] = y[i] if T[i] == class_k else 0
        # n_mat[i, k] = 1 if T[i] == class_k else 0
        y_mat = np.zeros((n_samples, n_classes))
        n_mat = np.zeros((n_samples, n_classes))
        
        # 填充矩阵 (这步可以用 numpy 高级索引优化，但即使循环也很快，因为只是一次线性扫描)
        for i in range(n_samples):
            if T[i] in class_to_idx:
                col_idx = class_to_idx[T[i]]
                y_mat[i, col_idx] = y[i]
                n_mat[i, col_idx] = 1
        
        # 计算全局总和 (用于计算 Right Node)
        G_y = y_mat.sum(axis=0) # Shape: (n_classes, )
        G_n = n_mat.sum(axis=0) # Shape: (n_classes, )
        
        # --- 特征随机采样 (兼容 RandomizedUpliftTree) ---
        # 检查是否定义了 max_features (UpliftForest 会用到)
        if hasattr(self, 'max_features'):
             # ... 复用你原有的 max_features 逻辑 ...
             if self.max_features == 'sqrt':
                 n_select = int(np.sqrt(n_features))
             elif self.max_features == 'log2':
                 n_select = int(np.log2(n_features))
             elif isinstance(self.max_features, float):
                 n_select = int(self.max_features * n_features)
             else:
                 n_select = n_features
             n_select = max(1, min(n_select, n_features))
             feature_indices = np.random.choice(n_features, n_select, replace=False)
        else:
             feature_indices = range(n_features)
        # -----------------------------------------------

        # 遍历选中的特征
        for feat_idx in feature_indices:
            # 1. 获取排序索引
            X_col = X[:, feat_idx]
            sorted_idx = np.argsort(X_col)
            
            X_sorted = X_col[sorted_idx]
            y_mat_sorted = y_mat[sorted_idx]
            n_mat_sorted = n_mat[sorted_idx]
            
            # 2. 向量化计算左侧累积和 (Left Node Stats)
            # cumsum 后，第 i 行代表前 i+1 个样本归入左节点时的 sum_y 和 count
            L_y = np.cumsum(y_mat_sorted, axis=0) # Shape: (N, n_classes)
            L_n = np.cumsum(n_mat_sorted, axis=0) # Shape: (N, n_classes)
            
            # 3. 计算右侧统计量 (Right Node Stats)
            # R = Total - L
            R_y = G_y - L_y
            R_n = G_n - L_n
            
            # 4. 有效性检查 (Positivity Constraints)
            # 必须所有组样本量 >= min_samples_treatment
            # 且总样本量 >= min_samples_leaf (其实上面隐含了，但为了保险可以显式加)
            
            # 使用 numpy 广播检查每一行是否满足条件
            # min_samples_treatment 约束: 所有 class 的计数都要达标
            valid_mask = (L_n >= self.min_samples_treatment).all(axis=1) & \
                         (R_n >= self.min_samples_treatment).all(axis=1)
            
            # 过滤掉不满足约束的切分点
            if not np.any(valid_mask):
                continue
                
            # 5. 寻找合法的切分位置 (Thresholds)
            # 我们只能在特征值发生变化的地方切分
            # X_sorted[i] < X_sorted[i+1]
            unique_mask = np.concatenate([np.diff(X_sorted) != 0, [False]])
            
            # 最终的 mask 是: 满足样本约束 AND 是特征值跳变点
            final_mask = valid_mask & unique_mask
            
            if not np.any(final_mask):
                continue
            
            # 6. 向量化计算 Gain
            # 只计算 final_mask 为 True 的行
            # 避免除以 0 (加上极小值 1e-9)
            
            # 提取有效行
            L_y_valid = L_y[final_mask]
            L_n_valid = L_n[final_mask]
            R_y_valid = R_y[final_mask]
            R_n_valid = R_n[final_mask]
            
            # 计算均值 Mu (Mean Response)
            mu_L = L_y_valid / (L_n_valid + 1e-9)
            mu_R = R_y_valid / (R_n_valid + 1e-9)
            
            # 计算 CATE (Treatment - Control)
            # 假设 0 号位置是 Control (我们在开头构建 all_classes 时把 0 放在了第一个)
            # shape: (n_valid_splits, n_treatments)
            tau_L = mu_L[:, 1:] - mu_L[:, 0:1] 
            tau_R = mu_R[:, 1:] - mu_R[:, 0:1]
            
            # 计算 Gain (Euclidean Distance / Heterogeneity)
            # Gain = N_L * sum(tau_L^2) + N_R * sum(tau_R^2)
            # 注意: L_n_valid 的求和是左节点的总样本数 (axis=1 求和得到总数)
            size_L = L_n_valid.sum(axis=1)
            size_R = R_n_valid.sum(axis=1)
            
            score_L = np.sum(tau_L**2, axis=1) * size_L
            score_R = np.sum(tau_R**2, axis=1) * size_R
            
            gain = score_L + score_R
            
            # 7. 找当前特征的最佳切分点
            current_best_idx = np.argmax(gain)
            current_max_gain = gain[current_best_idx]
            
            if current_max_gain > best_gain:
                best_gain = current_max_gain
                best_feature_idx = feat_idx
                
                # 还原回原始数据中的 threshold
                # gain 数组对应的索引是在 final_mask 为 True 的子集里的索引
                # 我们需要找到它在 X_sorted 中的真实位置
                valid_indices = np.where(final_mask)[0]
                real_split_idx = valid_indices[current_best_idx]
                
                # 阈值取当前点和下一点的中位数，或者直接取当前点
                best_threshold = X_sorted[real_split_idx]

        return best_feature_idx, best_threshold

    def _grow_tree(self, X, y, T, depth):
        # 计算当前节点值
        node_cates = self._calculate_cate_vector(y, T)
        
        # 如果无法计算（例如根节点就缺数据，或递归到底），兜底为0
        if node_cates is None:
            node_cates = {k: 0.0 for k in self.treatment_classes_}

        # 终止条件
        if (depth >= self.max_depth) or (len(y) < self.min_samples_leaf):
            return UniversalUpliftTreeNode(depth=depth, is_leaf=True, value=node_cates)
        
        # 寻找分裂
        feat_idx, threshold = self._find_best_split(X, y, T)
        
        if feat_idx is None:
            return UniversalUpliftTreeNode(depth=depth, is_leaf=True, value=node_cates)
        
        # 执行分裂
        mask_left = X[:, feat_idx] <= threshold
        mask_right = ~mask_left
        
        node = UniversalUpliftTreeNode(depth=depth, is_leaf=False, value=node_cates)
        node.feature_idx = feat_idx
        node.threshold = threshold
        
        node.left = self._grow_tree(X[mask_left], y[mask_left], T[mask_left], depth + 1)
        node.right = self._grow_tree(X[mask_right], y[mask_right], T[mask_right], depth + 1)
        
        return node
    
    def print_tree(self, node=None, spacing=""):
        """可视化树结构"""
        if node is None: node = self.root
        
        # 格式化输出 Value (适配二元和多元显示)
        val_str = ", ".join([f"T{k}={v:.2f}" for k, v in node.value.items()])
        
        if node.is_leaf:
            print(f"{spacing}☘️ Leaf: Uplift [{val_str}]")
            return
        
        print(f"{spacing}🌱 If Feat_{node.feature_idx} <= {node.threshold:.2f} (Avg: [{val_str}])")
        print(f"{spacing}--> True:")
        self.print_tree(node.left, spacing + "  |")
        print(f"{spacing}--> False:")
        self.print_tree(node.right, spacing + "  |")

class RandomizedUpliftTree(UniversalUpliftTree):
    """
    支持特征随机采样的 Uplift Tree，专为 Forest 设计
    """
    def __init__(self, max_features='sqrt', **kwargs):
        super().__init__(**kwargs)
        self.max_features = max_features

    def _find_best_split(self, X, y, T):
        best_gain = -np.inf
        best_feature_idx = None
        best_threshold = None
        
        n_samples, n_features = X.shape
        
        # --- 🆕 核心改动: 特征随机采样 ---
        if self.max_features == 'sqrt':
            n_select = int(np.sqrt(n_features))
        elif self.max_features == 'log2':
            n_select = int(np.log2(n_features))
        elif isinstance(self.max_features, float):
            n_select = int(self.max_features * n_features)
        else:
            n_select = n_features # None or 'all'
            
        # 确保至少选1个
        n_select = max(1, min(n_select, n_features))
        
        # 随机选择特征索引
        feature_indices = np.random.choice(n_features, n_select, replace=False)
        # -------------------------------
        
        # 只遍历被选中的特征
        for feat_idx in feature_indices:
            unique_values = np.unique(X[:, feat_idx])
            
            # 百分位加速
            if len(unique_values) > 100:
                 thresholds = np.percentile(unique_values, np.linspace(5, 95, 20))
            else:
                 thresholds = unique_values
            
            for threshold in thresholds:
                # ... (以下逻辑与原版完全一致) ...
                mask_left = X[:, feat_idx] <= threshold
                mask_right = ~mask_left
                
                if (np.sum(mask_left) < self.min_samples_leaf) or \
                   (np.sum(mask_right) < self.min_samples_leaf):
                    continue
                
                y_L, T_L = y[mask_left], T[mask_left]
                y_R, T_R = y[mask_right], T[mask_right]
                
                if not (self._check_constraints(T_L) and self._check_constraints(T_R)):
                    continue
                
                cates_L = self._calculate_cate_vector(y_L, T_L)
                cates_R = self._calculate_cate_vector(y_R, T_R)
                
                if cates_L is None or cates_R is None:
                    continue
                
                sum_sq_L = sum([v**2 for v in cates_L.values()])
                sum_sq_R = sum([v**2 for v in cates_R.values()])
                
                gain = len(y_L) * sum_sq_L + len(y_R) * sum_sq_R
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature_idx = feat_idx
                    best_threshold = threshold
                    
        return best_feature_idx, best_threshold

class UniversalUpliftForest:
    def __init__(self, 
                 n_estimators=100, 
                 max_depth=5, 
                 min_samples_leaf=10, 
                 min_samples_treatment=5, 
                 max_features='sqrt',
                 n_jobs=-1,
                 random_state=None):
        """
        全能型 Uplift Random Forest
        Args:
            n_estimators: 树的数量
            n_jobs: 并行 CPU 核数 (-1 代表使用所有核)
            ... 其他参数传给 Tree ...
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_treatment = min_samples_treatment
        self.max_features = max_features
        self.n_jobs = n_jobs
        self.random_state = random_state
        
        self.trees_ = []
        self.is_binary_ = False

    def fit(self, X, y, T):
        # 设置随机种子
        rng = np.random.default_rng(self.random_state)
        
        # 简单的预检查来确定是否为二元 (用于 predict 时的 flatten)
        unique_t = np.unique(T)
        treatment_classes = [t for t in unique_t if t != 0]
        self.is_binary_ = (len(treatment_classes) == 1)
        
        # 定义单棵树的训练函数 (用于并行)
        def _train_single_tree(seed):
            # 1. Bootstrap Sampling (Bagging)
            n_samples = X.shape[0]
            indices = rng.choice(n_samples, n_samples, replace=True)
            X_boot, y_boot, T_boot = X[indices], y[indices], T[indices]
            
            # 2. 初始化并训练树
            tree = RandomizedUpliftTree(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                min_samples_treatment=self.min_samples_treatment,
                max_features=self.max_features
            )
            tree.fit(X_boot, y_boot, T_boot)
            return tree
        
        # 并行训练
        # generate seeds for each tree
        seeds = rng.integers(0, 100000, size=self.n_estimators)
        
        self.trees_ = Parallel(n_jobs=self.n_jobs)(
            delayed(_train_single_tree)(seed) for seed in seeds
        )
        
        return self

    def predict(self, X):
        """
        聚合所有树的预测结果 (Averaging)
        """
        # 并行预测
        all_preds = Parallel(n_jobs=self.n_jobs)(
            delayed(tree.predict)(X) for tree in self.trees_
        )
        
        # all_preds 是一个 list, 包含了 n_estimators 个 array
        # 每个 array 的形状可能是 (N, ) [二元] 或 (N, K) [多元]
        
        # 转换为 numpy 数组: (n_estimators, N) 或 (n_estimators, N, K)
        all_preds = np.array(all_preds)
        
        # 计算均值 (Axis 0 是树的维度)
        avg_preds = np.mean(all_preds, axis=0)
        
        return avg_preds
