import numpy as np
from joblib import Parallel, delayed

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
        全能型 Uplift Tree (支持 IPW 加权)
        """
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_treatment = min_samples_treatment
        
        self.root = None
        self.treatment_classes_ = None 
        self.is_binary_ = False        

    def fit(self, X, y, T, sample_weight=None, p_scores=None):
        """
        [修改] 适配 OCU 框架：支持样本权重和倾向得分
        """
        # 1. 权重优先级处理：优先使用 sample_weight，其次根据 p_scores 生成
        if sample_weight is None:
            if p_scores is not None:
                # 理论公式: w = T/p + (1-T)/(1-p)
                sample_weight = np.where(T >= 1, 1.0 / p_scores, 1.0 / (1 - p_scores))
            else:
                sample_weight = np.ones_like(y, dtype=np.float64)
        
        # 2. 识别 Treatment 类别
        unique_t = np.unique(T)
        self.treatment_classes_ = sorted([t for t in unique_t if t != 0])
        
        if len(self.treatment_classes_) == 0:
            raise ValueError("数据中只包含 T=0，无法进行 Uplift 建模。")
            
        self.is_binary_ = (len(self.treatment_classes_) == 1)
        
        # 3. 递归构建加权树
        self.root = self._grow_tree(X, y, T, sample_weight, depth=0)
        return self

    def predict(self, X, p_scores=None):
        """
        [修改] 增加 p_scores 参数以适配 OCU 接口兼容性
        """
        predictions = []
        for i in range(len(X)):
            cates_dict = self._predict_single(X[i], self.root)
            pred_vec = [cates_dict.get(k, 0.0) for k in self.treatment_classes_]
            predictions.append(pred_vec)
        
        result = np.array(predictions)
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

    def _calculate_cate_vector(self, y, T, weights):
        """
        [修改] 计算加权 CATE 向量 (Phase 2)
        """
        cates = {}
        
        # 获取加权对照组均值
        mask_c = (T == 0)
        if not np.any(mask_c): return None
        mu_c = np.average(y[mask_c], weights=weights[mask_c])
        
        # 遍历每一个 Treatment 计算加权增益
        for k in self.treatment_classes_:
            mask_k = (T == k)
            if not np.any(mask_k): return None
            mu_k = np.average(y[mask_k], weights=weights[mask_k])
            cates[k] = mu_k - mu_c
            
        return cates

    def _check_constraints(self, T, weights):
        """
        [修改] 基于权重之和进行约束检查 (Phase 3)
        """
        # 检查加权对照组
        if np.sum(weights[T == 0]) < self.min_samples_treatment:
            return False
        
        # 检查所有加权干预组
        for k in self.treatment_classes_:
            if np.sum(weights[T == k]) < self.min_samples_treatment:
                return False
        return True

    def _find_best_split(self, X, y, T, weights):
        """
        [修改] 向量化加权分裂逻辑 (Phase 1)
        """
        best_gain = -np.inf
        best_feature_idx = None
        best_threshold = None
        
        n_samples, n_features = X.shape
        all_classes = [0] + self.treatment_classes_
        n_classes = len(all_classes)
        class_to_idx = {t: i for i, t in enumerate(all_classes)}
        
        # 构造加权统计矩阵
        # y_mat 存储 w*y, n_mat 存储 w (即加权样本计数)
        y_mat = np.zeros((n_samples, n_classes))
        n_mat = np.zeros((n_samples, n_classes))
        
        for i in range(n_samples):
            if T[i] in class_to_idx:
                col_idx = class_to_idx[T[i]]
                w = weights[i]
                y_mat[i, col_idx] = y[i] * w
                n_mat[i, col_idx] = w
        
        G_y = y_mat.sum(axis=0) 
        G_n = n_mat.sum(axis=0) 
        
        # 特征随机采样 (兼容 Forest)
        if hasattr(self, 'max_features'):
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

        for feat_idx in feature_indices:
            X_col = X[:, feat_idx]
            sorted_idx = np.argsort(X_col)
            
            X_sorted = X_col[sorted_idx]
            y_mat_sorted = y_mat[sorted_idx]
            n_mat_sorted = n_mat[sorted_idx]
            
            # 向量化计算左侧加权累积和
            L_y = np.cumsum(y_mat_sorted, axis=0) 
            L_n = np.cumsum(n_mat_sorted, axis=0) 
            
            # 计算右侧加权统计量
            R_y = G_y - L_y
            R_n = G_n - L_n
            
            # 加权样本量约束检查
            valid_mask = (L_n >= self.min_samples_treatment).all(axis=1) & \
                         (R_n >= self.min_samples_treatment).all(axis=1) & \
                         (L_n.sum(axis=1) >= self.min_samples_leaf) & \
                         (R_n.sum(axis=1) >= self.min_samples_leaf)
            
            unique_mask = np.concatenate([np.diff(X_sorted) != 0, [False]])
            final_mask = valid_mask & unique_mask
            
            if not np.any(final_mask):
                continue
                
            # 提取有效切分点的数据
            L_y_v, L_n_v = L_y[final_mask], L_n[final_mask]
            R_y_v, R_n_v = R_y[final_mask], R_n[final_mask]
            
            # 计算加权均值与 CATE
            mu_L = L_y_v / (L_n_v + 1e-9)
            mu_R = R_y_v / (R_n_v + 1e-9)
            
            tau_L = mu_L[:, 1:] - mu_L[:, 0:1] 
            tau_R = mu_R[:, 1:] - mu_R[:, 0:1]
            
            # 计算加权 Gain (异质性增益)
            size_L = L_n_v.sum(axis=1)
            size_R = R_n_v.sum(axis=1)
            
            gain = np.sum(tau_L**2, axis=1) * size_L + np.sum(tau_R**2, axis=1) * size_R
            
            current_best_idx = np.argmax(gain)
            if gain[current_best_idx] > best_gain:
                best_gain = gain[current_best_idx]
                best_feature_idx = feat_idx
                valid_indices = np.where(final_mask)[0]
                best_threshold = X_sorted[valid_indices[current_best_idx]]

        return best_feature_idx, best_threshold

    def _grow_tree(self, X, y, T, weights, depth):
        """
        [修改] 递归生长加权树
        """
        node_cates = self._calculate_cate_vector(y, T, weights)
        
        if node_cates is None:
            node_cates = {k: 0.0 for k in self.treatment_classes_}

        # 终止条件: 深度或加权总样本量
        if (depth >= self.max_depth) or (np.sum(weights) < self.min_samples_leaf):
            return UniversalUpliftTreeNode(depth=depth, is_leaf=True, value=node_cates)
        
        feat_idx, threshold = self._find_best_split(X, y, T, weights)
        
        if feat_idx is None:
            return UniversalUpliftTreeNode(depth=depth, is_leaf=True, value=node_cates)
        
        mask_left = X[:, feat_idx] <= threshold
        mask_right = ~mask_left
        
        node = UniversalUpliftTreeNode(depth=depth, is_leaf=False, value=node_cates)
        node.feature_idx = feat_idx
        node.threshold = threshold
        
        node.left = self._grow_tree(X[mask_left], y[mask_left], T[mask_left], weights[mask_left], depth + 1)
        node.right = self._grow_tree(X[mask_right], y[mask_right], T[mask_right], weights[mask_right], depth + 1)
        
        return node
    
    def print_tree(self, node=None, spacing=""):
        if node is None: node = self.root
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
    def __init__(self, max_features='sqrt', **kwargs):
        super().__init__(**kwargs)
        self.max_features = max_features

class UniversalUpliftForest:
    def __init__(self, n_estimators=100, max_depth=5, min_samples_leaf=10, 
                 min_samples_treatment=5, max_features='sqrt', n_jobs=-1, random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_treatment = min_samples_treatment
        self.max_features = max_features
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.trees_ = []
        self.is_binary_ = False

    def fit(self, X, y, T, sample_weight=None, p_scores=None):
        """
        [修改] 森林训练逻辑适配加权体系 (Phase 4)
        """
        rng = np.random.default_rng(self.random_state)
        
        unique_t = np.unique(T)
        treatment_classes = [t for t in unique_t if t != 0]
        self.is_binary_ = (len(treatment_classes) == 1)

        # 解决权重透传
        if sample_weight is None:
            if p_scores is not None:
                sample_weight = np.where(T >= 1, 1.0 / p_scores, 1.0 / (1 - p_scores))
            else:
                sample_weight = np.ones_like(y, dtype=np.float64)

        def _train_single_tree(seed):
            # 1. Bootstrap Sampling
            n_samples = X.shape[0]
            indices = rng.choice(n_samples, n_samples, replace=True)
            X_boot, y_boot, T_boot, w_boot = X[indices], y[indices], T[indices], sample_weight[indices]
            
            # 2. 初始化并训练加权树
            tree = RandomizedUpliftTree(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                min_samples_treatment=self.min_samples_treatment,
                max_features=self.max_features
            )
            tree.fit(X_boot, y_boot, T_boot, sample_weight=w_boot)
            return tree
        
        seeds = rng.integers(0, 100000, size=self.n_estimators)
        self.trees_ = Parallel(n_jobs=self.n_jobs)(
            delayed(_train_single_tree)(seed) for seed in seeds
        )
        return self

    def predict(self, X, p_scores=None):
        """
        聚合所有树的结果
        """
        all_preds = Parallel(n_jobs=self.n_jobs)(
            delayed(tree.predict)(X) for tree in self.trees_
        )
        all_preds = np.array(all_preds)
        avg_preds = np.mean(all_preds, axis=0)
        return avg_preds