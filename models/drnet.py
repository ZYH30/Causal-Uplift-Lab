import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import KFold

class SharedBottomNet(nn.Module):
    def __init__(self, input_dim, num_treatments, hidden_dim=64):
        super(SharedBottomNet, self).__init__()
        
        # 1. Shared Bottom (表示层)
        # 负责提取所有 Treatment 共用的特征
        self.shared_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim), # BN 加速收敛
            nn.Dropout(0.1),            # 防止过拟合
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 2. Multi-Heads (塔层)
        # 为每个 Treatment (包括对照组) 创建一个独立的 Head
        # nn.ModuleList 确保这些层被 PyTorch 注册和追踪
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1) # 输出标量预测值
            ) for _ in range(num_treatments)
        ])

    def forward(self, x):
        # 提取共享特征
        representation = self.shared_layer(x)
        
        # 获取所有 Head 的预测结果
        outputs = []
        for head in self.heads:
            outputs.append(head(representation))
            
        # 拼接结果: [Batch, Num_Treatments]
        return torch.cat(outputs, dim=1)

class DRNetEstimator(BaseEstimator):
    def __init__(self, 
                 hidden_dim=64, 
                 learning_rate=0.001, 
                 batch_size=256, 
                 epochs=50, 
                 clip_propensity=0.05,
                 first_stage_model_y=None, # 第一阶段 Nuisance Model
                 first_stage_model_t=None):
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.clip_propensity = clip_propensity
        
        # 默认使用简单模型作为 First Stage (保证双重鲁棒性)
        self.model_y_base = first_stage_model_y if first_stage_model_y else LinearRegression()
        self.model_t_base = first_stage_model_t if first_stage_model_t else LogisticRegression(multi_class='multinomial')
        
        self.net_ = None
        self.scaler_ = StandardScaler()
        self.treatment_classes_ = None
        
    def _compute_dr_target(self, X, y, T):
        """
        核心步骤: 计算双重鲁棒伪标签 (DR Pseudo-Label)
        """
        n_samples = len(y)
        dr_targets = np.zeros(n_samples)
        
        # 使用 Cross-Fitting 避免过拟合 (类似 DML)
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        
        for train_idx, val_idx in kf.split(X):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            T_tr, T_val = T[train_idx], T[val_idx]
            
            # 1. 训练 Propensity Model P(T|X)
            # 这里必须支持多分类
            model_t = self.model_t_base.__class__(**self.model_t_base.get_params())
            model_t.fit(X_tr, T_tr)
            probs = model_t.predict_proba(X_val) # [N_val, K]
            
            # 2. 训练 Outcome Model E[Y|X, T]
            # 为了简单，我们对每个 Treatment 训练一个回归器 (T-Learner 风格)
            mu_preds = np.zeros((len(val_idx), len(self.treatment_classes_)))
            
            for i, t_k in enumerate(self.treatment_classes_):
                # 只用该 Treatment 的数据训练
                mask_tr = (T_tr == t_k)
                if np.sum(mask_tr) > 0:
                    model_y = self.model_y_base.__class__(**self.model_y_base.get_params())
                    model_y.fit(X_tr[mask_tr], y_tr[mask_tr])
                    mu_preds[:, i] = model_y.predict(X_val)
            
            # 3. 计算 DR Target
            for i in range(len(val_idx)):
                t_obs = T_val[i]
                # 获取 t_obs 对应的索引
                t_idx = np.where(self.treatment_classes_ == t_obs)[0][0]
                
                # 获取 mu(x, t_obs)
                mu_val = mu_preds[i, t_idx]
                
                # 获取 e(x, t_obs) 并截断
                e_val = probs[i, t_idx]
                e_val = max(e_val, self.clip_propensity) # Clipping
                
                # DR 公式: mu + (y - mu) / e
                dr_targets[val_idx[i]] = mu_val + (y_val[i] - mu_val) / e_val
                
        return dr_targets

    def fit(self, X, y, T):
        # 1. 数据预处理
        X = self.scaler_.fit_transform(X) # 深度学习必须标准化
        self.treatment_classes_ = np.unique(T)
        num_treatments = len(self.treatment_classes_)
        input_dim = X.shape[1]
        
        # 2. 计算 DR Target (第一阶段)
        # 这一步生成了每个样本的 "去偏 Label"
        print("⚡ [Phase 1] Computing DR Targets via Cross-Fitting...")
        dr_targets = self._compute_dr_target(X, y, T)
        
        # 3. 初始化 PyTorch 网络
        self.net_ = SharedBottomNet(input_dim, num_treatments, self.hidden_dim)
        optimizer = torch.optim.Adam(self.net_.parameters(), lr=self.learning_rate)
        loss_fn = nn.MSELoss()
        
        # 转换为 Tensor
        X_tensor = torch.tensor(X, dtype=torch.float32)
        target_tensor = torch.tensor(dr_targets, dtype=torch.float32).view(-1, 1)
        T_tensor = torch.tensor(T, dtype=torch.long)
        
        # Map Treatment Value to Index (0, 1, 2...)
        # 确保 T 的值对应 Head 的索引
        t_mapper = {t: i for i, t in enumerate(self.treatment_classes_)}
        T_indices = torch.tensor([t_mapper[t] for t in T], dtype=torch.long)
        
        # 4. 训练循环 (Phase 2)
        print(f"🔥 [Phase 2] Training DRNet with {num_treatments} Heads...")
        self.net_.train()
        dataset = torch.utils.data.TensorDataset(X_tensor, target_tensor, T_indices)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for batch_x, batch_target, batch_t in dataloader:
                optimizer.zero_grad()
                
                # Forward: 获取所有 Head 的预测 [Batch, K]
                all_preds = self.net_(batch_x)
                
                # Masking: 我们只优化对应 observed T 的那个 Head
                # gather 选取对应 t 索引的预测值
                selected_preds = all_preds.gather(1, batch_t.view(-1, 1))
                
                # Loss Calculation
                loss = loss_fn(selected_preds, batch_target)
                
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                print(f"   Epoch {epoch+1}/{self.epochs}, Loss: {epoch_loss / len(dataloader):.4f}")
                
        return self

    def predict(self, X):
        """
        返回 CATE 矩阵 (相对于 Control 组)
        Shape: (N, K-1) 
        例如: T=[0, 1, 2], 返回 [CATE_1, CATE_2]
        """
        self.net_.eval()
        X_scaled = self.scaler_.transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        
        with torch.no_grad():
            # 获取所有 Head 的预测 [N, K]
            all_preds = self.net_(X_tensor).numpy()
            
        # 假设第 0 个 class 是 control (需要 fit 时数据里有 0)
        # 找到 T=0 的列索引
        control_idx = np.where(self.treatment_classes_ == 0)[0][0]
        control_pred = all_preds[:, control_idx].reshape(-1, 1)
        
        cates = []
        # 遍历所有非 Control 的 Treatment
        for i, t_val in enumerate(self.treatment_classes_):
            if i == control_idx:
                continue
            # CATE = Y_t - Y_c
            cate_k = all_preds[:, i].reshape(-1, 1) - control_pred
            cates.append(cate_k)
            
        return np.hstack(cates)