# Observational Causal Uplift (OCU) Framework

> **基于因果图拓扑与两阶段架构的鲁棒增量推断框架**

## 1. 项目背景与目标

在计算广告、个性化医疗和智能营销等领域，**增量建模 (Uplift Modeling)** 的核心目标是估计个体级别的条件平均处理效应 (Conditional Average Treatment Effect, CATE)：


$$\tau(X_i) = \mathbb{E}[Y_i(1) | X_i] - \mathbb{E}[Y_i(0) | X_i]$$


然而，在海量观测数据 (Observational Data) 中，特征空间往往充斥着高维混杂因素、工具变量、对撞因子等复杂拓扑结构。传统机器学习或盲目的逆概率加权 (IPW) 极易陷入高维共线性与选择偏差陷阱。

**本项目 (OCU Framework)** 旨在提出并验证一种新型的两阶段因果增量评估框架。通过引入先验因果有向无环图 (DAG)，自动识别最小充分调整集 (Minimal Sufficient Adjustment Set) 与精度变量 (Precision Variables)，并在第二阶段与各类底层 Uplift 模型（S/T/X-Learner、标签转换、因果森林）进行**深入到数学公式级别的理论对齐与融合**，从而实现对 CATE 的无偏且低方差的鲁棒估计。

---

## 2. 面向的核心问题

本框架致力于解决传统因果推断在工业落地中的三大痛点：

1. **高维共线性与正定性违背 (Positivity Violation)**：传统 IPW 通常将所有观测变量 (或根混杂节点 Root Confounders) 纳入倾向分模型，导致多重共线性，且倾向分极易趋近于 0 或 1，引发方差爆炸。
2. **工具变量与对撞因子污染**：盲目使用全部特征会导致工具变量 (IV) 被纳入调整集（放大估计方差或引入 M-Bias），或对撞因子 (Collider) 被控制（打开后门路径，引入新混杂）。
3. **元学习器 (Meta-Learners) 与去混杂权重的生硬结合**：开源库在应用 IPW 权重时往往采用统一的 `sample_weight` 传参，忽略了不同算法 (如 Class Transform) 底层的数学推导，导致“双重逆概率加权”等理论错误。

---

## 3. 二阶段框架的核心创新点

OCU 框架通过 `FirstStageDeconfounder` 和 `SecondStageUpliftTrainer` 的解耦设计，优雅地解决了上述问题。

### Stage 1: 因果图驱动的特征隔离与去偏 (Deconfounding)

* **智能特征集提取**：基于给定的因果图结构，框架自动提取**最优近端混杂集** ($C_{opt}$, Proximal Confounders) 用于切断后门路径，同时提取**精度变量** ($S_{prec}$, 仅影响 $Y$ 的变量) 用于降低后续估计的方差。
* **低维稳健的倾向分估计**：仅使用 $C_{opt}$ 拟合倾向分 $e(X)$，避免了高维根混杂带来的过拟合。生成的 IPW 权重经过了截断与稳定化处理，确保满足重叠性假设 (Overlap Assumption)。

### Stage 2: 理论对齐的异质性效应建模 (Uplift Modeling)

* **构造伪随机对照试验 (Pseudo-RCT)**：将 $C_{opt} \cup S_{prec}$ 作为建模特征空间，将第一阶段产出的 IPW 权重 $W$ 和倾向分 $e(X)$ 精准注入底层的基模型。

---

## 4. 创新实现细节：多模型适配策略

在 `models/` 目录下，本项目对主流的 Uplift 算法进行了深度重构，确保去混杂逻辑与算法数学原理的绝对对齐：

1. **S-Learner / T-Learner (元学习器)**：
* **逻辑**：严格基于加权经验风险最小化 (Weighted ERM)。通过在 XGBoost 等回归器中注入 `sample_weight`，在消除混杂的伪总提上拟合响应曲面。


2. **X-Learner (交叉学习器)**：
* **创新**：传统 X-Learner 的最后一步融合 $\tau = g(x)\tau_0 + (1-g(x))\tau_1$ 往往使用全量特征内部拟合倾向分 $g(x)$。本框架直接截断内部拟合，**复用第一阶段基于 $C_{opt}$ 产出的无偏倾向分**，彻底杜绝了 M-Bias 的引入。


3. **Class Transform (标签转换法)**：
* **创新**：理论上，转换标签 $Y^* = Y \cdot \frac{T - e(x)}{e(x)(1-e(x))}$ 自身已包含了逆概率去偏机制。如果在此回归器中再传入 `sample_weight`，将导致方差呈 $1/e(x)^2$ 爆炸。**代码中做出了极其关键的强制剥离**，在拟合时显式去除了 `sample_weight` 传参，完美避开理论陷阱。


4. **Universal Uplift Forest (因果森林)**：
* **创新**：重写了决策树的底层分裂准则。在计算异质性分裂增益时，所有统计量（目标和 $G_y$、样本计数 $G_n$）均被替换为**加权统计量**；叶子节点的纯度与样本量约束也变更为**加权样本量约束**，确保在每个分裂子空间内均实现无偏的 CATE 划分。



---

## 5. 整体项目设计逻辑

整个 `main.py` 是实验的中央引擎，设计了严谨的对照消融实验 (Ablation Study)：

1. **数据生成与处理机制**：支持高度复杂的合成数据生成（具有 Ground Truth，包含 $C_{root} \rightarrow C_{prox}$ 的高维漏斗网络），并完美兼容真实业务数据（Lenta 零售数据集、TTS 数据集）。
2. **多基线对比实验**：对于每一个算法，依次运行 `Naive`（忽略因果图直接拟合）、`Trad_IPW_NoPrec`（全量变量去偏无精度变量）、`Trad_IPW_WithPrec` 以及本项目的 `OCU_Ours`。
3. **无偏评估体系**：在 `UpliftMetrics` 中，摒弃了传统的瑕疵评估法，针对观测数据使用 IPW 修正的 AUUC 和 Qini 系数：

$$V_i^T = \frac{T_i \cdot Y_i}{e(X_i)}, \quad V_i^C = \frac{(1-T_i) \cdot Y_i}{1-e(X_i)}$$



并通过 Spearman 秩相关系数衡量核心的干预效应排序能力。

---

## 6. 项目代码逻辑关系 (Directory Structure)

```text
uplift_lab_based_causalGraph/
│
├── main.py                     # 实验主引擎：数据流转、模型实例化、批量消融实验与结果聚合
├── ocu_framework.py            # OCU 核心框架：包含 FirstStageDeconfounder 与 SecondStageUpliftTrainer
├── getProxConAndRegVars.py     # DAG 图论引擎：负责寻找最小充分调整集和后门路径分析
├── lgb_models.py               # 倾向分核心估计器：封装了 Optuna 优化的 LightGBM 分类器
├── evaluation.py               # 可视化与评估引擎：绘制无偏的 AUUC 曲线与 Decile 增益图
│
├── models/                     # 底层 Uplift 模型库 (深度重构以兼容 OCU 框架)
│   ├── baseClass.py            # 估算器基类，定义 fit/predict 抽象接口
│   ├── meta_learners.py        # S-Learner, T-Learner, X-Learner (融入阶段 1 P-Score)
│   ├── class_transform.py      # 标签转换法 (严格防范双重逆概率加权)
│   └── tree.py                 # 全能因果树/森林 (底层分裂准则实现 IPW 权重注入)
│
└── data/                       # 存放真实数据集 (如 Lenta, TTS 等)

```

* **逻辑流转**：
`main.py` $\rightarrow$ 获取数据集与 DAG $\rightarrow$ 交由 `ocu_framework.py (FirstStage)` 提取 $C_{opt}$ 并产出 IPW $\rightarrow$ 交由 `ocu_framework.py (SecondStage)` 打包特征与权重 $\rightarrow$ 路由至 `models/*.py` 依据各自数学特性执行拟合 $\rightarrow$ 回传预测结果至 `main.py` 的评估模块进行绘图与指标计算。