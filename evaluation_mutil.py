"""
Uplift 模型评估对比 - 终极修正版 (Final Patch Version)
适用环境: Python 3.13 + scikit-learn 1.6+
修复内容:
1. [关键] 注入 check_matplotlib_support 补丁，修复 scikit-uplift 在 sklearn 1.6 下的崩溃问题。
2. [关键] 使用循环调用规避 CausalML 的 unhashable list 错误。
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

# ==========================================
# 🛠️ 关键补丁：修复 scikit-uplift 报错
# ==========================================
import sklearn.utils
try:
    from sklearn.utils import check_matplotlib_support
except ImportError:
    # 如果 sklearn 1.6+ 删除了该函数，我们手动补上
    def check_matplotlib_support(caller_name):
        try:
            import matplotlib
        except ImportError:
            raise ImportError(f"{caller_name} requires matplotlib")
    # 注入到 sklearn.utils 中
    sklearn.utils.check_matplotlib_support = check_matplotlib_support
    print("✅ 已应用 sklearn 1.6+ 兼容性补丁 (check_matplotlib_support)")

# 补丁应用后，再导入 sklift
try:
    from sklift.metrics import uplift_auc_score, qini_auc_score
    from sklift.viz import plot_uplift_curve
    SKLIFT_INSTALLED = True
except ImportError as e:
    print(f"⚠️ sklift 导入失败: {e}")
    SKLIFT_INSTALLED = False

warnings.filterwarnings('ignore')

# ==========================================
# 主逻辑
# ==========================================
def run_package_comparison():
    print("\n🚀 Step 1: 生成模拟数据...")
    np.random.seed(2025)
    N = 10000

    # 生成数据
    y_true = np.random.binomial(1, 0.2, N)
    t_true = np.random.binomial(1, 0.5, N)
    
    # 模拟模型：Model Good (有效果), Model Random (无效果)
    pred_good = np.random.normal(0, 1, N) + (y_true * t_true * 2.0)
    pred_random = np.random.normal(0, 1, N)

    # 构造 DataFrame (CausalML 需要)
    df = pd.DataFrame({
        'y': y_true,
        't': t_true,
        'model_good': pred_good,
        'model_random': pred_random
    }).reset_index(drop=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # -------------------------------------------------
    # 🟢 Step 2: CausalML (循环调用模式)
    # -------------------------------------------------
    print("🚀 Step 2: 调用 CausalML...")
    try:
        from causalml.metrics import get_cumgain, get_qini
        
        # ⚠️ 必须一个个算，不能传列表，否则报 unhashable list
        models = ['model_good', 'model_random']
        auuc_curves = pd.DataFrame()
        qini_scores = {}

        for model in models:
            # 计算曲线数据
            c_gain = get_cumgain(
                df, 
                outcome_col='y', 
                treatment_col='t', 
                treatment_effect_col=model # 单个字符串，绝对安全
            )
            
            # 统一格式：有些版本返回Series，有些返回DataFrame
            if isinstance(c_gain, pd.DataFrame):
                auuc_curves[model] = c_gain.iloc[:, 0]
            elif isinstance(c_gain, pd.Series):
                auuc_curves[model] = c_gain
            
            # 计算 Qini 分数
            q_score = get_qini(
                df, 
                outcome_col='y', 
                treatment_col='t', 
                treatment_effect_col=model
            )
            # 解析分数
            if hasattr(q_score, 'iloc'):
                val = q_score.iloc[0]
            elif hasattr(q_score, 'values'):
                val = q_score.values[0] if len(q_score.values)>0 else 0
            else:
                val = q_score
            qini_scores[model] = float(val)

        print(f"   ✅ CausalML Qini: {qini_scores}")

        # 绘图
        ax = axes[0]
        auuc_curves.plot(ax=ax, linewidth=2)
        # 画随机线
        if not auuc_curves.empty:
            end_val = auuc_curves.iloc[-1, 0]
            ax.plot([0, auuc_curves.index.max()], [0, end_val], 'k--', label='Random', alpha=0.5)
        
        ax.set_title("CausalML: AUUC (Cumulative Gain)", fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

    except ImportError:
        print("❌ 未安装 causalml")
    except Exception as e:
        print(f"❌ CausalML 运行出错: {e}")

    # -------------------------------------------------
    # 🟠 Step 3: scikit-uplift (使用补丁后)
    # -------------------------------------------------
    print("🚀 Step 3: 调用 scikit-uplift...")
    if SKLIFT_INSTALLED:
        try:
            # 准备数据 (必须是一维数组)
            y_val = df['y'].values
            t_val = df['t'].values
            score_val = df['model_good'].values

            # 计算指标
            u_auc = uplift_auc_score(y_true=y_val, uplift=score_val, treatment=t_val)
            q_auc = qini_auc_score(y_true=y_val, uplift=score_val, treatment=t_val)
            
            print(f"   ✅ sklift Uplift AUC: {u_auc:.4f}")
            print(f"   ✅ sklift Qini AUC:   {q_auc:.4f}")

            # 绘图
            ax = axes[1]
            plot_uplift_curve(
                y_true=y_val, 
                uplift=score_val, 
                treatment=t_val, 
                random=True, 
                ax=ax,
                name=f'Model Good'
            )
            ax.set_title(f"scikit-uplift: Uplift Curve\n(AUC={u_auc:.4f})", fontweight='bold')
            ax.grid(True, alpha=0.3)

        except Exception as e:
            print(f"❌ sklift 运行出错: {e}")
            axes[1].text(0.5, 0.5, f"Error: {str(e)[:50]}", ha='center')
    else:
        print("❌ sklift 未导入成功")
        axes[1].text(0.5, 0.5, "Import Failed", ha='center')

    # -------------------------------------------------
    # 保存结果
    # -------------------------------------------------
    plt.tight_layout()
    save_file = './results/final_comparison.png'
    plt.savefig(save_file, dpi=150)
    print(f"\n✨ 成功! 图片已保存至: {save_file}")
    # plt.show()

if __name__ == "__main__":
    run_package_comparison()