import json
import re
import pandas as pd
from graphviz import Digraph
import os

# 增强的绘图函数，重点优化中文显示
def draw_causal_graph(CDict, graphName='graph'):
    # 创建Digraph对象并设置字体和输出编码
    g = Digraph('Causal Graph', encoding='utf-8')
    
    # 添加graphviz特定的全局字体配置，使用fontname和fontpath，同时设置dpi
    g.attr('graph', 
           fontname='SimHei',
           fontsize='10',
           charset='UTF-8',
           dpi='600')  # 显式指定字符集和分辨率
    
    # 添加节点和边
    for child, parents in CDict.items():
        # 为每个节点设置更详细的字体属性
        g.node(name=child, 
               color='blue', 
               fontname='SimHei',
               fontsize='10',
               style='filled',
               fillcolor="#A1D722")  # 轻微填充背景以增强可读性
        if len(parents) != 0:
            for parent in parents:
                g.edge(parent, child, color='green')
        '''
        else:
            g.edge('root', child, color='red')
        '''
    
    # 使用render方法，显式设置字体路径和输出格式
    g.render(filename=graphName, 
             format='png', 
             view=False,
             cleanup=True)  # 自动清理中间文件

def add_missing_ancestors(ancestors_dict):
    # 获取所有键的集合
    keySet = set(ancestors_dict.keys())
    
    # 获取所有值的集合（所有父节点的集合）
    valueSet = set()
    for parents in ancestors_dict.values():
        valueSet.update(parents)
    
    # 计算差值：只出现在值中但不在键中的变量
    diff_set = valueSet - keySet
    
    # 为差值集合中的每个变量添加空父节点列表
    for var in diff_set:
        ancestors_dict[var] = []

    return ancestors_dict

# 改进文件读取函数，确保更好的编码处理
def read_ancestors_from_file(file_path):
    encodings = ['utf-8', 'gbk', 'gb2312', 'utf-16']
    content = None
    
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read()
            break
        except UnicodeDecodeError:
            continue
    
    if content is None:
        raise ValueError(f"无法使用任何支持的编码读取文件: {file_path}")
    
    match = re.search(r'Ancestors:\s*(\{.*\})', content, re.DOTALL)
    if match:
        try:
            ancestors_dict = eval(match.group(1))
            
            # 检查并添加缺失的祖先
            ancestors_dict = add_missing_ancestors(ancestors_dict)
            
            return ancestors_dict
        except Exception as e:
            raise ValueError(f"解析字典时出错: {e}")
    else:
        raise ValueError("无法从文件中提取Ancestors字典")

# 增强的中英文映射函数，特别处理可能的编码问题
def create_eng_chi_mapping(csv_path):
    # 增加更多的字体编码支持
    encodings = ['utf-8', 'gbk', 'gb2312', 'utf-16', 'latin-1']
    df = None
    
    for encoding in encodings:
        try:
            df = pd.read_csv(csv_path, encoding=encoding)
            # 验证读取的数据是否包含有效的中文字符
            if any(any('\u4e00' <= char <= '\u9fff' for char in str(row.iloc[1])) 
                  for _, row in df.iterrows() if len(row) >= 2):
                break
        except UnicodeDecodeError:
            continue
    
    if df is None:
        # 最后的尝试
        try:
            df = pd.read_csv(csv_path, encoding='utf-8', errors='replace')
        except Exception as e:
            raise ValueError(f"无法读取CSV文件: {e}")
    
    # 创建更健壮的映射字典
    mapping = {}
    for _, row in df.iterrows():
        if len(row) >= 2:
            try:
                key = str(row.iloc[0])
                # 对值进行多步骤编码处理
                value = row.iloc[1]
                # 确保是字符串
                if not isinstance(value, str):
                    value = str(value)
                
                # 尝试修复可能的编码问题
                if '�' in value:
                    # 尝试不同的编码转换
                    for enc in ['gbk', 'utf-8', 'gb2312']:
                        try:
                            value_bytes = bytes(str(row.iloc[1]), 'latin-1')
                            value = value_bytes.decode(enc)
                            if '�' not in value:
                                break
                        except:
                            continue
                
                # 只有当值包含有效中文字符时才添加到映射
                if any('\u4e00' <= char <= '\u9fff' for char in value) or key == value:
                    mapping[key] = value
                else:
                    # 如果没有有效中文字符，尝试直接使用原始值
                    mapping[key] = str(row.iloc[1])
            except Exception as e:
                print(f"处理映射时出错: {e}")
                continue
    
    return mapping

# 转换函数保持不变
def convert_to_chinese_dict(eng_dict, mapping_dict):
    chi_dict = {}
    all_nodes = set(eng_dict.keys())
    for parents in eng_dict.values():
        all_nodes.update(parents)
    
    node_mapping = {}
    for node in all_nodes:
        if node in mapping_dict:
            chinese_name = mapping_dict[node]
            if any('\u4e00' <= char <= '\u9fff' for char in chinese_name):
                node_mapping[node] = chinese_name
            else:
                node_mapping[node] = node
        else:
            node_mapping[node] = node
    
    for child, parents in eng_dict.items():
        chi_child = node_mapping[child]
        chi_parents = [node_mapping[parent] for parent in parents]
        chi_dict[chi_child] = chi_parents
    
    return chi_dict

# 主函数
if __name__ == "__main__":
    # 文件路径
    # ancestors_file = "d:\PostPhD\code\heartIll\Ancestors.txt"
    csv_file = "./data/Eng_Chi_Cols.csv"
    
    try:
        # 读取英文祖先集字典
        # english_ancestors = {'main_talk_dur': ['duotou_3rd_cnt_new', 'is_pay_study_cust', 'age', 'login_count_30d', 'xyl_score', 'voice_role_cd'], 'duotou_3rd_cnt_new': ['login_count_90d', 'last_mon_ext_apply_cnt', 'last_5103_wdraw_to_date'], 'voice_role_cd': ['census_pr_cd', 'credit_lim_new', 'age', 'min_recent_login_days', 'last_5103_appl_time_to_date', 'reg_dt_to_date', 'last_5103_wdraw_to_date', 'accum_loan_amt', 'occ_cd', 'ocr_ethnic'], 'is_pay_study_cust': ['last_5103_appl_time_to_date', 'gender_cd', 'is_internet_busi_cust', 'is_freq_traveler', 'age'], 'age': [], 'login_count_30d': ['login_count_7d', 'login_count_3d'], 'xyl_score': ['last_180d_5103_wdraw_app_amt', 'last_180_day_5103_wdraw_baff_cnt', 'last_180d_5103_wdraw_app_days', 'last_180d_5103_lim_use_rate', 'duotou_3rd_cnt_new', 'login_count_30d'], 'last_180_day_5103_wdraw_baff_cnt': ['last_180d_5103_lim_use_rate', 'last_180_day_5103_appl_wdraw_refuse_cnt', 'last_180d_5103_wdraw_app_amt', 'last_180d_5103_wdraw_app_days', 'last_180d_5103_lim_pass_rate'], 'login_count_7d': ['last_180d_5103_wdraw_app_days', 'login_count_3d', 'last_5103_wdraw_to_date'], 'last_180d_5103_lim_use_rate': ['last_180_day_5103_wdraw_baff_cnt', 'last_180d_5103_loan_days', 'last_180_day_5103_appl_wdraw_refuse_cnt', 'last_180d_5103_wdraw_app_amt', 'last_180d_5103_wdraw_app_days'], 'login_count_3d': [], 'gender_cd': [], 'last_180d_5103_loan_amt': ['last_180d_5103_loan_days', 'last_180d_5103_lim_pass_rate'], 'last_180d_5103_wdraw_app_amt': ['last_180d_5103_wdraw_app_days', 'last_180_day_5103_wdraw_baff_cnt', 'last_180d_5103_loan_amt'], 'is_internet_busi_cust': [], 'is_freq_traveler': ['edu_deg_cd', 'house_loan_cnt_lvl', 'is_internet_busi_cust'], 'last_180d_5103_wdraw_app_days': ['last_180d_5103_wdraw_app_amt'], 'last_mon_ext_apply_cnt': [], 'last_5103_wdraw_to_date': ['login_count_7d', 'last_180_day_5103_appl_wdraw_refuse_cnt'], 'edu_deg_cd': ['marital_status_cd'],  'house_loan_cnt_lvl': ['credit_org_num_lvl', 'income_range_ind', 'zx_job_name', 'marital_status_cd', 'bus_loan_cnt_lvl', 'zx_job_title', 'zx_empl_status', 'zx_occ_th2', 'edu_deg_cd', 'zx_occ_th1', 'zx_occ_th3', 'car_loan_cnt_lvl', 'occ_cd'], 'last_180d_5103_loan_days': ['credit_org_num_lvl', 'income_range_ind', 'marital_status_cd', 'bus_loan_cnt_lvl', 'card_credit_avg_lvl', 'zx_occ_th2', 'edu_deg_cd', 'last_180_day_5103_loan_wdraw_cnt', 'cl_org_num_lvl', 'zx_occ_th1', 'last_30_day_5103_loan_wdraw_cnt', 'car_loan_cnt_lvl'], 'last_180d_5103_lim_pass_rate': ['spl_prin_sum'], 'card_credit_avg_lvl': ['credit_org_num_lvl', 'income_range_ind', 'zx_job_name', 'marital_status_cd', 'bus_loan_cnt_lvl', 'zx_job_title', 'zx_empl_status', 'zx_occ_th2', 'edu_deg_cd', 'house_loan_cnt_lvl', 'last_180_day_5103_loan_wdraw_cnt', 'zx_occ_th1', 'zx_occ_th3', 'car_loan_cnt_lvl', 'occ_cd'], 'spl_prin_sum': ['last_30_day_5103_loan_wdraw_cnt'], 'last_30_day_5103_loan_wdraw_cnt': ['is_tel_sale_pos_answer_user'], 'marital_status_cd': ['is_student_cust_gp'], 'residence_pr_cd': ['census_pr_cd'], 'is_tel_sale_pos_answer_user': [], 'is_student_cust_gp': ['census_pr_cd', 'residence_pr_cd'], 'census_pr_cd': [], 'car_loan_cnt_lvl': [], 'bus_loan_cnt_lvl': [], 'zx_occ_th2': [], 'last_180_day_5103_loan_wdraw_cnt': [], 'ocr_ethnic': [], 'zx_empl_status': [], 'accum_loan_amt': [], 'cl_org_num_lvl': [], 'reg_dt_to_date': [], 'occ_cd': [], 'zx_occ_th3': [], 'last_180_day_5103_appl_wdraw_refuse_cnt': [], 'last_5103_appl_time_to_date': [], 'credit_lim_new': [], 'min_recent_login_days': [], 'zx_job_name': [], 'zx_job_title': [], 'zx_occ_th1': [], 'credit_org_num_lvl': [], 'login_count_90d': [], 'income_range_ind': []}
        
        english_ancestors = {'response_att': ['k_var_cheque_category_width_15d', 'k_var_days_between_visits_1m', 'k_var_disc_per_cheque_15d', 'k_var_cheque_15d', 'mean_discount_depth_15d', 'group'], 'group': ['crazy_purchases_cheque_count_12m', 'sale_count_12m_g32', 'sale_sum_12m_g26', 'cheque_count_12m_g56', 'disc_sum_6m_g34', 'cheque_count_12m_g32', 'promo_share_15d'], 'k_var_cheque_category_width_15d': ['food_share_15d', 'k_var_disc_per_cheque_15d', 'promo_share_15d', 'k_var_cheque_group_width_15d', 'mean_discount_depth_15d'], 'k_var_days_between_visits_1m': ['stdev_days_between_visits_15d', 'k_var_days_between_visits_15d', 'cheque_count_6m_g40', 'food_share_1m', 'k_var_disc_per_cheque_15d', 'perdelta_days_between_visits_15_30d', 'cheque_count_6m_g32', 'cheque_count_12m_g41', 'k_var_cheque_category_width_15d'], 'k_var_disc_per_cheque_15d': ['cheque_count_12m_g41', 'mean_discount_depth_15d'], 'k_var_cheque_15d': ['k_var_cheque_category_width_15d', 'stdev_days_between_visits_15d', 'k_var_days_between_visits_1m', 'k_var_days_between_visits_15d', 'k_var_sku_per_cheque_15d', 'food_share_15d', 'k_var_disc_per_cheque_15d', 'promo_share_15d', 'k_var_cheque_group_width_15d', 'mean_discount_depth_15d'], 'mean_discount_depth_15d': [], 'cheque_count_6m_g32': ['disc_sum_6m_g34'], 'cheque_count_6m_g40': ['disc_sum_6m_g34', 'cheque_count_6m_g32', 'cheque_count_12m_g46', 'cheque_count_6m_g33', 'sale_count_12m_g54', 'cheque_count_12m_g32', 'cheque_count_6m_g41', 'cheque_count_12m_g41'], 'cheque_count_12m_g41': ['cheque_count_12m_g33', 'cheque_count_6m_g41', 'cheque_count_12m_g42'], 'food_share_1m': ['disc_sum_6m_g34', 'cheque_count_6m_g40', 'cheque_count_6m_g46', 'cheque_count_6m_g32', 'cheque_count_6m_g41', 'crazy_purchases_cheque_count_12m', 'cheque_count_6m_g33'], 'cheque_count_12m_g33': ['cheque_count_6m_g33'], 'disc_sum_6m_g34': [], 'cheque_count_6m_g33': ['disc_sum_6m_g34', 'sale_sum_6m_g54'], 'cheque_count_12m_g42': ['cheque_count_12m_g25', 'disc_sum_6m_g34', 'cheque_count_12m_g33']}
        '''
        # subGraph
        target = 'main_talk_dur'
        subGraph = {}
        subGraph[target] = english_ancestors[target]
        for parent in english_ancestors[target]:
            subGraph[parent] = english_ancestors[parent]
        english_ancestors = subGraph
        '''
        # 检查并添加缺失的祖先
        english_ancestors = add_missing_ancestors(english_ancestors)
        print(english_ancestors)

        # 创建中英文映射
        # eng_chi_mapping = create_eng_chi_mapping(csv_file)
        
        # 转换为中文祖先集字典
        # chinese_ancestors = convert_to_chinese_dict(english_ancestors, eng_chi_mapping)
        
        # 绘制中文因果图
        # draw_causal_graph(chinese_ancestors, graphName='Agent_Chinese_subgraph')

        # 绘制英文因果图
        draw_causal_graph(english_ancestors, graphName='lenta_English_graph')
        
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()