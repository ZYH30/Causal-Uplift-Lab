import collections
import itertools
from graphviz import Digraph
import os
from tqdm import tqdm

# 增强的绘图函数，重点优化中文显示
def draw_causal_graph(CDict, graphName='graph'):
    # (此函数不变，保持原样)
    g = Digraph('Causal Graph', encoding='utf-8')
    g.attr('graph', 
           fontname='SimHei',
           fontsize='10',
           charset='UTF-8')
    
    all_nodes = set(CDict.keys())
    for parents in CDict.values():
        all_nodes.update(parents)
        
    for node in all_nodes:
        g.node(name=node, 
               color='blue', 
               fontname='SimHei',
               fontsize='10',
               style='filled',
               fillcolor='#f9f9ff')

    for child, parents in CDict.items():
        if len(parents) != 0:
            for parent in parents:
                g.edge(parent, child, color='green')
    try:
        g.render(filename=graphName, 
                 format='png', 
                 view=False,
                 cleanup=True)
        print(f"因果图已保存为 {graphName}.png")
    except Exception as e:
        print(f"无法渲染因果图（可能未安装Graphviz或SimHei字体）：{e}")

# --- 辅助函数：图操作 (不变) ---

def get_parents(graph: dict, node: str) -> set:
    return set(graph.get(node, []))

def get_children(graph_adj: dict, node: str) -> set:
    return set(graph_adj.get(node, []))

def invert_graph(graph: dict) -> dict:
    """
    反转因果图，将 (子 -> [父]) 格式转换为 (父 -> [子]) 格式。
    这对于执行后代搜索和有向路径查找至关重要。
    """
    # 1. 收集图中出现过的所有节点
    # 必须同时收集键（子节点）和值（父节点），
    # 因为根节点可能只作为父节点出现，而没有自己的条目。
    all_nodes = set(graph.keys())
    all_parents = set()
    for parents in graph.values():
        all_parents.update(parents)
    
    all_nodes_total = all_nodes.union(all_parents)
    
    # 2. 为所有节点初始化一个空的邻接表（父->子）
    # 这确保了即使一个节点只有父节点没有子节点，它也会出现在图中。
    graph_adj = {node: [] for node in all_nodes_total}
    
    # 3. 填充邻接表
    for node, parents in graph.items():
        for p in parents:
            if p in graph_adj:
                graph_adj[p].append(node) # 将 'node' 添加为 'p' 的子节点
    return graph_adj

def get_ancestors(graph: dict, node: str) -> set:
    """
    获取一个节点的所有祖先（不包括节点自身）。
    使用广度优先搜索 (BFS) 向上遍历父节点。
    
    :param graph: 因果图 (子 -> [父])
    :param node: 目标节点
    :return: 包含所有祖先的集合
    """
    ancestors = set()
    if node not in graph:
        return ancestors # 如果节点不在图的键中，它没有父节点
        
    queue = collections.deque(graph.get(node, [])) # 从直接父节点开始
    visited = set(graph.get(node, [])) # 跟踪已访问节点以避免重复
    
    while queue:
        current = queue.popleft()
        if current not in ancestors:
            ancestors.add(current)
            # 获取当前节点的父节点
            parents = graph.get(current, [])
            for p in parents:
                if p not in visited:
                    visited.add(p)
                    queue.append(p)
    return ancestors

def get_descendants(graph_adj: dict, node: str) -> set:
    """
    获取一个节点的所有后代（不包括节点自身）。
    使用广度优先搜索 (BFS) 向下遍历子节点。
    
    :param graph_adj: 反转的因果图 (父 -> [子])
    :param node: 目标节点
    :return: 包含所有后代的集合
    """
    descendants = set()
    if node not in graph_adj:
        return descendants # 如果节点不在图的键中，它没有子节点
        
    queue = collections.deque(graph_adj.get(node, [])) # 从直接子节点开始
    visited = set(graph_adj.get(node, []))
    
    while queue:
        current = queue.popleft()
        if current not in descendants:
            descendants.add(current)
            # 获取当前节点的子节点
            children = graph_adj.get(current, [])
            for c in children:
                if c not in visited:
                    visited.add(c)
                    queue.append(c)
    return descendants

# --- *优化点 1*: 高效d-分离 (替换 9.6M 路径搜索) ---

def is_d_separated_bfs(
    start_node: str, 
    end_node: str, 
    given: set, 
    graph: dict, 
    graph_adj: dict, 
    descendants_cache: dict
) -> bool:
    """
    使用 BFS (广度优先搜索) 检查 'start_node' 和 'end_node' 是否被 'given' d-分离。
    这是一种非路径枚举的高效算法 (O(V+E))。
    它通过寻找是否存在任何 "活动路径" (active trail) 来工作。
    """
    
    # 状态元组: (node, direction)
    # direction 'up' 意味着我们从一个子节点到达 'node' (e.g. child <- node)
    # direction 'down' 意味着我们从一个父节点到达 'node' (e.g. parent -> node)
    queue = collections.deque()
    visited = set()

    # 1. 初始化队列：从 start_node 向所有方向出发
    # 向下 (-> children)
    for child in get_children(graph_adj, start_node):
        queue.append((child, 'down'))
        visited.add((child, 'down'))
    
    # 向上 (-> parents)
    for parent in get_parents(graph, start_node):
        queue.append((parent, 'up'))
        visited.add((parent, 'up'))

    # 2. 开始 BFS 遍历
    while queue:
        current_node, direction = queue.popleft()

        if current_node == end_node:
            # 找到了一个活动路径！它们 *没有* d-分离。
            return False

        # --- 检查路径是否在 current_node 处被阻断 ---
        
        # 检查 current_node 是否是碰撞点且被 "激活"
        is_activated_collider = False
        if current_node in given or given.intersection(descendants_cache.get(current_node, set())):
            is_activated_collider = True

        # --- 根据来的方向，决定下一步如何走 ---
        
        if direction == 'up':
            # 路径形如: ...child <- current_node
            
            # 1. 向上走 (链): ...child <- current_node <- parent
            # 仅当 current_node (非碰撞点) *不在* given 中时，路径才活动
            if current_node not in given:
                for parent in get_parents(graph, current_node):
                    if (parent, 'up') not in visited:
                        visited.add((parent, 'up'))
                        queue.append((parent, 'up'))

            # 2. 向下走 (分叉): ...child <- current_node -> other_child
            # 仅当 current_node (非碰撞点) *不在* given 中时，路径才活动
            if current_node not in given:
                for child in get_children(graph_adj, current_node):
                    if (child, 'down') not in visited:
                        visited.add((child, 'down'))
                        queue.append((child, 'down'))

        elif direction == 'down':
            # 路径形如: ...parent -> current_node
            
            # 1. 向上走 (碰撞): ...parent -> current_node <- other_parent
            # 仅当 current_node (碰撞点) *被激活* 时，路径才活动
            if is_activated_collider:
                for parent in get_parents(graph, current_node):
                    if (parent, 'up') not in visited:
                        visited.add((parent, 'up'))
                        queue.append((parent, 'up'))

            # 2. 向下走 (链): ...parent -> current_node -> child
            # 仅当 current_node (非碰撞点) *不在* given 中时，路径才活动
            if current_node not in given:
                for child in get_children(graph_adj, current_node):
                    if (child, 'down') not in visited:
                        visited.add((child, 'down'))
                        queue.append((child, 'down'))

    # 队列为空，且从未到达 end_node。
    # 意味着所有路径都被阻断了，它们 *是* d-分离的。
    return True

# --- *优化点 2*: 高效路径存在性检查 ---

def exists_directed_path_without(
    start: str, 
    end: str, 
    block_node: str, 
    graph_adj: dict
) -> bool:
    """
    使用 BFS 检查是否存在从 'start' 到 'end' 的有向路径，
    且该路径 *不* 经过 'block_node'。
    """
    if start == end:
        return True
    
    queue = collections.deque([start])
    visited = {start}

    while queue:
        current = queue.popleft()
        
        for neighbor in graph_adj.get(current, []):
            if neighbor == end:
                return True # 找到了！
            
            if neighbor == block_node:
                continue # 路径被阻断，停止这个分支
                
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
                
    return False # 未找到路径


# --- 核心函数：*已优化* ---

def find_c_total(graph: dict, graph_adj: dict, t: str, y: str) -> set:
    """
    *已优化*：识别全部混杂集 C_Total。
    使用高效的 BFS 路径检查，而非路径枚举。
    """
    an_t = get_ancestors(graph, t)
    an_y = get_ancestors(graph, y)
    potential_confounders = an_t.intersection(an_y)
    
    c_total = set()
    
    for c in potential_confounders:
        # *优化点*：使用 O(V+E) 的 BFS 检查，替换 find_all_directed_paths
        if exists_directed_path_without(c, y, t, graph_adj):
            # 这不是一个纯工具变量，它是一个真正的混杂变量
            c_total.add(c)
            
    return c_total

def find_c_root(graph: dict, c_total: set) -> set:
    """
    识别根混杂集 C_Root。(此函数已足够快，无需更改)
    """
    c_root = set()
    for c in c_total:
        parents_c = get_parents(graph, c)
        if not parents_c.intersection(c_total):
            c_root.add(c)
    return c_root

def find_proximal_set_by_dsep(c_total: set, t: str, graph: dict, graph_adj: dict) -> set:
    """
    *已优化*：使用 d-分离搜索 C_P (近端混杂集)。
    使用高效的 BFS d-分离检查器，并预先缓存后代。
    """
    if not c_total:
        return set() 

    c_total_list = list(c_total)
    
    # *优化点*：预先计算所有相关节点的后代集
    # 我们需要它来检查碰撞点的激活
    nodes_to_cache = set(c_total_list)
    for c in c_total_list:
        # 碰撞点可能在 C_Total 之外，但也需要检查
        # 为 C_Total 中所有节点的父节点也缓存后代
        nodes_to_cache.update(get_parents(graph, c))
        
    descendants_cache = {
        node: get_descendants(graph_adj, node) for node in nodes_to_cache
    }

    # 1. 搜索最小子集 (O(2^N) 搜索是必须的，以保证最小性)
    for k in range(len(c_total_list) + 1):
        # 打印进度，了解外层循环
        # print(f"[优化器] 正在检查大小为 {k} / {len(c_total_list)} 的 C_P 子集...")
        
        for subset in itertools.combinations(c_total_list, k):
            given_set = set(subset) 
            distal_set = c_total - given_set
            
            is_valid_separator = True 
            
            # 2. 检查 d-分离条件
            for c_i in distal_set:
                # *优化点*：使用 O(V+E) 的 BFS 检查，替换路径枚举
                if not is_d_separated_bfs(
                    t, c_i, given_set, graph, graph_adj, descendants_cache
                ):
                    is_valid_separator = False
                    break 
            
            if is_valid_separator:
                print(f"[优化器] 找到最小 C_P (大小 {k})！")
                return given_set
                
    print(f"[优化器] 未找到比 C_Total 更小的集合，返回 C_Total。")
    return c_total 

def find_confounder_sets(graph: dict, t: str = 't', y: str = 'y') -> dict:
    """
    识别给定因果图中的三种混杂集合（d-分离版本）。
    
    :param graph: 因果图 (子 -> [父])
    :param t: 处理变量 (默认为 't')
    :param y: 结果变量 (默认为 'y')
    :return: 包含 'total', 'proximal', 'root' 三个集合的字典
    """
    # 1. 预处理
    graph_adj = invert_graph(graph)
    
    # print("  [优化器] 正在计算 C_Total (使用高效 BFS)...")
    c_total = find_c_total(graph, graph_adj, t, y)
    # print(f"  ... C_Total 计算完毕，大小: {len(c_total)}")

    c_root = find_c_root(graph, c_total)
    
    # print("  [优化器] 正在计算 C_P (使用高效 d-分离 BFS)...")
    c_proximal = find_proximal_set_by_dsep(c_total, t, graph, graph_adj)
    # print("  ... C_P 计算完毕。")

    # 返回排序后的结果，以保证输出一致性
    return {
        'total': sorted(list(c_total)),
        'proximal': sorted(list(c_proximal)),
        'root': sorted(list(c_root))
    }

# --- *新功能*：获取 Y-Model 协变量 (不变) ---
def get_ancestors_at_level(nodes: set, level: int, graph: dict) -> set:
    # (此函数不变，保持原样)
    if level < 1:
        return set()
    current_nodes = nodes
    for _ in range(level):
        next_level_ancestors = set()
        for node in current_nodes:
            next_level_ancestors.update(get_parents(graph, node))
        current_nodes = next_level_ancestors
        if not current_nodes: 
            return set()
    return current_nodes

def get_y_model_covariates(graph: dict, graph_adj: dict, t: str = 't', y: str = 'y') -> tuple:
    # (此函数不变，保持原样)
    warning_message = None
    pa_y = get_parents(graph, y)
    
    if t in pa_y:
        return (pa_y, warning_message)
    
    desc_t = get_descendants(graph_adj, t) 
    an_y = get_ancestors(graph, y)       
    M_all = desc_t.intersection(an_y)      
    
    u_M = pa_y.intersection(M_all)
    Z_base = pa_y - u_M
    
    An_1 = get_ancestors_at_level(u_M, 1, graph)
    if t in An_1:
        final_set = Z_base.union(An_1)
        return (final_set, warning_message)

    An_2 = get_ancestors_at_level(u_M, 2, graph)
    if t in An_2:
        final_set = Z_base.union(An_2)
        return (final_set, warning_message)

    An_3 = get_ancestors_at_level(u_M, 3, graph)
    if t in An_3:
        final_set = Z_base.union(An_3)
        return (final_set, warning_message)

    warning_message = f"警告：策略变量 {t} 与结果 {y} 的直接父节点距离较远（超过3级中介），因果效应可能较弱或路径复杂。"
    final_set = Z_base.union({t})
    return (final_set, warning_message)

# --- 主函数（集成所有功能）(不变) ---
def analyze_causal_graph(graph_name: str, graph: dict, t: str = 't', y: str = 'y'):
    """
    对给定的因果图执行完整的两步法变量选择分析。
    """
    print("-" * 50)
    print(f"开始分析: {graph_name}")
    
    # 0. 绘制因果图
    draw_causal_graph(graph, graphName=graph_name)
    
    # 1. 预处理
    graph_adj = invert_graph(graph)
    
    # --- 第一步：T-Model 变量选择 (去混杂) ---
    print("\n--- 第一步分析 (T-Model：去混杂) ---")
    # print("  [优化器] 正在计算 C_Total (使用高效 BFS)...")
    c_total = find_c_total(graph, graph_adj, t, y)
    # print(f"  ... C_Total 计算完毕，大小: {len(c_total)}")

    c_root = find_c_root(graph, c_total)
    
    # print("  [优化器] 正在计算 C_P (使用高效 d-分离 BFS)...")
    c_proximal = find_proximal_set_by_dsep(c_total, t, graph, graph_adj)
    # print("  ... C_P 计算完毕。")

    confounder_sets = {
        'total': sorted(list(c_total)),
        'proximal': sorted(list(c_proximal)),
        'root': sorted(list(c_root))
    }
    
    print(f"\n全部混杂集 (C_Total): {confounder_sets['total']}")
    print(f"根混杂集 (C_Root):   {confounder_sets['root']}")
    print(f"近端混杂集 (C_P):     {confounder_sets['proximal']}")
    print(f"推荐的 T-Model 调整集 (IPW/PSM): {confounder_sets['proximal']}")
    
    # --- 第二步：Y-Model 变量选择 (反事实估计) ---
    print("\n--- 第二步分析 (Y-Model：反事实估计) ---")
    y_covariates, warning = get_y_model_covariates(graph, graph_adj, t, y)
    y_covariates_f = sorted(list(y_covariates))
    print(f"推荐的 Y-Model 协变量集 (回归): {y_covariates_f}")
    if warning:
        print(f"\n{warning}")
    
    print("-" * 50 + "\n")

    # 5. 返回排序后的结果，以保证输出一致性
    return confounder_sets, y_covariates_f

def detect_parent_confounders(graph, target, conType):
    """
    混杂变量检测函数

    参数:
    - graph: 字典类型，表示因果图，键为变量名，值为父节点列表
    - target: 字符串类型，目标变量名
    - conType : 混杂类型：'total', 'proximal', 'root'

    返回:
    - confounders: 字典类型，键为目标变量的父节点，值为该父节点的混杂变量列表
    """

    # 第一步：构建祖先关系映射表
    # ancestor_map将存储每个父节点的所有祖先节点
    ancestor_map = {}
    pa_y = get_parents(graph, target)
    
    # 遍历目标变量的所有直接父节点
    for node in tqdm(pa_y, desc="处理父节点", unit="节点"):
        # 保留原有的打印信息
        print('Process node: {}.'.format(node))
        confounder_sets = find_confounder_sets(graph, node, target)
        if conType == 'total':
            ancestor_map[node] = confounder_sets['total']
        elif conType == 'root':
            ancestor_map[node] = confounder_sets['root']
        elif conType == 'proximal':
            ancestor_map[node] = confounder_sets['proximal']
        else:
            print('conType is not set.')
    
    return ancestor_map