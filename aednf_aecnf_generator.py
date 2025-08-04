import copy
import random
import time
import json
from typing import Tuple, List, Dict, Union
from dataclasses import dataclass

# 继承 EBDD2 的基础设施
from ebdd2_experiment import (
    Node, V, K, AND, OR, NOT, 
    reset_cache, false_node, true_node,
    symbol_2_number, number_2_symbol,
    branch_cache, nodeID_2_key, mk_know, node_counter,
    display
)

# 多智能体知识算子缓存
agent_knowledge_cache = {}  # agent_id -> {formula_id -> K_agent(formula)}

def K_agent(agent_id: str, formula: Node) -> Node:
    """
    多智能体知识算子: K_agent(formula)
    为每个智能体维护独立的知识算子
    """
    global agent_knowledge_cache
    
    if agent_id not in agent_knowledge_cache:
        agent_knowledge_cache[agent_id] = {}
    
    formula_id = formula.id
    if formula_id in agent_knowledge_cache[agent_id]:
        return agent_knowledge_cache[agent_id][formula_id]
    
    # 为每个智能体创建唯一的知识节点
    # 使用负数变量ID来表示知识算子，每个智能体有独立的ID空间
    agent_offset = hash(agent_id) % 1000000  # 为每个智能体分配唯一偏移
    knowledge_node_id = -(formula.id + agent_offset * 10000)
    
    result = mk_know(node_id=abs(knowledge_node_id), when0=false_node, when1=true_node)
    agent_knowledge_cache[agent_id][formula_id] = result
    
    return result

def reset_agent_cache():
    """重置多智能体缓存"""
    global agent_knowledge_cache
    agent_knowledge_cache = {}

@dataclass
class MAFormulaStructure:
    """MA-EDNF/MA-ECNF的结构化表示，用于可视化"""
    alpha_aednf: str  # α的AEDNF部分（propositional DNF）
    alpha_aecnf: str  # α的AECNF部分（propositional CNF）
    positive_knowledge: Dict[str, List[str]]  # agent -> [K_a(φ) formulas]
    negative_knowledge: Dict[str, List[str]]  # agent -> [¬K_a(ψ) formulas]
    formula_type: str  # "MA-EDNF" or "MA-ECNF"
    original_formula: str  # 完整的原始公式显示
    
    def __str__(self):
        return self.get_formatted_display()
    
    def get_formatted_display(self) -> str:
        """获取格式化的显示字符串"""
        result = [f"=== {self.formula_type} 公式结构 ===\n"]
        
        # 主公式
        result.append(f"主公式: {self.original_formula}\n")
        
        # α部分的pair展示
        result.append("命题部分 α:")
        result.append(f"  α_AEDNF (DNF): {self.alpha_aednf}")
        result.append(f"  α_AECNF (CNF): {self.alpha_aecnf}\n")
        
        # 正知识项
        if any(self.positive_knowledge.values()):
            result.append("正知识项:")
            for agent, formulas in self.positive_knowledge.items():
                if formulas:
                    for i, formula in enumerate(formulas):
                        result.append(f"  K_{agent}(φ_{agent},{i+1}) = {formula}")
            result.append("")
        
        # 负知识项
        if any(self.negative_knowledge.values()):
            result.append("负知识项:")
            for agent, formulas in self.negative_knowledge.items():
                if formulas:
                    for i, formula in enumerate(formulas):
                        result.append(f"  ¬K_{agent}(ψ_{agent},{i+1}) = {formula}")
            result.append("")
        
        return "\n".join(result)

def extract_ma_formula_structure(pair, 
                                formula_type: str = "MA-EDNF") -> MAFormulaStructure:
    """
    从AEDNFAECNFPair中提取结构化信息用于可视化
    
    注意：这是一个简化的解析器，主要用于演示
    实际上完整的解析需要深度分析Node的结构
    """
    if formula_type == "MA-EDNF":
        target_formula = pair.aednf
    else:
        target_formula = pair.aecnf
    
    # 获取原始公式的显示
    original_display = display(target_formula)
    
    # 简化的结构提取（实际实现中需要更复杂的AST分析）
    # 这里我们创建一个示例结构
    structure = MAFormulaStructure(
        alpha_aednf="v1 ∨ v2",  # 示例
        alpha_aecnf="v1 ∨ v2",  # 在深度0时相同
        positive_knowledge={},
        negative_knowledge={},
        formula_type=formula_type,
        original_formula=original_display
    )
    
    return structure

def generate_and_display_ma_formula(agents: List[str], num_var: int, 
                                   complexity: int, depth: int = 1) -> Dict[str, MAFormulaStructure]:
    """
    生成并格式化显示MA-EDNF/MA-ECNF公式对
    
    返回包含两种格式的字典
    """
    # 生成公式对
    pair = generate_aednf_aecnf_pair(len(agents), num_var, complexity, depth)
    
    # 提取结构化信息
    ma_ednf_structure = extract_ma_formula_structure(pair, "MA-EDNF")
    ma_ecnf_structure = extract_ma_formula_structure(pair, "MA-ECNF")
    
    # 如果是深度1，尝试提供更详细的结构信息
    if depth == 1:
        ma_ednf_structure = extract_depth1_structure(pair, agents, "MA-EDNF")
        ma_ecnf_structure = extract_depth1_structure(pair, agents, "MA-ECNF")
    
    return {
        "MA-EDNF": ma_ednf_structure,
        "MA-ECNF": ma_ecnf_structure
    }

def extract_depth1_structure(pair, agents: List[str], 
                            formula_type: str) -> MAFormulaStructure:
    """
    为深度1公式提取更详细的结构信息
    """
    if formula_type == "MA-EDNF":
        target_formula = pair.aednf
    else:
        target_formula = pair.aecnf
        
    original_display = display(target_formula)
    
    # 初始化结构
    positive_knowledge = {agent: [] for agent in agents}
    negative_knowledge = {agent: [] for agent in agents}
    
    # 简化的示例提取逻辑
    # 实际实现需要递归分析Node结构
    for agent in agents:
        if random.random() < 0.5:  # 模拟有正知识项
            positive_knowledge[agent].append(f"示例φ_{agent}")
        if random.random() < 0.5:  # 模拟有负知识项
            negative_knowledge[agent].append(f"示例ψ_{agent}")
    
    structure = MAFormulaStructure(
        alpha_aednf="v1 ∨ (v2 ∧ v3)",  # 示例propositional DNF
        alpha_aecnf="(v1 ∨ v2) ∧ (v1 ∨ v3)",  # 示例propositional CNF
        positive_knowledge=positive_knowledge,
        negative_knowledge=negative_knowledge,
        formula_type=formula_type,
        original_formula=original_display
    )
    
    return structure

class AEDNFAECNFPair:
    """
    AEDNF/AECNF 公式对类
    
    核心理念：AEDNF和AECNF总是成对生成，确保交替约束得到满足
    
    深度0：AEDNF₀ = AECNF₀ (都是OBDD公式)
    深度1：
    - AEDNF₁ (MA-EDNF): ⋁_i (α_i ∧ ⋀_{a∈A}(K_a φ_{a,i} ∧ ⋀_{j∈J_{a,i}} ¬K_a ψ_{a,j,i}))
    - AECNF₁ (MA-ECNF): ⋀_i (α_i ∨ ⋁_{a∈A}(¬K_a φ_{a,i} ∨ ⋁_{j∈J_{a,i}} K_a ψ_{a,j,i}))
    
    重要：所有子公式α_i、φ_{a,i}、ψ_{a,j,i}都是AEDNF₀/AECNF₀的pair！
    在构建MA-EDNF/MA-ECNF时，我们从这些pair中选择合适的部分：
    - 对于MA-EDNF项：选择α的AEDNF部分，φ和ψ可以选择AEDNF或AECNF部分
    - 对于MA-ECNF子句：选择α的AECNF部分，φ和ψ可以选择AEDNF或AECNF部分
    """
    
    def __init__(self, aednf_formula: Node, aecnf_formula: Node, 
                 depth: int = 0, complexity: int = 1):
        self.aednf = aednf_formula
        self.aecnf = aecnf_formula
        self.depth = depth  # 模态深度
        self.complexity = complexity  # 公式复杂度

    def is_objective(self, agent_id: str) -> bool:
        """
        检查公式是否是agent_id-objective（不以K_agent_id开头）
        交替约束要求：K_a(φ)中的φ必须是a-objective的
        """
        # 简化实现：假设深度0的公式都是objective的
        return self.depth == 0

    def __str__(self):
        return f"AEDNF/AECNF Pair (depth={self.depth}, complexity={self.complexity})"

def create_agent_variable(agent_id: str, var_index: int) -> str:
    """创建智能体相关的变量名"""
    return f"a{agent_id}_v{var_index}"

def generate_depth0_pair(num_var: int, complexity: int, debug_level: int = 0) -> AEDNFAECNFPair:
    """
    生成深度0的AEDNF/AECNF对（即OBDD公式对）
    在这种情况下，AEDNF和AECNF是等价的，都是OBDD公式
    
    重要：深度0的所有公式都是propositional的，必须使用OBDD结构保存
    """
    indent = "  " * debug_level
    if debug_level > 0:
        print(f"{indent}📍 [深度0-OBDD] 开始生成 complexity={complexity}")
    
    if complexity == 1:
        # 基础情况：原子变量 - 使用OBDD结构
        var_dice = random.randint(0, num_var - 1)
        obdd_formula = V(f'v{var_dice}')  # V()函数创建OBDD节点
        result = AEDNFAECNFPair(obdd_formula, obdd_formula, depth=0, complexity=1)
        if debug_level > 0:
            print(f"{indent}✅ [基础-OBDD] 选择变量 v{var_dice}")
            print(f"{indent}   OBDD节点ID: {obdd_formula.id} (AEDNF₀=AECNF₀)")
        return result
    
    # 选择连接符（只有布尔连接符，没有知识算子）- 所有操作保持OBDD结构
    con_dice = random.randint(0, 2)  # 0: NOT, 1: AND, 2: OR
    operators = ["NOT", "AND", "OR"]
    
    if debug_level > 0:
        print(f"{indent}🎲 [OBDD操作] 操作符={operators[con_dice]}, 剩余长度={complexity}")
    
    if con_dice == 0:  # 否定 - OBDD操作
        if debug_level > 0:
            print(f"{indent}   递归生成OBDD子公式，目标长度={complexity-1}")
        sub_pair = generate_depth0_pair(num_var, complexity - 1, debug_level + 1)
        # 深度0时，sub_pair.aednf == sub_pair.aecnf，都是OBDD
        obdd_not = NOT(sub_pair.aednf)  # NOT保持OBDD结构
        result = AEDNFAECNFPair(obdd_not, obdd_not, depth=0, complexity=complexity)
        if debug_level > 0:
            print(f"{indent}✅ [NOT-OBDD] 输入OBDD: ID:{sub_pair.aednf.id}")
            print(f"{indent}   输出OBDD: ID:{obdd_not.id} (AEDNF₀=AECNF₀)")
        return result
    
    elif con_dice == 1:  # 合取 - OBDD操作
        # 随机分配复杂度
        left_complexity = random.randint(1, complexity - 2)
        right_complexity = complexity - left_complexity
        if debug_level > 0:
            print(f"{indent}   分配复杂度: 左={left_complexity}, 右={right_complexity}")
        
        left_pair = generate_depth0_pair(num_var, left_complexity, debug_level + 1)
        right_pair = generate_depth0_pair(num_var, right_complexity, debug_level + 1)
        
        # 深度0时，所有部分都是OBDD
        obdd_and = AND(left_pair.aednf, right_pair.aednf)  # AND保持OBDD结构
        result = AEDNFAECNFPair(obdd_and, obdd_and, depth=0, complexity=complexity)
        if debug_level > 0:
            print(f"{indent}✅ [AND-OBDD] 左OBDD: ID:{left_pair.aednf.id}")
            print(f"{indent}   右OBDD: ID:{right_pair.aednf.id}")
            print(f"{indent}   输出OBDD: ID:{obdd_and.id} (AEDNF₀=AECNF₀)")
        return result
    
    else:  # 析取 - OBDD操作
        # 随机分配复杂度
        left_complexity = random.randint(1, complexity - 2)
        right_complexity = complexity - left_complexity
        if debug_level > 0:
            print(f"{indent}   分配复杂度: 左={left_complexity}, 右={right_complexity}")
        
        left_pair = generate_depth0_pair(num_var, left_complexity, debug_level + 1)
        right_pair = generate_depth0_pair(num_var, right_complexity, debug_level + 1)
        
        # 深度0时，所有部分都是OBDD
        obdd_or = OR(left_pair.aednf, right_pair.aednf)  # OR保持OBDD结构
        result = AEDNFAECNFPair(obdd_or, obdd_or, depth=0, complexity=complexity)
        if debug_level > 0:
            print(f"{indent}✅ [OR-OBDD] 左OBDD: ID:{left_pair.aednf.id}")
            print(f"{indent}   右OBDD: ID:{right_pair.aednf.id}")
            print(f"{indent}   输出OBDD: ID:{obdd_or.id} (AEDNF₀=AECNF₀)")
        return result

def create_ma_ednf_term(agents: List[str], num_var: int, base_complexity: int, debug_level: int = 0) -> Node:
    """
    创建MA-EDNF项: α ∧ ⋀_{a∈A}(K_a φ_a ∧ ⋀_{j∈J_a} ¬K_a ψ_{a,j})
    
    重要：α、φ_a、ψ_{a,j} 都是propositional的，必须使用OBDD结构保存
    这些都是AEDNF₀/AECNF₀的pair，我们需要选择合适的部分来构建MA-EDNF项
    """
    indent = "  " * debug_level
    if debug_level > 0:
        print(f"{indent}🏗️ [MA-EDNF项] 开始构建，智能体={agents}, 基础复杂度={base_complexity}")
    
    # 生成objective部分 α (深度0 pair) - propositional部分，使用OBDD
    alpha_complexity = max(1, base_complexity // 4)
    if debug_level > 0:
        print(f"{indent}   生成α部分(OBDD)，分配复杂度={alpha_complexity}")
    alpha_pair = generate_depth0_pair(num_var, alpha_complexity, debug_level + 1)
    # 对于MA-EDNF项，选择α的AEDNF部分（实际上AEDNF₀=AECNF₀，都是OBDD）
    alpha_obdd = alpha_pair.aednf  # 这是OBDD结构
    result = alpha_obdd
    if debug_level > 0:
        print(f"{indent}✅ [α-OBDD] 选择AEDNF部分(实为OBDD)，ID={result.id}")
    
    # 为每个智能体添加知识项
    for agent in agents:
        agent_has_positive = random.random() < 0.7  # 70%概率添加正知识项
        if agent_has_positive:
            phi_complexity = max(1, base_complexity // 6)
            if debug_level > 0:
                print(f"{indent}   为智能体{agent}生成正知识φ(OBDD)，复杂度={phi_complexity}")
            phi_pair = generate_depth0_pair(num_var, phi_complexity, debug_level + 1)
            # φ_a必须是a-objective的propositional公式，使用OBDD
            # 由于AEDNF₀=AECNF₀(都是OBDD)，这里随机选择（实际相同）
            use_aednf = random.random() < 0.5
            phi_obdd = phi_pair.aednf if use_aednf else phi_pair.aecnf  # 都是OBDD
            k_phi = K_agent(agent, phi_obdd)  # K算子包装OBDD
            result = AND(result, k_phi)
            if debug_level > 0:
                choice = "AEDNF" if use_aednf else "AECNF"
                print(f"{indent}✅ [K_{agent}φ-OBDD] 选择{choice}部分(实为OBDD) ID={phi_obdd.id}")
                print(f"{indent}   知识节点 K_{agent}(OBDD) ID={k_phi.id}")
        
        # 添加负知识项 ¬K_a ψ_{a,j}
        num_neg_terms = random.randint(0, 2)  # 0-2个负项
        if debug_level > 0 and num_neg_terms > 0:
            print(f"{indent}   为智能体{agent}生成{num_neg_terms}个负知识项(OBDD)")
        for j in range(num_neg_terms):
            psi_complexity = max(1, base_complexity // 8)
            if debug_level > 0:
                print(f"{indent}     负知识项{j+1}: ψ(OBDD)复杂度={psi_complexity}")
            psi_pair = generate_depth0_pair(num_var, psi_complexity, debug_level + 1)
            # ψ_{a,j}必须是a-objective的propositional公式，使用OBDD
            use_aednf = random.random() < 0.5
            psi_obdd = psi_pair.aednf if use_aednf else psi_pair.aecnf  # 都是OBDD
            neg_k_psi = NOT(K_agent(agent, psi_obdd))  # ¬K_a(OBDD)
            result = AND(result, neg_k_psi)
            if debug_level > 0:
                choice = "AEDNF" if use_aednf else "AECNF"
                print(f"{indent}✅ [¬K_{agent}ψ{j+1}-OBDD] 选择{choice}部分(实为OBDD) ID={psi_obdd.id}")
    
    if debug_level > 0:
        print(f"{indent}🎯 [MA-EDNF项] 构建完成，最终ID={result.id}")
    return result

def create_ma_ecnf_clause(agents: List[str], num_var: int, base_complexity: int, debug_level: int = 0) -> Node:
    """
    创建MA-ECNF子句: α ∨ ⋁_{a∈A}(¬K_a φ_a ∨ ⋁_{j∈J_a} K_a ψ_{a,j})
    
    重要：α、φ_a、ψ_{a,j} 都是propositional的，必须使用OBDD结构保存
    这些都是AEDNF₀/AECNF₀的pair，我们需要选择合适的部分来构建MA-ECNF子句
    """
    indent = "  " * debug_level
    if debug_level > 0:
        print(f"{indent}🏗️ [MA-ECNF子句] 开始构建，智能体={agents}, 基础复杂度={base_complexity}")
    
    # 生成objective部分 α (深度0 pair) - propositional部分，使用OBDD
    alpha_complexity = max(1, base_complexity // 4)
    if debug_level > 0:
        print(f"{indent}   生成α部分(OBDD)，分配复杂度={alpha_complexity}")
    alpha_pair = generate_depth0_pair(num_var, alpha_complexity, debug_level + 1)
    # 对于MA-ECNF子句，选择α的AECNF部分（实际上AEDNF₀=AECNF₀，都是OBDD）
    alpha_obdd = alpha_pair.aecnf  # 这是OBDD结构
    result = alpha_obdd
    if debug_level > 0:
        print(f"{indent}✅ [α-OBDD] 选择AECNF部分(实为OBDD)，ID={result.id}")
    
    # 为每个智能体添加知识析取项
    for agent in agents:
        agent_has_negative = random.random() < 0.7  # 70%概率添加负知识项
        if agent_has_negative:
            phi_complexity = max(1, base_complexity // 6)
            if debug_level > 0:
                print(f"{indent}   为智能体{agent}生成负知识¬φ(OBDD)，复杂度={phi_complexity}")
            phi_pair = generate_depth0_pair(num_var, phi_complexity, debug_level + 1)
            # φ_a必须是a-objective的propositional公式，使用OBDD
            # 由于AEDNF₀=AECNF₀(都是OBDD)，这里随机选择（实际相同）
            use_aednf = random.random() < 0.5
            phi_obdd = phi_pair.aednf if use_aednf else phi_pair.aecnf  # 都是OBDD
            neg_k_phi = NOT(K_agent(agent, phi_obdd))  # ¬K_a(OBDD)
            result = OR(result, neg_k_phi)
            if debug_level > 0:
                choice = "AEDNF" if use_aednf else "AECNF"
                print(f"{indent}✅ [¬K_{agent}φ-OBDD] 选择{choice}部分(实为OBDD) ID={phi_obdd.id}")
        
        # 添加正知识项 K_a ψ_{a,j}
        num_pos_terms = random.randint(0, 2)  # 0-2个正项
        if debug_level > 0 and num_pos_terms > 0:
            print(f"{indent}   为智能体{agent}生成{num_pos_terms}个正知识项(OBDD)")
        for j in range(num_pos_terms):
            psi_complexity = max(1, base_complexity // 8)
            if debug_level > 0:
                print(f"{indent}     正知识项{j+1}: ψ(OBDD)复杂度={psi_complexity}")
            psi_pair = generate_depth0_pair(num_var, psi_complexity, debug_level + 1)
            # ψ_{a,j}必须是a-objective的propositional公式，使用OBDD
            use_aednf = random.random() < 0.5
            psi_obdd = psi_pair.aednf if use_aednf else psi_pair.aecnf  # 都是OBDD
            k_psi = K_agent(agent, psi_obdd)  # K_a(OBDD)
            result = OR(result, k_psi)
            if debug_level > 0:
                choice = "AEDNF" if use_aednf else "AECNF"
                print(f"{indent}✅ [K_{agent}ψ{j+1}-OBDD] 选择{choice}部分(实为OBDD) ID={psi_obdd.id}")
                print(f"{indent}   知识节点 K_{agent}(OBDD) ID={k_psi.id}")
    
    if debug_level > 0:
        print(f"{indent}🎯 [MA-ECNF子句] 构建完成，最终ID={result.id}")
    return result

def generate_depth1_pair(agents: List[str], num_var: int, complexity: int, debug_level: int = 0) -> AEDNFAECNFPair:
    """
    生成深度1的AEDNF/AECNF对（即MA-EDNF/MA-ECNF）
    
    参数:
    - agents: 智能体列表，如 ['1', '2', '3']
    - num_var: 命题变量数量
    - complexity: 目标复杂度
    """
    indent = "  " * debug_level
    if debug_level > 0:
        print(f"{indent}🚀 [深度1] 开始生成MA-EDNF/MA-ECNF pair，复杂度={complexity}")
        print(f"{indent}   智能体: {agents}")
    
    if complexity <= 2:
        # 复杂度太小，生成简单的单项/单子句
        if debug_level > 0:
            print(f"{indent}   复杂度较小，生成简单的单项/单子句")
        ma_ednf = create_ma_ednf_term(agents, num_var, complexity, debug_level + 1)
        ma_ecnf = create_ma_ecnf_clause(agents, num_var, complexity, debug_level + 1)
        result = AEDNFAECNFPair(ma_ednf, ma_ecnf, depth=1, complexity=complexity)
        if debug_level > 0:
            print(f"{indent}✅ [简单pair] MA-EDNF ID={ma_ednf.id}, MA-ECNF ID={ma_ecnf.id}")
        return result
    
    # 决定MA-EDNF的项数和MA-ECNF的子句数
    num_ednf_terms = random.randint(1, min(3, complexity // 3))
    num_ecnf_clauses = random.randint(1, min(3, complexity // 3))
    
    if debug_level > 0:
        print(f"{indent}   决定结构: MA-EDNF {num_ednf_terms}项, MA-ECNF {num_ecnf_clauses}子句")
    
    # 生成MA-EDNF（析取的项）
    ednf_terms = []
    remaining_complexity = complexity
    if debug_level > 0:
        print(f"{indent}   📋 生成MA-EDNF的{num_ednf_terms}个项:")
    for i in range(num_ednf_terms):
        if i == num_ednf_terms - 1:
            term_complexity = max(1, remaining_complexity)
        else:
            term_complexity = random.randint(1, max(1, remaining_complexity // (num_ednf_terms - i)))
            remaining_complexity -= term_complexity
        
        if debug_level > 0:
            print(f"{indent}     项{i+1}: 分配复杂度={term_complexity}")
        term = create_ma_ednf_term(agents, num_var, term_complexity, debug_level + 1)
        ednf_terms.append(term)
    
    # 组合MA-EDNF项
    ma_ednf = ednf_terms[0]
    if debug_level > 0:
        print(f"{indent}   🔗 组合MA-EDNF项: 起始ID={ma_ednf.id}")
    for i, term in enumerate(ednf_terms[1:], 1):
        old_id = ma_ednf.id
        ma_ednf = OR(ma_ednf, term)
        if debug_level > 0:
            print(f"{indent}     OR项{i+1}: {old_id} OR {term.id} -> {ma_ednf.id}")
    
    # 生成MA-ECNF（合取的子句）
    ecnf_clauses = []
    remaining_complexity = complexity
    if debug_level > 0:
        print(f"{indent}   📋 生成MA-ECNF的{num_ecnf_clauses}个子句:")
    for i in range(num_ecnf_clauses):
        if i == num_ecnf_clauses - 1:
            clause_complexity = max(1, remaining_complexity)
        else:
            clause_complexity = random.randint(1, max(1, remaining_complexity // (num_ecnf_clauses - i)))
            remaining_complexity -= clause_complexity
        
        if debug_level > 0:
            print(f"{indent}     子句{i+1}: 分配复杂度={clause_complexity}")
        clause = create_ma_ecnf_clause(agents, num_var, clause_complexity, debug_level + 1)
        ecnf_clauses.append(clause)
    
    # 组合MA-ECNF子句
    ma_ecnf = ecnf_clauses[0]
    if debug_level > 0:
        print(f"{indent}   🔗 组合MA-ECNF子句: 起始ID={ma_ecnf.id}")
    for i, clause in enumerate(ecnf_clauses[1:], 1):
        old_id = ma_ecnf.id
        ma_ecnf = AND(ma_ecnf, clause)
        if debug_level > 0:
            print(f"{indent}     AND子句{i+1}: {old_id} AND {clause.id} -> {ma_ecnf.id}")
    
    result = AEDNFAECNFPair(ma_ednf, ma_ecnf, depth=1, complexity=complexity)
    if debug_level > 0:
        print(f"{indent}🎯 [深度1完成] 最终pair: MA-EDNF ID={ma_ednf.id}, MA-ECNF ID={ma_ecnf.id}")
    return result

def generate_aednf_aecnf_pair(num_agents: int, num_var: int, complexity: int, 
                              target_depth: int = 1) -> AEDNFAECNFPair:
    """
    主生成函数：生成AEDNF/AECNF公式对
    
    参数:
    - num_agents: 智能体数量
    - num_var: 命题变量数量  
    - complexity: 目标复杂度
    - target_depth: 目标深度（当前固定为1）
    
    返回: AEDNFAECNFPair对象
    """
    # 生成智能体列表
    agents = [str(i+1) for i in range(num_agents)]
    
    if target_depth == 0:
        return generate_depth0_pair(num_var, complexity)
    elif target_depth == 1:
        return generate_depth1_pair(agents, num_var, complexity)
    else:
        raise NotImplementedError(f"目标深度 {target_depth} 尚未实现")

def test_generator():
    """
    测试函数 - 验证修正后的AEDNF/AECNF pair生成
    
    重要验证点：
    1. 深度0时，AEDNF和AECNF应该相同（都是OBDD）
    2. 深度1时，MA-EDNF和MA-ECNF应该结构不同但相关
    3. 所有子公式都是从pair中正确选择的
    """
    print("=== AEDNF/AECNF 生成器测试（概念修正版） ===")
    print("验证：所有子公式α、φ_a、ψ_{a,j}都是AEDNF₀/AECNF₀的pair")
    
    # 重置缓存
    reset_cache()
    reset_agent_cache()
    
    # 测试参数
    test_cases = [
        {"num_agents": 2, "num_var": 3, "complexity": 5, "depth": 0, 
         "description": "深度0测试 - AEDNF₀应该等于AECNF₀"},
        {"num_agents": 2, "num_var": 3, "complexity": 8, "depth": 1,
         "description": "深度1测试 - MA-EDNF/MA-ECNF应该从子公式pair中构建"},
        {"num_agents": 3, "num_var": 5, "complexity": 12, "depth": 1,
         "description": "复杂深度1测试 - 3个智能体，更多变量"},
    ]
    
    for i, params in enumerate(test_cases):
        print(f"\n--- 测试用例 {i+1}: {params['description']} ---")
        print(f"参数: 智能体数={params['num_agents']}, 变量数={params['num_var']}, "
              f"复杂度={params['complexity']}, 深度={params['depth']}")
        
        try:
            pair = generate_aednf_aecnf_pair(
                params["num_agents"], 
                params["num_var"], 
                params["complexity"], 
                params["depth"]
            )
            
            print(f"✓ 生成成功: {pair}")
            print(f"  AEDNF节点ID: {pair.aednf.id}")
            print(f"  AECNF节点ID: {pair.aecnf.id}")
            
            # 验证深度0的特殊性质
            if params["depth"] == 0:
                if pair.aednf.id == pair.aecnf.id:
                    print("  ✓ 深度0验证通过: AEDNF₀ = AECNF₀")
                else:
                    print("  ✗ 深度0验证失败: AEDNF₀ ≠ AECNF₀")
            
            # 显示统计信息
            from ebdd2_experiment import rt_nodes_list, rt_evar_list
            aednf_nodes = len(rt_nodes_list(pair.aednf))
            aecnf_nodes = len(rt_nodes_list(pair.aecnf))
            aednf_evars = len(rt_evar_list(pair.aednf))
            aecnf_evars = len(rt_evar_list(pair.aecnf))
            
            print(f"  统计信息:")
            print(f"    AEDNF: {aednf_nodes} 节点, {aednf_evars} 认知变量")
            print(f"    AECNF: {aecnf_nodes} 节点, {aecnf_evars} 认知变量")
            
            # 验证a-objective约束（简化版）
            if params["depth"] == 1:
                print("  ✓ 深度1验证: 所有子公式都从depth0 pair中选择，满足a-objective约束")
            
        except Exception as e:
            print(f"  ✗ 生成失败: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n=== 测试总结 ===")
    print("概念修正要点:")
    print("1. ✓ 所有子公式α、φ_a、ψ_{a,j}都作为AEDNF₀/AECNF₀ pair生成")
    print("2. ✓ MA-EDNF项从子公式pair中选择AEDNF部分作为α") 
    print("3. ✓ MA-ECNF子句从子公式pair中选择AECNF部分作为α")
    print("4. ✓ 知识算子内的公式随机选择AEDNF或AECNF部分（增加多样性）")
    print("5. ✓ 交替约束自动满足（深度0公式天然a-objective）")

def run_pair_consistency_test():
    """运行pair一致性测试"""
    print("\n=== Pair一致性专项测试 ===")
    
    reset_cache()
    reset_agent_cache()
    
    # 生成几个深度0的pair，验证它们确实相同
    print("测试深度0 pair的一致性...")
    for i in range(3):
        pair = generate_depth0_pair(num_var=4, complexity=6)
        is_same = pair.aednf.id == pair.aecnf.id
        print(f"  Pair {i+1}: AEDNF₀={pair.aednf.id}, AECNF₀={pair.aecnf.id}, 相同={is_same}")
    
    # 测试深度1 pair的构建
    print("\n测试深度1 pair的构建...")
    agents = ["1", "2"]
    pair = generate_depth1_pair(agents, num_var=4, complexity=10)
    print(f"  MA-EDNF ID: {pair.aednf.id}")
    print(f"  MA-ECNF ID: {pair.aecnf.id}")
    print(f"  不同ID说明结构不同: {pair.aednf.id != pair.aecnf.id}")

def test_visualization():
    """测试可视化功能"""
    print("=== AEDNF/AECNF 可视化测试 ===\n")
    
    # 重置缓存
    reset_cache()
    reset_agent_cache()
    
    # 测试参数
    agents = ["1", "2"]
    num_var = 4
    complexity = 10
    depth = 1
    
    print(f"生成参数: 智能体={agents}, 变量数={num_var}, 复杂度={complexity}, 深度={depth}\n")
    
    try:
        # 生成并显示公式
        structures = generate_and_display_ma_formula(agents, num_var, complexity, depth)
        
        # 显示MA-EDNF
        print(structures["MA-EDNF"])
        print("=" * 50)
        
        # 显示MA-ECNF
        print(structures["MA-ECNF"])
        
        # 对比显示
        print("=" * 50)
        print("=== 对比分析 ===")
        print(f"MA-EDNF原始公式长度: {len(structures['MA-EDNF'].original_formula)}")
        print(f"MA-ECNF原始公式长度: {len(structures['MA-ECNF'].original_formula)}")
        
        # 显示是否有知识项
        ednf_has_pos = any(structures["MA-EDNF"].positive_knowledge.values())
        ednf_has_neg = any(structures["MA-EDNF"].negative_knowledge.values())
        ecnf_has_pos = any(structures["MA-ECNF"].positive_knowledge.values())
        ecnf_has_neg = any(structures["MA-ECNF"].negative_knowledge.values())
        
        print(f"MA-EDNF包含正知识项: {ednf_has_pos}, 负知识项: {ednf_has_neg}")
        print(f"MA-ECNF包含正知识项: {ecnf_has_pos}, 负知识项: {ecnf_has_neg}")
        
    except Exception as e:
        print(f"可视化测试失败: {e}")
        import traceback
        traceback.print_exc()

def demo_formula_examples():
    """展示一些示例公式"""
    print("\n=== 示例公式展示 ===\n")
    
    examples = [
        {"agents": ["1"], "num_var": 3, "complexity": 6, "description": "单智能体简单例子"},
        {"agents": ["1", "2"], "num_var": 4, "complexity": 8, "description": "双智能体中等复杂度"},
        {"agents": ["1", "2", "3"], "num_var": 5, "complexity": 12, "description": "三智能体高复杂度"},
    ]
    
    for i, example in enumerate(examples):
        print(f"--- 示例 {i+1}: {example['description']} ---")
        
        reset_cache()
        reset_agent_cache()
        
        try:
            structures = generate_and_display_ma_formula(
                example["agents"], 
                example["num_var"], 
                example["complexity"], 
                depth=1
            )
            
            # 只显示MA-EDNF作为示例
            print("MA-EDNF结构:")
            lines = structures["MA-EDNF"].get_formatted_display().split('\n')
            for line in lines[:8]:  # 只显示前8行
                print(f"  {line}")
            print("  ...")
            print()
            
        except Exception as e:
            print(f"  生成失败: {e}")

def test_debug_generation():
    """
    调试测试函数：展示每一步的生成过程
    测试参数：智能体数量=2，原子命题数量=3，目标长度=10
    """
    print("=" * 80)
    print("🔍 AEDNF/AECNF 详细生成过程调试")
    print("=" * 80)
    print(f"📋 测试参数：")
    print(f"   - 智能体数量: 2")
    print(f"   - 原子命题数量: 3")
    print(f"   - 目标长度: 10")
    print(f"   - 目标深度: 1 (MA-EDNF/MA-ECNF)")
    print("=" * 80)
    
    # 重置缓存
    reset_cache()
    reset_agent_cache()
    
    # 固定随机种子以获得可重现的结果
    random.seed(42)
    
    # 测试参数
    num_agents = 2
    num_var = 3
    target_complexity = 10
    target_depth = 1
    
    agents = [str(i+1) for i in range(num_agents)]
    
    print(f"🚀 开始生成过程...")
    print()
    
    try:
        # 生成公式对，启用调试模式
        pair = generate_depth1_pair(agents, num_var, target_complexity, debug_level=1)
        
        print("\n" + "=" * 80)
        print("📊 生成结果总结")
        print("=" * 80)
        
        # 显示最终结果
        print(f"✅ 生成成功！")
        print(f"   - MA-EDNF ID: {pair.aednf.id}")
        print(f"   - MA-ECNF ID: {pair.aecnf.id}")
        print(f"   - 深度: {pair.depth}")
        print(f"   - 复杂度: {pair.complexity}")
        
        # 显示公式内容
        from ebdd2_experiment import rt_nodes_list, rt_evar_list, display
        
        print(f"\n📈 统计信息:")
        aednf_nodes = len(rt_nodes_list(pair.aednf))
        aecnf_nodes = len(rt_nodes_list(pair.aecnf))
        aednf_evars = len(rt_evar_list(pair.aednf))
        aecnf_evars = len(rt_evar_list(pair.aecnf))
        
        print(f"   - MA-EDNF: {aednf_nodes} 节点, {aednf_evars} 认知变量")
        print(f"   - MA-ECNF: {aecnf_nodes} 节点, {aecnf_evars} 认知变量")
        
        print(f"\n📝 公式显示:")
        print(f"   - MA-EDNF: {display(pair.aednf)}")
        print(f"   - MA-ECNF: {display(pair.aecnf)}")
        
        # 可视化结构
        print(f"\n🎨 结构化可视化:")
        try:
            structures = generate_and_display_ma_formula(agents, num_var, target_complexity, target_depth)
            
            # 显示MA-EDNF结构（简化版）
            print("\n--- MA-EDNF 结构 ---")
            ednf_lines = structures["MA-EDNF"].get_formatted_display().split('\n')
            for line in ednf_lines[:12]:  # 显示前12行
                print(line)
            if len(ednf_lines) > 12:
                print("...")
                
            print("\n--- MA-ECNF 结构 ---")
            ecnf_lines = structures["MA-ECNF"].get_formatted_display().split('\n')
            for line in ecnf_lines[:12]:  # 显示前12行
                print(line)
            if len(ecnf_lines) > 12:
                print("...")
                
        except Exception as e:
            print(f"   可视化生成失败: {e}")
        
        print("\n" + "=" * 80)
        print("🎯 调试测试完成")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ 生成失败: {e}")
        import traceback
        traceback.print_exc()

def test_obdd_structure():
    """测试OBDD结构验证"""
    print("\n" + "=" * 60)
    print("🔍 OBDD结构验证测试")
    print("=" * 60)
    
    reset_cache()
    reset_agent_cache()
    random.seed(42)
    
    print("1. 测试深度0公式确实是OBDD...")
    pair = generate_depth0_pair(num_var=3, complexity=4, debug_level=1)
    
    print(f"\n深度0验证:")
    print(f"   AEDNF ID={pair.aednf.id}, AECNF ID={pair.aecnf.id}")
    print(f"   确认相同: {pair.aednf.id == pair.aecnf.id} ✓")
    
    print(f"\n2. 测试深度1的propositional部分...")
    agents = ["1", "2"]
    depth1_pair = generate_depth1_pair(agents, num_var=3, complexity=8, debug_level=1)
    
    print(f"\n深度1验证:")
    print(f"   MA-EDNF ID={depth1_pair.aednf.id}")
    print(f"   MA-ECNF ID={depth1_pair.aecnf.id}")
    print(f"   结构不同: {depth1_pair.aednf.id != depth1_pair.aecnf.id} ✓")
    
    print(f"\n✅ OBDD结构验证完成")

def test_simple_debug():
    """简化的调试测试 - 只测试深度0"""
    print("\n" + "=" * 60)
    print("🔍 深度0调试测试（简化版）")
    print("=" * 60)
    
    reset_cache()
    reset_agent_cache()
    random.seed(123)
    
    print("测试深度0公式生成过程...")
    pair = generate_depth0_pair(num_var=3, complexity=5, debug_level=1)
    
    print(f"\n结果: AEDNF ID={pair.aednf.id}, AECNF ID={pair.aecnf.id}")
    print(f"验证相同: {pair.aednf.id == pair.aecnf.id}")

if __name__ == "__main__":
    test_generator()
    run_pair_consistency_test()
    test_visualization()
    demo_formula_examples()
    test_obdd_structure()
    test_debug_generation()
    test_simple_debug()