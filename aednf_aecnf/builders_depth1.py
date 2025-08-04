"""
AEDNF/AECNF深度1操作构建器
支持多代理认知算子，输入输出都是AEDNF/AECNF对
"""

from typing import List, Tuple, Set
from .models import (
    ObjectiveFormula, KnowledgeLiteral, AEDNFTerm, AECNFClause, 
    AEDNF, AECNF, AEDNFAECNFPair, create_objective_pair
)
from .obdd import Node, OBDDBuilder, AND, OR, NOT, implies, display_traditional, reset_cache, mk_know, false_node, true_node

# 导入深度0构建器
from .builders import Depth0Builder

class Depth1Builder:
    """
    深度1的AEDNF/AECNF操作构建器
    
    支持：
    1. 深度0的客观公式操作（land, lor, lnot等）
    2. 深度1的认知算子 K_a(φ)
    3. 多代理支持
    4. 输入输出都是AEDNF/AECNF对
    """
    
    def __init__(self, agents: Set[str] = None):
        """
        初始化构建器
        
        Args:
            agents: 代理集合，例如 {'a1', 'a2', 'a3'}
        """
        self.agents = agents or {'a1', 'a2', 'a3', 'a4'}
        self.depth0_builder = Depth0Builder()
    
    def create_atom(self, var_name: str) -> AEDNFAECNFPair:
        """创建原子变量（深度0）"""
        return self.depth0_builder.create_atom(var_name)
    
    def create_true(self) -> AEDNFAECNFPair:
        """创建真值⊤（深度0）"""
        return self.depth0_builder.create_true()
    
    def create_false(self) -> AEDNFAECNFPair:
        """创建假值⊥（深度0）"""
        return self.depth0_builder.create_false()
    
    def land(self, pair1: AEDNFAECNFPair, pair2: AEDNFAECNFPair) -> AEDNFAECNFPair:
        """
        逻辑合取 (AND)
        
        如果输入都是深度0，使用深度0构建器
        如果输入包含深度1，需要特殊处理
        """
        # 检查深度
        if pair1.aednf.depth == 0 and pair2.aednf.depth == 0:
            return self.depth0_builder.land(pair1, pair2)
        else:
            # 深度1的AND操作
            return self._land_depth1(pair1, pair2)
    
    def lor(self, pair1: AEDNFAECNFPair, pair2: AEDNFAECNFPair) -> AEDNFAECNFPair:
        """
        逻辑析取 (OR)
        
        如果输入都是深度0，使用深度0构建器
        如果输入包含深度1，需要特殊处理
        """
        # 检查深度
        if pair1.aednf.depth == 0 and pair2.aednf.depth == 0:
            return self.depth0_builder.lor(pair1, pair2)
        else:
            # 深度1的OR操作
            return self._lor_depth1(pair1, pair2)
    
    def lnot(self, pair: AEDNFAECNFPair) -> AEDNFAECNFPair:
        """
        逻辑否定 (NOT)
        
        如果输入是深度0，使用深度0构建器
        如果输入是深度1，需要特殊处理
        """
        if pair.aednf.depth == 0:
            return self.depth0_builder.lnot(pair)
        else:
            # 深度1的NOT操作
            return self._lnot_depth1(pair)
    
    def lknow(self, pair: AEDNFAECNFPair, agent: str) -> AEDNFAECNFPair:
        """
        认知算子 K_a(φ)
        
        Args:
            pair: 深度0的AEDNF/AECNF对
            agent: 代理名称，必须在agents集合中
        """
        if agent not in self.agents:
            raise ValueError(f"代理 {agent} 不在代理集合 {self.agents} 中")
        
        if pair.aednf.depth != 0:
            raise ValueError("K算子只能应用于深度0的公式")
        
        # 创建知识文字
        knowledge_literal = KnowledgeLiteral(
            agent=agent,
            formula=pair.aednf,  # 使用AEDNF作为子公式
            negated=False
        )
        
        # K_a(φ) 在AEDNF中表示为：K_a(φ)（客观部分为真，不影响逻辑）
        aednf_term = AEDNFTerm(
            objective_part=ObjectiveFormula(obdd_node_id=true_node.id, description="⊤"),
            positive_literals=[knowledge_literal],
            negative_literals=[]
        )
        
        # K_a(φ) 在AECNF中表示为：K_a(φ)（客观部分为真，不影响逻辑）
        aecnf_clause = AECNFClause(
            objective_part=ObjectiveFormula(obdd_node_id=true_node.id, description="⊤"),
            positive_literals=[knowledge_literal],
            negative_literals=[]
        )
        
        # 创建深度1的AEDNF和AECNF
        aednf = AEDNF(terms=[aednf_term], depth=1)
        aecnf = AECNF(clauses=[aecnf_clause], depth=1)
        
        return AEDNFAECNFPair(aednf=aednf, aecnf=aecnf)
    
    def lknow_not(self, pair: AEDNFAECNFPair, agent: str) -> AEDNFAECNFPair:
        """
        认知算子 ¬K_a(φ)
        
        Args:
            pair: 深度0的AEDNF/AECNF对
            agent: 代理名称
        """
        if agent not in self.agents:
            raise ValueError(f"代理 {agent} 不在代理集合 {self.agents} 中")
        
        if pair.aednf.depth != 0:
            raise ValueError("¬K算子只能应用于深度0的公式")
        
        # 创建否定的知识文字
        knowledge_literal = KnowledgeLiteral(
            agent=agent,
            formula=pair.aednf,
            negated=True
        )
        
        # ¬K_a(φ) 在AEDNF中表示为：¬K_a(φ)
        aednf_term = AEDNFTerm(
            objective_part=ObjectiveFormula(obdd_node_id=false_node.id, description="⊥"),  # 客观部分为假
            positive_literals=[],
            negative_literals=[knowledge_literal]  # ¬K_a(φ)
        )
        
        # ¬K_a(φ) 在AECNF中表示为：¬K_a(φ)
        aecnf_clause = AECNFClause(
            objective_part=ObjectiveFormula(obdd_node_id=false_node.id, description="⊥"),  # 客观部分为假
            positive_literals=[],
            negative_literals=[knowledge_literal]  # ¬K_a(φ)
        )
        
        # 创建深度1的AEDNF和AECNF
        aednf = AEDNF(terms=[aednf_term], depth=1)
        aecnf = AECNF(clauses=[aecnf_clause], depth=1)
        
        return AEDNFAECNFPair(aednf=aednf, aecnf=aecnf)
    
    def _land_depth1(self, pair1: AEDNFAECNFPair, pair2: AEDNFAECNFPair) -> AEDNFAECNFPair:
        """
        深度1的AND操作
        
        AEDNF: (α₁ ∧ ⋀K_a φ_a) ∧ (α₂ ∧ ⋀K_b φ_b) = (α₁∧α₂) ∧ ⋀(K_a φ_a ∧ K_b φ_b)
        AECNF: (α₁ ∨ ⋁¬K_a φ_a) ∧ (α₂ ∨ ⋁¬K_b φ_b) = (α₁∧α₂) ∨ ⋁(¬K_a φ_a ∨ ¬K_b φ_b)
        """
        new_terms = []
        new_clauses = []
        
        # 处理AEDNF：合并所有项
        for term1 in pair1.aednf.terms:
            for term2 in pair2.aednf.terms:
                # 合并客观部分
                obj_node1 = self._get_obdd_node_from_term(term1)
                obj_node2 = self._get_obdd_node_from_term(term2)
                combined_obj = AND(obj_node1, obj_node2)
                
                # 合并知识文字：所有知识文字都保留
                combined_literals = term1.positive_literals + term2.positive_literals
                combined_neg_literals = term1.negative_literals + term2.negative_literals
                
                new_term = AEDNFTerm(
                    objective_part=ObjectiveFormula(obdd_node_id=combined_obj.id, description=""),
                    positive_literals=combined_literals,
                    negative_literals=combined_neg_literals
                )
                new_terms.append(new_term)
        
        # 处理AECNF：合并所有子句
        for clause1 in pair1.aecnf.clauses:
            for clause2 in pair2.aecnf.clauses:
                # 合并客观部分
                obj_node1 = self._get_obdd_node_from_clause(clause1)
                obj_node2 = self._get_obdd_node_from_clause(clause2)
                combined_obj = AND(obj_node1, obj_node2)
                
                # 合并知识文字：所有知识文字都保留
                combined_literals = clause1.positive_literals + clause2.positive_literals
                combined_neg_literals = clause1.negative_literals + clause2.negative_literals
                
                new_clause = AECNFClause(
                    objective_part=ObjectiveFormula(obdd_node_id=combined_obj.id, description=""),
                    positive_literals=combined_literals,
                    negative_literals=combined_neg_literals
                )
                new_clauses.append(new_clause)
        
        return AEDNFAECNFPair(
            aednf=AEDNF(terms=new_terms, depth=1),
            aecnf=AECNF(clauses=new_clauses, depth=1)
        )
    
    def _lor_depth1(self, pair1: AEDNFAECNFPair, pair2: AEDNFAECNFPair) -> AEDNFAECNFPair:
        """
        深度1的OR操作
        
        AEDNF: (α₁ ∧ ⋀K_a φ_a) ∨ (α₂ ∧ ⋀K_b φ_b) = (α₁∧⋀K_a φ_a) ∨ (α₂∧⋀K_b φ_b)
        AECNF: (α₁ ∨ ⋁¬K_a φ_a) ∨ (α₂ ∨ ⋁¬K_b φ_b) = (α₁∨α₂) ∨ ⋁(¬K_a φ_a ∨ ¬K_b φ_b)
        """
        # AEDNF：直接合并项（析取）
        new_terms = pair1.aednf.terms + pair2.aednf.terms
        
        # AECNF：合并客观部分，合并知识文字
        new_clauses = []
        
        # 合并所有客观部分
        all_obj_nodes = []
        for clause in pair1.aecnf.clauses:
            obj_node = self._get_obdd_node_from_clause(clause)
            all_obj_nodes.append(obj_node)
        for clause in pair2.aecnf.clauses:
            obj_node = self._get_obdd_node_from_clause(clause)
            all_obj_nodes.append(obj_node)
        
        # 合并客观部分（析取）
        if all_obj_nodes:
            combined_obj = all_obj_nodes[0]
            for obj_node in all_obj_nodes[1:]:
                combined_obj = OR(combined_obj, obj_node)
        else:
            from .obdd import false_node
            combined_obj = false_node
        
        # 合并所有知识文字
        all_pos_literals = []
        all_neg_literals = []
        for clause in pair1.aecnf.clauses:
            all_pos_literals.extend(clause.positive_literals)
            all_neg_literals.extend(clause.negative_literals)
        for clause in pair2.aecnf.clauses:
            all_pos_literals.extend(clause.positive_literals)
            all_neg_literals.extend(clause.negative_literals)
        
        # 创建合并后的子句
        new_clause = AECNFClause(
            objective_part=ObjectiveFormula(obdd_node_id=combined_obj.id, description=""),
            positive_literals=all_pos_literals,
            negative_literals=all_neg_literals
        )
        new_clauses.append(new_clause)
        
        return AEDNFAECNFPair(
            aednf=AEDNF(terms=new_terms, depth=1),
            aecnf=AECNF(clauses=new_clauses, depth=1)
        )
    
    def _lnot_depth1(self, pair: AEDNFAECNFPair) -> AEDNFAECNFPair:
        """
        深度1的NOT操作
        
        根据德摩根定律：
        ¬(α ∧ ⋀K_a φ_a) = ¬α ∨ ⋁¬K_a φ_a
        ¬(α ∨ ⋁¬K_a φ_a) = ¬α ∧ ⋀K_a φ_a
        """
        # 对于深度1，我们需要：
        # 1. 否定客观部分
        # 2. 对每个知识文字取反
        
        new_terms = []
        new_clauses = []
        
        # 处理AEDNF项（变成AECNF子句）
        for term in pair.aednf.terms:
            # 否定客观部分
            obj_node = self._get_obdd_node_from_term(term)
            neg_obj_node = NOT(obj_node)
            
            # 否定所有知识文字，正确分配到positive和negative
            pos_literals = []
            neg_literals = []
            for lit in term.positive_literals:
                # 肯定的变成否定的
                neg_literals.append(KnowledgeLiteral(
                    agent=lit.agent,
                    formula=lit.formula,
                    negated=True
                ))
            for lit in term.negative_literals:
                # 否定的变成肯定的
                pos_literals.append(KnowledgeLiteral(
                    agent=lit.agent,
                    formula=lit.formula,
                    negated=False
                ))
            
            # 创建AECNF子句
            new_clause = AECNFClause(
                objective_part=ObjectiveFormula(obdd_node_id=neg_obj_node.id, description=""),
                positive_literals=pos_literals,
                negative_literals=neg_literals
            )
            new_clauses.append(new_clause)
        
        # 处理AECNF子句（变成AEDNF项）
        for clause in pair.aecnf.clauses:
            # 否定客观部分
            obj_node = self._get_obdd_node_from_clause(clause)
            neg_obj_node = NOT(obj_node)
            
            # 否定所有知识文字，正确分配到positive和negative
            pos_literals = []
            neg_literals = []
            for lit in clause.positive_literals:
                # 肯定的变成否定的
                neg_literals.append(KnowledgeLiteral(
                    agent=lit.agent,
                    formula=lit.formula,
                    negated=True
                ))
            for lit in clause.negative_literals:
                # 否定的变成肯定的
                pos_literals.append(KnowledgeLiteral(
                    agent=lit.agent,
                    formula=lit.formula,
                    negated=False
                ))
            
            # 创建AEDNF项
            new_term = AEDNFTerm(
                objective_part=ObjectiveFormula(obdd_node_id=neg_obj_node.id, description=""),
                positive_literals=pos_literals,
                negative_literals=neg_literals
            )
            new_terms.append(new_term)
        
        return AEDNFAECNFPair(
            aednf=AEDNF(terms=new_terms, depth=1),
            aecnf=AECNF(clauses=new_clauses, depth=1)
        )
    
    def _get_obdd_node_from_term(self, term: AEDNFTerm) -> Node:
        """从AEDNF项中提取OBDD节点"""
        from .obdd import nodeID_2_key, branch_cache
        key = nodeID_2_key[term.objective_part.obdd_node_id]
        return branch_cache[key]
    
    def _get_obdd_node_from_clause(self, clause: AECNFClause) -> Node:
        """从AECNF子句中提取OBDD节点"""
        from .obdd import nodeID_2_key, branch_cache
        key = nodeID_2_key[clause.objective_part.obdd_node_id]
        return branch_cache[key]
    
    def _get_obdd_node(self, formula) -> Node:
        """从AEDNF/AECNF公式中提取OBDD节点"""
        if formula.depth == 0:
            # 深度0，直接获取客观部分的OBDD节点
            from .obdd import nodeID_2_key, branch_cache
            
            if hasattr(formula, 'terms'):  # AEDNF
                term = formula.terms[0]  # 深度0只有一个项
                key = nodeID_2_key[term.objective_part.obdd_node_id]
            elif hasattr(formula, 'clauses'):  # AECNF
                clause = formula.clauses[0]  # 深度0只有一个子句
                key = nodeID_2_key[clause.objective_part.obdd_node_id]
            else:
                raise ValueError("未知的公式类型")
            
            return branch_cache[key]
        else:
            raise ValueError("此方法只处理深度0的公式") 