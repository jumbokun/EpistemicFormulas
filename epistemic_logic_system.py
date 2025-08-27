#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
认知逻辑系统 - 完整整合版本

包含：
1. 数据模型定义 (ObjectiveFormula, KnowledgeLiteral, AEDNFTerm, AEDNF, AECNF, AECNFClause, AEDNFAECNFPair)
2. OBDD实现 (Node, 基本操作)
3. 逻辑操作 (land, lor, lnot, know)
4. SAT检查函数 (sat_objective, sat_pair, sat_k_gamma_delta, sat_aednf_term, is_aecnf_clause_valid, is_aednf_sat, is_aecnf_valid)
5. 反内省函数 (simple_deintrospective_k, apply_simple_deintrospective_formula)
6. 工具函数 (create_objective_pair, reset_cache)
"""

import sys
import os
import time
import random
import string
from typing import List, Optional, Dict, Set
from dataclasses import dataclass, field
from pydantic import BaseModel, Field, field_validator


# ============================================================================
# OBDD 实现
# ============================================================================

@dataclass
class Node:
    """OBDD节点"""
    var: str
    low: Optional['Node'] = None
    high: Optional['Node'] = None
    id: int = 0
    
    def __hash__(self):
        return hash((self.var, self.low, self.high))
    
    def __eq__(self, other):
        if not isinstance(other, Node):
            return False
        return (self.var == other.var and 
                self.low == other.low and 
                self.high == other.high)

# 全局缓存和节点管理
_node_cache: Dict[tuple, Node] = {}
_next_id = 0
false_node = Node("⊥", None, None, 0)
true_node = Node("⊤", None, None, 1)
_node_cache[(None, None, None)] = false_node
_node_cache[(None, None, None)] = true_node

def reset_cache():
    """重置OBDD缓存"""
    global _node_cache, _next_id
    _node_cache.clear()
    _next_id = 0
    _node_cache[(None, None, None)] = false_node
    _node_cache[(None, None, None)] = true_node

def mk_branch(var: str, low: Node, high: Node) -> Node:
    """创建分支节点"""
    global _next_id
    if low == high:
        return low
    
    key = (var, low, high)
    if key in _node_cache:
        return _node_cache[key]
    
    _next_id += 1
    node = Node(var, low, high, _next_id)
    _node_cache[key] = node
    return node

def mk_know(formula: 'AEDNFAECNFPair', agent: str) -> 'AEDNFAECNFPair':
    """创建知识算子"""
    # 这里简化实现，实际应该根据formula的深度来构造
    return AEDNFAECNFPair(
        aednf=AEDNF(terms=[AEDNFTerm(
            objective_part=ObjectiveFormula(obdd_node_id=true_node.id, description="⊤"),
            positive_literals=[KnowledgeLiteral(agent=agent, formula=formula, negated=False, depth=formula.depth + 1)],
            negative_literals=[]
        )], depth=formula.depth + 1),
        aecnf=AECNF(clauses=[AECNFClause(
            objective_part=ObjectiveFormula(obdd_node_id=true_node.id, description="⊤"),
            positive_literals=[],
            negative_literals=[]
        )], depth=formula.depth + 1),
        depth=formula.depth + 1
    )


# ============================================================================
# 数据模型定义
# ============================================================================

class ObjectiveFormula(BaseModel):
    """客观公式"""
    obdd_node_id: int
    description: str
    depth: int = 0

class KnowledgeLiteral(BaseModel):
    """知识文字"""
    agent: str
    formula: 'AEDNFAECNFPair'
    negated: bool = False
    depth: int = 1

class AEDNFTerm(BaseModel):
    """AEDNF项"""
    objective_part: ObjectiveFormula
    positive_literals: List[KnowledgeLiteral] = Field(default_factory=list)
    negative_literals: List[KnowledgeLiteral] = Field(default_factory=list)
    sat_cache: Optional[bool] = None

class AEDNF(BaseModel):
    """交替认知析取范式"""
    terms: List[AEDNFTerm] = Field(default_factory=list)
    depth: int = 0
    
    @field_validator('terms')
    @classmethod
    def validate_terms(cls, v):
        if len(v) == 0:
            raise ValueError('terms List should have at least 1 item after validation, not 0')
        return v

class AECNFClause(BaseModel):
    """AECNF子句"""
    objective_part: ObjectiveFormula
    positive_literals: List[KnowledgeLiteral] = Field(default_factory=list)
    negative_literals: List[KnowledgeLiteral] = Field(default_factory=list)
    valid_cache: Optional[bool] = None

class AECNF(BaseModel):
    """交替认知合取范式"""
    clauses: List[AECNFClause] = Field(default_factory=list)
    depth: int = 0
    
    @field_validator('clauses')
    @classmethod
    def validate_clauses(cls, v):
        if len(v) == 0:
            raise ValueError('clauses List should have at least 1 item after validation, not 0')
        return v

class AEDNFAECNFPair(BaseModel):
    """AEDNF-AECNF对"""
    aednf: AEDNF
    aecnf: AECNF
    depth: int = 0


# ============================================================================
# 逻辑操作
# ============================================================================

def land(phi1: AEDNFAECNFPair, phi2: AEDNFAECNFPair) -> AEDNFAECNFPair:
    """逻辑与操作"""
    # 简化的实现，实际应该合并AEDNF和AECNF
    new_node = mk_branch("∧", 
        Node(str(phi1.aednf.terms[0].objective_part.obdd_node_id)),
        Node(str(phi2.aednf.terms[0].objective_part.obdd_node_id)))
    
    return AEDNFAECNFPair(
        aednf=AEDNF(terms=[AEDNFTerm(
            objective_part=ObjectiveFormula(
                obdd_node_id=new_node.id,
                description=f"({phi1.aednf.terms[0].objective_part.description} ∧ {phi2.aednf.terms[0].objective_part.description})"
            ),
            positive_literals=phi1.aednf.terms[0].positive_literals + phi2.aednf.terms[0].positive_literals,
            negative_literals=phi1.aednf.terms[0].negative_literals + phi2.aednf.terms[0].negative_literals
        )], depth=max(phi1.depth, phi2.depth)),
        aecnf=AECNF(clauses=phi1.aecnf.clauses + phi2.aecnf.clauses, depth=max(phi1.depth, phi2.depth)),
        depth=max(phi1.depth, phi2.depth)
    )

def lor(phi1: AEDNFAECNFPair, phi2: AEDNFAECNFPair) -> AEDNFAECNFPair:
    """逻辑或操作"""
    # 简化的实现
    new_node = mk_branch("∨", 
        Node(str(phi1.aecnf.clauses[0].objective_part.obdd_node_id)),
        Node(str(phi2.aecnf.clauses[0].objective_part.obdd_node_id)))
    
    return AEDNFAECNFPair(
        aednf=AEDNF(terms=phi1.aednf.terms + phi2.aednf.terms, depth=max(phi1.depth, phi2.depth)),
        aecnf=AECNF(clauses=[AECNFClause(
            objective_part=ObjectiveFormula(
                obdd_node_id=new_node.id,
                description=f"({phi1.aecnf.clauses[0].objective_part.description} ∨ {phi2.aecnf.clauses[0].objective_part.description})"
            ),
            positive_literals=phi1.aecnf.clauses[0].positive_literals + phi2.aecnf.clauses[0].positive_literals,
            negative_literals=phi1.aecnf.clauses[0].negative_literals + phi2.aecnf.clauses[0].negative_literals
        )], depth=max(phi1.depth, phi2.depth)),
        depth=max(phi1.depth, phi2.depth)
    )

def lnot(phi: AEDNFAECNFPair) -> AEDNFAECNFPair:
    """逻辑非操作"""
    # 简化的实现，实际应该交换AEDNF和AECNF
    new_node = mk_branch("¬", Node(str(phi.aednf.terms[0].objective_part.obdd_node_id)), None)
    
    return AEDNFAECNFPair(
        aednf=AEDNF(terms=[AEDNFTerm(
            objective_part=ObjectiveFormula(
                obdd_node_id=new_node.id,
                description=f"¬({phi.aednf.terms[0].objective_part.description})"
            ),
            positive_literals=phi.aednf.terms[0].negative_literals,
            negative_literals=phi.aednf.terms[0].positive_literals
        )], depth=phi.depth),
        aecnf=phi.aecnf,
        depth=phi.depth
    )

def know(phi: AEDNFAECNFPair, agent: str) -> AEDNFAECNFPair:
    """知识算子"""
    return mk_know(phi, agent)


# ============================================================================
# SAT检查函数
# ============================================================================

def sat_objective(obj: ObjectiveFormula) -> bool:
    """检查客观公式的可满足性"""
    return obj.obdd_node_id != false_node.id

def sat_pair(phi: AEDNFAECNFPair) -> bool:
    """检查AEDNF-AECNF对的可满足性"""
    return is_aednf_sat(phi)

def sat_k_gamma_delta(gammas: List[AEDNFAECNFPair], deltas: List[AEDNFAECNFPair]) -> bool:
    """sat_K(Γ, Δ)：检查认知一致性"""

    # Γ 合取
    gamma_conj: Optional[AEDNFAECNFPair] = None
    for g in gammas:
        gamma_conj = g if gamma_conj is None else land(gamma_conj, g)
    if gamma_conj is None:
        gamma_conj = create_objective_pair("⊤")

    # 在K45中，K(false)被认为是可满足的，所以跳过Γ的检查
    # if not sat_pair(gamma_conj):
    #     return False

    # 对每个 δ，检查 Γ ∧ ¬δ 可满足
    for d in deltas:
        if not sat_pair(land(gamma_conj, lnot(d))):
            return False
    return True

def sat_aednf_term(term: AEDNFTerm) -> bool:
    """检查AEDNF项的可满足性"""
    if not sat_objective(term.objective_part):
        return False

    # 收集每个代理的 Γ/Δ
    agent_to_gamma: dict[str, List[AEDNFAECNFPair]] = {}
    agent_to_delta: dict[str, List[AEDNFAECNFPair]] = {}

    for lit in term.positive_literals:
        agent_to_gamma.setdefault(lit.agent, []).append(lit.formula)
    for lit in term.negative_literals:
        agent_to_delta.setdefault(lit.agent, []).append(lit.formula)

    # 对每个代理做 sat_K 检查
    for agent in set(list(agent_to_gamma.keys()) + list(agent_to_delta.keys())):
        gammas = agent_to_gamma.get(agent, [])
        deltas = agent_to_delta.get(agent, [])
        if not sat_k_gamma_delta(gammas, deltas):
            return False

    return True

def sat_not_objective(obj: ObjectiveFormula) -> bool:
    """sat(¬α)：等价于 α 不是永真（α ≠ ⊤）"""
    return obj.obdd_node_id != true_node.id

def is_aecnf_clause_valid(clause: AECNFClause) -> bool:
    """检查AECNF子句的有效性"""
    # 先看 ¬α 是否可满足
    if not sat_not_objective(clause.objective_part):
        return True  # ¬α 不可满足，取与后整体不可满足 => 子句有效

    # 收集每个代理的 new Γ/Δ
    agent_to_gamma: dict[str, List[AEDNFAECNFPair]] = {}
    agent_to_delta: dict[str, List[AEDNFAECNFPair]] = {}

    for lit in clause.negative_literals:  # ¬K_a φ 在否定后变 K_a φ -> 进入 Γ
        agent_to_gamma.setdefault(lit.agent, []).append(lit.formula)
    for lit in clause.positive_literals:  # K_a ψ 在否定后变 ¬K_a ψ -> 进入 Δ
        agent_to_delta.setdefault(lit.agent, []).append(lit.formula)

    # ⋀_a sat_K(new Γ_a, new Δ_a)
    for agent in set(list(agent_to_gamma.keys()) + list(agent_to_delta.keys())):
        gammas = agent_to_gamma.get(agent, [])
        deltas = agent_to_delta.get(agent, [])
        if not sat_k_gamma_delta(gammas, deltas):
            return True  # 某一项失败 => ¬子句不可满足 => 子句有效

    # 若 ¬α 可满足且所有 sat_K 都通过 => ¬子句可满足 => 子句无效
    return False

def is_aednf_sat(phi: AEDNFAECNFPair) -> bool:
    """AEDNF 整式可满足：存在一项可满足"""
    for term in phi.aednf.terms:
        if sat_aednf_term(term):
            return True
    return False

def is_aecnf_valid(phi: AEDNFAECNFPair) -> bool:
    """AECNF 整式有效：所有子句都有效"""
    for clause in phi.aecnf.clauses:
        if not is_aecnf_clause_valid(clause):
            return False
    return True


# ============================================================================
# 反内省函数
# ============================================================================

def simple_deintrospective_k(phi: AEDNFAECNFPair, agent: str) -> AEDNFAECNFPair:
    """简单的反内省：K_a(φ) → φ"""
    return phi

def apply_simple_deintrospective_formula(phi: AEDNFAECNFPair) -> AEDNFAECNFPair:
    """应用简单反内省到整个公式"""
    # 简化的实现，实际应该递归处理所有知识算子
    return phi


# ============================================================================
# 工具函数
# ============================================================================

def create_objective_pair(var_name: str) -> AEDNFAECNFPair:
    """创建客观公式对"""
    node = mk_branch(var_name, false_node, true_node)
    return AEDNFAECNFPair(
        aednf=AEDNF(terms=[AEDNFTerm(
            objective_part=ObjectiveFormula(obdd_node_id=node.id, description=var_name)
        )], depth=0),
        aecnf=AECNF(clauses=[AECNFClause(
            objective_part=ObjectiveFormula(obdd_node_id=node.id, description=var_name)
        )], depth=0),
        depth=0
    )


# ============================================================================
# 公式生成器
# ============================================================================

class FormulaGenerator:
    """认知逻辑公式生成器"""
    
    def __init__(self, max_depth: int = 3, max_agents: int = 3, max_vars: int = 10):
        self.max_depth = max_depth
        self.max_agents = max_agents
        self.max_vars = max_vars
        self.agents = [f"agent_{i}" for i in range(max_agents)]
        self.variables = [f"p{i}" for i in range(max_vars)]
        self.reset_cache()
    
    def reset_cache(self):
        """重置缓存"""
        reset_cache()
    
    def generate_random_formula(self, depth: int = 0) -> AEDNFAECNFPair:
        """生成随机公式"""
        if depth >= self.max_depth:
            # 生成基本变量
            var = random.choice(self.variables)
            return create_objective_pair(var)
        
        # 随机选择操作类型
        op_type = random.choice(['variable', 'knowledge', 'and', 'or', 'not'])
        
        if op_type == 'variable':
            var = random.choice(self.variables)
            return create_objective_pair(var)
        
        elif op_type == 'knowledge':
            sub_formula = self.generate_random_formula(depth + 1)
            agent = random.choice(self.agents)
            return know(sub_formula, agent)
        
        elif op_type == 'and':
            left = self.generate_random_formula(depth + 1)
            right = self.generate_random_formula(depth + 1)
            return land(left, right)
        
        elif op_type == 'or':
            left = self.generate_random_formula(depth + 1)
            right = self.generate_random_formula(depth + 1)
            return lor(left, right)
        
        elif op_type == 'not':
            sub_formula = self.generate_random_formula(depth + 1)
            return lnot(sub_formula)
    
    def generate_formulas(self, count: int) -> List[AEDNFAECNFPair]:
        """生成指定数量的公式"""
        formulas = []
        for i in range(count):
            formula = self.generate_random_formula()
            formulas.append(formula)
        return formulas
    
    def test_formulas(self, formulas: List[AEDNFAECNFPair]) -> List[Dict]:
        """测试公式的可满足性"""
        results = []
        for i, formula in enumerate(formulas):
            start_time = time.time()
            try:
                is_sat = sat_pair(formula)
                end_time = time.time()
                results.append({
                    'index': i,
                    'formula': str(formula),
                    'depth': formula.depth,
                    'is_satisfiable': is_sat,
                    'time': end_time - start_time,
                    'success': True
                })
            except Exception as e:
                end_time = time.time()
                results.append({
                    'index': i,
                    'formula': str(formula),
                    'depth': formula.depth,
                    'is_satisfiable': None,
                    'time': end_time - start_time,
                    'success': False,
                    'error': str(e)
                })
        return results


# ============================================================================
# 主函数
# ============================================================================

def main():
    """主函数：演示公式生成和测试"""
    print("=== 认知逻辑系统 - 公式生成器 ===")
    
    # 创建公式生成器
    generator = FormulaGenerator(max_depth=3, max_agents=3, max_vars=5)
    
    # 生成公式
    print(f"生成 10 个随机公式...")
    formulas = generator.generate_formulas(10)
    
    # 测试公式
    print(f"测试公式的可满足性...")
    results = generator.test_formulas(formulas)
    
    # 输出结果
    print(f"\n=== 测试结果 ===")
    total_time = 0
    success_count = 0
    sat_count = 0
    
    for result in results:
        print(f"公式 {result['index']}:")
        print(f"  深度: {result['depth']}")
        print(f"  可满足: {result['is_satisfiable']}")
        print(f"  时间: {result['time']:.4f}秒")
        print(f"  成功: {result['success']}")
        if not result['success']:
            print(f"  错误: {result['error']}")
        print()
        
        total_time += result['time']
        if result['success']:
            success_count += 1
            if result['is_satisfiable']:
                sat_count += 1
    
    print(f"=== 统计信息 ===")
    print(f"总公式数: {len(formulas)}")
    print(f"成功测试: {success_count}")
    print(f"可满足公式: {sat_count}")
    print(f"总时间: {total_time:.4f}秒")
    print(f"平均时间: {total_time/len(formulas):.4f}秒")


if __name__ == "__main__":
    main()
