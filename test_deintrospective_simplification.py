#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试反内省和简化功能
"""

from epistemic_logic_system import (
    create_objective_pair, know, land, lor, lnot, sat_pair,
    simplify_pair, simple_deintrospective_k
)

def test_deintrospective():
    """测试反内省功能"""
    print("=== 测试反内省功能 ===")
    
    # 创建基本变量
    p = create_objective_pair("p")
    q = create_objective_pair("q")
    
    # 测试 K_a(p) 的反内省
    k_a_p = know(p, "agent_0")
    print(f"原始公式: {k_a_p}")
    print(f"反内省后: {simple_deintrospective_k(k_a_p, 'agent_0')}")
    print()

def test_simplification():
    """测试简化功能"""
    print("=== 测试简化功能 ===")
    
    # 创建基本变量
    p = create_objective_pair("p")
    q = create_objective_pair("q")
    
    # 测试与操作
    p_and_q = land(p, q)
    print(f"与操作前: {p_and_q}")
    simplified = simplify_pair(p_and_q)
    print(f"简化后: {simplified}")
    print()
    
    # 测试或操作
    p_or_q = lor(p, q)
    print(f"或操作前: {p_or_q}")
    simplified = simplify_pair(p_or_q)
    print(f"简化后: {simplified}")
    print()
    
    # 测试非操作
    not_p = lnot(p)
    print(f"非操作前: {not_p}")
    simplified = simplify_pair(not_p)
    print(f"简化后: {simplified}")
    print()

def test_knowledge_with_deintrospective():
    """测试知识算子中的反内省"""
    print("=== 测试知识算子中的反内省 ===")
    
    # 创建基本变量
    p = create_objective_pair("p")
    q = create_objective_pair("q")
    
    # 测试 K_a(p ∧ q)
    p_and_q = land(p, q)
    k_a_p_and_q = know(p_and_q, "agent_0")
    print(f"K_a(p ∧ q): {k_a_p_and_q}")
    print(f"可满足性: {sat_pair(k_a_p_and_q)}")
    print()
    
    # 测试嵌套知识算子
    k_b_q = know(q, "agent_1")
    k_a_k_b_q = know(k_b_q, "agent_0")
    print(f"K_a(K_b(q)): {k_a_k_b_q}")
    print(f"可满足性: {sat_pair(k_a_k_b_q)}")
    print()

def test_complex_formula():
    """测试复杂公式"""
    print("=== 测试复杂公式 ===")
    
    # 创建基本变量
    p = create_objective_pair("p")
    q = create_objective_pair("q")
    r = create_objective_pair("r")
    
    # 构造复杂公式: K_a(p ∧ q) ∨ K_b(¬r)
    k_a_p_and_q = know(land(p, q), "agent_0")
    k_b_not_r = know(lnot(r), "agent_1")
    complex_formula = lor(k_a_p_and_q, k_b_not_r)
    
    print(f"复杂公式: K_a(p ∧ q) ∨ K_b(¬r)")
    print(f"公式: {complex_formula}")
    print(f"可满足性: {sat_pair(complex_formula)}")
    print()

if __name__ == "__main__":
    test_deintrospective()
    test_simplification()
    test_knowledge_with_deintrospective()
    test_complex_formula()
