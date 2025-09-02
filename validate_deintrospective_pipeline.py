#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random
from typing import List

from archiv.models import create_objective_pair, AEDNFAECNFPair, AEDNF, AEDNFTerm, AECNF, AECNFClause, KnowledgeLiteral, ObjectiveFormula
from archiv.logical_operations import land, lor, lnot, know
from deintrospective import simple_deintrospective_k, simple_deintrospective_c, TraceLogger

AGENTS = ["a", "b"]
VARS = ["p", "q", "r", "s", "t", "u"]


def _canon_obj_desc(desc: str | None) -> str:
    if not desc:
        return ""
    if desc == "¬(⊤)":
        return "⊥"
    if desc == "¬(⊥)":
        return "⊤"
    return desc


def fol_of_pair(phi: AEDNFAECNFPair, prefer_aednf: bool = True) -> str:
    """
    展开公式为 FOL 表示
    
    Args:
        phi: 要展开的公式
        prefer_aednf: 是否优先使用 AEDNF 形式
                     True: 总是展开为 AEDNF 形式（析取项）
                     False: 总是展开为 AECNF 形式（合取子句）
    """
    if prefer_aednf:
        # 总是展开为 AEDNF 形式（析取项）
        if phi.aednf.terms:
            parts = [fol_of_term(t) for t in phi.aednf.terms]
            parts = [p for p in parts if p != "⊥"]  # AEDNF 中移除恒假项
            if not parts:
                return "⊥"
            return " ∨ ".join(parts) if len(parts) == 1 else " ∨ ".join(f"({p})" for p in parts)
        else:
            return "⊥"
    else:
        # 总是展开为 AECNF 形式（合取子句）
        if phi.aecnf.clauses:
            parts = [fol_of_clause(c) for c in phi.aecnf.clauses]
            parts = [p for p in parts if p != "⊤"]  # AECNF 中移除恒真子句
            if not parts:
                return "⊤"
            return " ∧ ".join(parts) if len(parts) == 1 else " ∧ ".join(f"({p})" for p in parts)
        else:
            return "⊤"


def fol_of_term(term: AEDNFTerm) -> str:
    # 规范化客观部
    obj_desc = _canon_obj_desc(term.objective_part.description)
    if obj_desc == "⊥":
        return "⊥"  # ⊥ ∧ φ ≡ ⊥
    conj: List[str] = []
    if obj_desc and obj_desc != "⊤":
        conj.append(obj_desc)
    for lit in term.positive_literals:
        # 在 AEDNF term 中，总是展开为 AEDNF 形式
        nested_formula = fol_of_pair(lit.formula, prefer_aednf=True)
        conj.append(f"K_{lit.agent}({nested_formula})")
    for lit in term.negative_literals:
        # 在 AEDNF term 中，总是展开为 AEDNF 形式
        nested_formula = fol_of_pair(lit.formula, prefer_aednf=True)
        conj.append(f"¬K_{lit.agent}({nested_formula})")
    if not conj:
        return "⊤"  # 无任何合取项则打印 ⊤
    return " ∧ ".join(conj)


def fol_of_clause(clause: AECNFClause) -> str:
    # 规范化客观部
    obj_desc = _canon_obj_desc(clause.objective_part.description)
    if obj_desc == "⊤":
        return "⊤"  # ⊤ ∨ φ ≡ ⊤
    disj: List[str] = []
    if obj_desc and obj_desc != "⊥":
        disj.append(obj_desc)
    for lit in clause.negative_literals:
        # 在 AECNF clause 中，总是展开为 AECNF 形式
        nested_formula = fol_of_pair(lit.formula, prefer_aednf=False)
        disj.append(f"¬K_{lit.agent}({nested_formula})")
    for lit in clause.positive_literals:
        # 在 AECNF clause 中，总是展开为 AECNF 形式
        nested_formula = fol_of_pair(lit.formula, prefer_aednf=False)
        disj.append(f"K_{lit.agent}({nested_formula})")
    if not disj:
        return "⊥"  # 无任何析取项则打印 ⊥
    return " ∨ ".join(disj)


def display_pair(phi: AEDNFAECNFPair, tag: str):
    print(f"\n=== {tag} ===")
    print(f"depth={phi.depth}")
    print(f"AEDNF terms={len(phi.aednf.terms)}")
    for i, term in enumerate(phi.aednf.terms, 1):
        print(f"  [T{i}] " + fol_of_term(term))
    print(f"AECNF clauses={len(phi.aecnf.clauses)}")
    for i, clause in enumerate(phi.aecnf.clauses, 1):
        print(f"  [C{i}] " + fol_of_clause(clause))


def signature(phi: AEDNFAECNFPair) -> tuple[str, str, int]:
    aednf_str = " ∨ ".join(fol_of_term(t) for t in phi.aednf.terms)
    aecnf_str = " ∧ ".join(fol_of_clause(c) for c in phi.aecnf.clauses)
    return aednf_str, aecnf_str, phi.depth

def signature_consistent(phi: AEDNFAECNFPair) -> tuple[str, str, int]:
    """
    一致性展开：AEDNF 总是展开为 AEDNF 形式，AECNF 总是展开为 AECNF 形式
    """
    aednf_str = " ∨ ".join(fol_of_term(t) for t in phi.aednf.terms)
    aecnf_str = " ∧ ".join(fol_of_clause(c) for c in phi.aecnf.clauses)
    return aednf_str, aecnf_str, phi.depth

def debug_formula_structure(phi: AEDNFAECNFPair, label: str = ""):
    """调试函数：显示公式的详细结构"""
    print(f"\n=== DEBUG {label} ===")
    print(f"Depth: {phi.depth}")
    print(f"AEDNF terms ({len(phi.aednf.terms)}):")
    for i, term in enumerate(phi.aednf.terms):
        print(f"  Term {i+1}:")
        print(f"    Objective: {term.objective_part.description} (ID: {term.objective_part.obdd_node_id})")
        print(f"    Positive literals ({len(term.positive_literals)}):")
        for j, lit in enumerate(term.positive_literals):
            print(f"      {j+1}. K_{lit.agent}(...) - depth: {lit.depth}")
            print(f"         Formula depth: {lit.formula.depth}")
        print(f"    Negative literals ({len(term.negative_literals)}):")
        for j, lit in enumerate(term.negative_literals):
            print(f"      {j+1}. ¬K_{lit.agent}(...) - depth: {lit.depth}")
            print(f"         Formula depth: {lit.formula.depth}")
    
    print(f"AECNF clauses ({len(phi.aecnf.clauses)}):")
    for i, clause in enumerate(phi.aecnf.clauses):
        print(f"  Clause {i+1}:")
        print(f"    Objective: {clause.objective_part.description} (ID: {clause.objective_part.obdd_node_id})")
        print(f"    Positive literals ({len(clause.positive_literals)}):")
        for j, lit in enumerate(clause.positive_literals):
            print(f"      {j+1}. K_{lit.agent}(...) - depth: {lit.depth}")
            print(f"         Formula depth: {lit.formula.depth}")
        print(f"    Negative literals ({len(clause.negative_literals)}):")
        for j, lit in enumerate(clause.negative_literals):
            print(f"      {j+1}. ¬K_{lit.agent}(...) - depth: {lit.depth}")
            print(f"         Formula depth: {lit.formula.depth}")


def generate_step(depth_cap: int) -> AEDNFAECNFPair:
    # 生成一个客观原子
    v = create_objective_pair(random.choice(VARS))
    return v


def run_pipeline(seed: int = 42, steps: int = 12, know_bias: float = 0.65):
    random.seed(seed)
    current = generate_step(0)
    display_pair(current, "init objective")

    for step in range(1, steps + 1):
        ops = ["know"] * int(know_bias * 10) + ["and", "or"] * 2 + ["not"]
        op = random.choice(ops)
        if op == "know":
            agent = random.choice(AGENTS)
            # 记录 before
            base = know(current, agent)
            before_aednf, before_aecnf, before_depth = signature_consistent(base)
            print(f"\n=== step {step}: know({agent}) + deintrospective ===")
            print("-- AEDNF BEFORE:\n" + (before_aednf or "⊥"))
            print("-- AECNF BEFORE:\n" + (before_aecnf or "⊤"))
            
            # 只对 step 4 显示详细调试信息
            if step == 4:
                debug_formula_structure(base, f"step {step} know({agent}) BEFORE")
                logger = TraceLogger(enabled=True)
                print(f"\n--- TRACE: 开始 AEDNF deintrospective for agent {agent} ---")
                da = simple_deintrospective_k(current, agent, logger)
                print(f"--- DEBUG: AEDNF deintrospective 结果 ---")
                print(f"   深度: {da.depth}")
                print(f"   terms数: {len(da.aednf.terms)}")
                for i, term in enumerate(da.aednf.terms):
                    print(f"   Term {i+1}: {fol_of_term(term)}")
            else:
                # 其他步骤只显示基本信息
                da = simple_deintrospective_k(current, agent)
            
            ca = simple_deintrospective_c(current, agent)
            
            current = AEDNFAECNFPair(aednf=da.aednf, aecnf=ca.aecnf, depth=max(da.depth, ca.depth))
            after_aednf, after_aecnf, after_depth = signature_consistent(current)
            print("-- AEDNF AFTER:\n" + (after_aednf or "⊥"))
            print("-- AECNF AFTER:\n" + (after_aecnf or "⊤"))
            print(f"depth: {before_depth} -> {after_depth}")
        elif op == "and":
            rhs = generate_step(0)
            current = land(current, rhs)
            display_pair(current, f"step {step}: land with new objective")
        elif op == "or":
            rhs = generate_step(0)
            current = lor(current, rhs)
            display_pair(current, f"step {step}: lor with new objective")
        elif op == "not":
            current = lnot(current)
            display_pair(current, f"step {step}: lnot")
        
        # 只验证到 step 5
        if step >= 5:
            print(f"\n=== 验证完成到 step {step}，停止 ===")
            break
    
    print("\nPipeline finished.")


if __name__ == "__main__":
    run_pipeline(seed=20250822, steps=16, know_bias=0.7)
