#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random
from typing import List

from archiv.models import create_objective_pair, AEDNFAECNFPair, AEDNF, AEDNFTerm, AECNF, AECNFClause, KnowledgeLiteral, ObjectiveFormula
from archiv.logical_operations import land, lor, lnot, know
from deintrospective import simple_deintrospective_k, simple_deintrospective_c

AGENTS = ["a", "b"]
VARS = ["p", "q", "r", "s", "t", "u"]


def fol_of_pair(phi: AEDNFAECNFPair) -> str:
    # 优先用 AEDNF 展开为析取项
    if phi.aednf.terms:
        parts = [fol_of_term(t) for t in phi.aednf.terms]
        return " ∨ ".join(parts) if len(parts) == 1 else " ∨ ".join(f"({p})" for p in parts)
    # 兜底用 AECNF 展开为合取子句
    if phi.aecnf.clauses:
        parts = [fol_of_clause(c) for c in phi.aecnf.clauses]
        return " ∧ ".join(parts) if len(parts) == 1 else " ∧ ".join(f"({p})" for p in parts)
    return "⊥"


def fol_of_term(term: AEDNFTerm) -> str:
    conj = [term.objective_part.description or "⊤"]
    for lit in term.positive_literals:
        conj.append(f"K_{lit.agent}(" + fol_of_pair(lit.formula) + ")")
    for lit in term.negative_literals:
        conj.append(f"¬K_{lit.agent}(" + fol_of_pair(lit.formula) + ")")
    return " ∧ ".join(conj)


def fol_of_clause(clause: AECNFClause) -> str:
    disj = [clause.objective_part.description or "⊥"]
    for lit in clause.negative_literals:
        disj.append(f"¬K_{lit.agent}(" + fol_of_pair(lit.formula) + ")")
    for lit in clause.positive_literals:
        disj.append(f"K_{lit.agent}(" + fol_of_pair(lit.formula) + ")")
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
            before_aednf, before_aecnf, before_depth = signature(base)
            print(f"\n=== step {step}: know({agent}) + deintrospective ===")
            print("-- AEDNF BEFORE:\n" + (before_aednf or "⊥"))
            print("-- AECNF BEFORE:\n" + (before_aecnf or "⊤"))
            # 对同一个源 Φ 同时构造 D_a[Φ] 与 C_a[Φ]，避免深度被加两次
            src = current
            da = simple_deintrospective_k(src, agent)
            ca = simple_deintrospective_c(src, agent)
            current = AEDNFAECNFPair(aednf=da.aednf, aecnf=ca.aecnf, depth=max(da.depth, ca.depth))
            after_aednf, after_aecnf, after_depth = signature(current)
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
    print("\nPipeline finished.")


if __name__ == "__main__":
    run_pipeline(seed=20250822, steps=16, know_bias=0.7)
