from deintrospective import deintrospective_k, decompose_term_by_agent
from archiv.models import create_objective_pair, KnowledgeLiteral, AEDNFTerm, AEDNF, AECNF, AEDNFAECNFPair
from archiv.logical_operations import know, lnot


def test_simple_deintrospective():
    """
    测试简单的去内省案例
    """
    print("=== 测试简单去内省案例 ===")
    
    # 创建原子命题
    v1 = create_objective_pair("v1")
    v2 = create_objective_pair("v2")
    p = create_objective_pair("p")
    
    # 创建知识文字
    k_a_p = KnowledgeLiteral(
        agent="a",
        formula=p,
        negated=False,
        depth=1
    )
    
    # 构造测试公式：Φ = v1 ∨ (v2 ∧ K_a(p))
    # 使用 know 函数创建 K_a(p)
    k_a_p_formula = know(p, "a")
    
    # 构造项
    term1 = v1.aednf.terms[0]  # v1
    term2 = k_a_p_formula.aednf.terms[0]  # K_a(p)
    
    # 构造 AEDNF
    test_aednf = AEDNF(
        terms=[term1, term2],
        depth=1
    )
    
    # 构造 AECNF（简化）
    from archiv.models import AECNFClause
    test_aecnf = AECNF(
        clauses=[AECNFClause(objective_part=v1.aednf.terms[0].objective_part)],
        depth=1
    )
    
    # 构造测试公式
    test_phi = AEDNFAECNFPair(
        aednf=test_aednf,
        aecnf=test_aecnf,
        depth=1
    )
    
    print("原始公式:")
    print(f"AEDNF: {len(test_phi.aednf.terms)} terms, depth={test_phi.aednf.depth}")
    for i, term in enumerate(test_phi.aednf.terms):
        print(f"  Term[{i}]: obj={term.objective_part.description}, "
              f"+lit={len(term.positive_literals)}, -lit={len(term.negative_literals)}")
    print()
    
    # 对代理 a 进行去内省
    print("对代理 a 进行去内省:")
    result_a = deintrospective_k(test_phi, "a")
    print(f"结果 AEDNF: {len(result_a.aednf.terms)} terms, depth={result_a.aednf.depth}")
    for i, term in enumerate(result_a.aednf.terms):
        print(f"  Term[{i}]: obj={term.objective_part.description}, "
              f"+lit={len(term.positive_literals)}, -lit={len(term.negative_literals)}")
    print()
    
    return result_a


def test_decompose_function():
    """
    测试项分解函数
    """
    print("=== 测试项分解函数 ===")
    
    # 创建原子命题
    v1 = create_objective_pair("v1")
    p = create_objective_pair("p")
    q = create_objective_pair("q")
    
    # 创建知识文字
    k_a_p = KnowledgeLiteral(agent="a", formula=p, negated=False, depth=1)
    k_b_q = KnowledgeLiteral(agent="b", formula=q, negated=False, depth=1)
    
    # 构造复杂项：v1 ∧ K_a(p) ∧ K_b(q)
    complex_term = AEDNFTerm(
        objective_part=v1.aednf.terms[0].objective_part,
        positive_literals=[k_a_p, k_b_q]
    )
    
    print("原始项:")
    print(f"  Objective: {complex_term.objective_part.description}")
    print(f"  Positive literals: {[f'{lit.agent}({lit.formula.aednf.terms[0].objective_part.description})' for lit in complex_term.positive_literals]}")
    print(f"  Negative literals: {[f'¬{lit.agent}({lit.formula.aednf.terms[0].objective_part.description})' for lit in complex_term.negative_literals]}")
    print()
    
    # 对代理 a 分解
    omega_a, theta_a = decompose_term_by_agent(complex_term, "a")
    print("对代理 a 分解:")
    print(f"  Ω_a (客观部分):")
    print(f"    Objective: {omega_a.objective_part.description}")
    print(f"    Positive literals: {[f'{lit.agent}({lit.formula.aednf.terms[0].objective_part.description})' for lit in omega_a.positive_literals]}")
    print(f"    Negative literals: {[f'¬{lit.agent}({lit.formula.aednf.terms[0].objective_part.description})' for lit in omega_a.negative_literals]}")
    print(f"  Θ_a (主观部分):")
    if theta_a:
        print(f"    Objective: {theta_a.objective_part.description}")
        print(f"    Positive literals: {[f'{lit.agent}({lit.formula.aednf.terms[0].objective_part.description})' for lit in theta_a.positive_literals]}")
        print(f"    Negative literals: {[f'¬{lit.agent}({lit.formula.aednf.terms[0].objective_part.description})' for lit in theta_a.negative_literals]}")
    else:
        print("    None")
    print()
    
    # 对代理 b 分解
    omega_b, theta_b = decompose_term_by_agent(complex_term, "b")
    print("对代理 b 分解:")
    print(f"  Ω_b (客观部分):")
    print(f"    Objective: {omega_b.objective_part.description}")
    print(f"    Positive literals: {[f'{lit.agent}({lit.formula.aednf.terms[0].objective_part.description})' for lit in omega_b.positive_literals]}")
    print(f"    Negative literals: {[f'¬{lit.agent}({lit.formula.aednf.terms[0].objective_part.description})' for lit in omega_b.negative_literals]}")
    print(f"  Θ_b (主观部分):")
    if theta_b:
        print(f"    Objective: {theta_b.objective_part.description}")
        print(f"    Positive literals: {[f'{lit.agent}({lit.formula.aednf.terms[0].objective_part.description})' for lit in theta_b.positive_literals]}")
        print(f"    Negative literals: {[f'¬{lit.agent}({lit.formula.aednf.terms[0].objective_part.description})' for lit in theta_b.negative_literals]}")
    else:
        print("    None")
    print()


if __name__ == "__main__":
    # 运行测试
    test_decompose_function()
    test_simple_deintrospective()
