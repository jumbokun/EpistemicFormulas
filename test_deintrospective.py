from deintrospective import deintrospective_k, decompose_term_by_agent, reorder_formula_by_agent
from archiv.models import create_objective_pair, KnowledgeLiteral
from archiv.logical_operations import land, lor, know


def display_pair_simple(pair):
    """简单显示公式对"""
    print(f"AEDNF: {len(pair.aednf.terms)} terms, depth={pair.aednf.depth}")
    for i, term in enumerate(pair.aednf.terms):
        print(f"  Term[{i}]: obj={term.objective_part.description}, "
              f"+lit={len(term.positive_literals)}, -lit={len(term.negative_literals)}")
    print(f"AECNF: {len(pair.aecnf.clauses)} clauses, depth={pair.aecnf.depth}")
    print()


def test_simple_deintrospective():
    """
    测试简单的去内省案例
    """
    print("=== 测试简单去内省案例 ===")
    
    # 创建原子命题
    v1 = create_objective_pair("v1")
    v2 = create_objective_pair("v2")
    v3 = create_objective_pair("v3")
    p = create_objective_pair("p")
    q = create_objective_pair("q")
    
    # 创建知识文字
    k_a_p = KnowledgeLiteral(
        agent="a",
        formula=p,
        negated=False,
        depth=1
    )
    k_b_q = KnowledgeLiteral(
        agent="b",
        formula=q,
        negated=False,
        depth=1
    )
    
    # 构造测试公式：Φ = v1 ∨ (v2 ∧ K_a(p)) ∨ (v3 ∧ K_b(q))
    term1 = v1  # v1
    term2 = land(v2, know(p, "a"))  # v2 ∧ K_a(p)
    term3 = land(v3, know(q, "b"))  # v3 ∧ K_b(q)
    
    # 构造 AEDNF
    from archiv.models import AEDNF, AECNF, AEDNFAECNFPair
    test_aednf = AEDNF(
        terms=[term1, term2, term3],
        depth=1
    )
    
    # 构造 AECNF（简化）
    test_aecnf = AECNF(
        clauses=[],
        depth=1
    )
    
    # 构造测试公式
    test_phi = AEDNFAECNFPair(
        aednf=test_aednf,
        aecnf=test_aecnf,
        depth=1
    )
    
    print("原始公式:")
    display_pair_simple(test_phi)
    
    # 对代理 a 进行去内省
    print("对代理 a 进行去内省:")
    result_a = deintrospective_k(test_phi, "a")
    display_pair_simple(result_a)
    
    # 对代理 b 进行去内省
    print("对代理 b 进行去内省:")
    result_b = deintrospective_k(test_phi, "b")
    display_pair_simple(result_b)
    
    return result_a, result_b


def test_complex_deintrospective():
    """
    测试复杂的去内省案例
    """
    print("=== 测试复杂去内省案例 ===")
    
    # 创建原子命题
    v1 = create_objective_pair("v1")
    v2 = create_objective_pair("v2")
    v3 = create_objective_pair("v3")
    v4 = create_objective_pair("v4")
    p = create_objective_pair("p")
    q = create_objective_pair("q")
    r = create_objective_pair("r")
    
    # 创建知识文字
    k_a_p = KnowledgeLiteral(agent="a", formula=p, negated=False, depth=1)
    k_a_q = KnowledgeLiteral(agent="a", formula=q, negated=False, depth=1)
    k_b_r = KnowledgeLiteral(agent="b", formula=r, negated=False, depth=1)
    not_k_a_p = KnowledgeLiteral(agent="a", formula=p, negated=True, depth=1)
    
    # 构造复杂公式：Φ = (v1 ∧ K_a(p)) ∨ (v2 ∧ ¬K_a(p)) ∨ (v3 ∧ K_a(q)) ∨ (v4 ∧ K_b(r))
    term1 = land(v1, know(p, "a"))      # v1 ∧ K_a(p)
    term2 = land(v2, lnot(know(p, "a")))  # v2 ∧ ¬K_a(p)
    term3 = land(v3, know(q, "a"))      # v3 ∧ K_a(q)
    term4 = land(v4, know(r, "b"))      # v4 ∧ K_b(r)
    
    # 构造 AEDNF
    from archiv.models import AEDNF, AECNF, AEDNFAECNFPair
    test_aednf = AEDNF(
        terms=[term1, term2, term3, term4],
        depth=1
    )
    
    test_aecnf = AECNF(
        clauses=[],
        depth=1
    )
    
    test_phi = AEDNFAECNFPair(
        aednf=test_aednf,
        aecnf=test_aecnf,
        depth=1
    )
    
    print("原始公式:")
    display_pair_simple(test_phi)
    
    # 对代理 a 进行去内省
    print("对代理 a 进行去内省:")
    result_a = deintrospective_k(test_phi, "a")
    display_pair_simple(result_a)
    
    return result_a


def test_reorder_function():
    """
    测试重新排序函数
    """
    print("=== 测试重新排序函数 ===")
    
    # 创建原子命题
    v1 = create_objective_pair("v1")
    v2 = create_objective_pair("v2")
    v3 = create_objective_pair("v3")
    v4 = create_objective_pair("v4")
    p = create_objective_pair("p")
    q = create_objective_pair("q")
    
    # 创建知识文字
    k_a_p = KnowledgeLiteral(agent="a", formula=p, negated=False, depth=1)
    k_b_q = KnowledgeLiteral(agent="b", formula=q, negated=False, depth=1)
    
    # 构造公式：Φ = v1 ∨ (v2 ∧ K_a(p)) ∨ v3 ∨ (v4 ∧ K_b(q))
    # 原始顺序：客观项 -> 主观项 -> 客观项 -> 主观项
    term1 = v1  # 客观
    term2 = land(v2, know(p, "a"))  # a-主观
    term3 = v3  # 客观
    term4 = land(v4, know(q, "b"))  # b-主观
    
    from archiv.models import AEDNF, AECNF, AEDNFAECNFPair
    test_aednf = AEDNF(
        terms=[term1, term2, term3, term4],
        depth=1
    )
    
    test_aecnf = AECNF(
        clauses=[],
        depth=1
    )
    
    test_phi = AEDNFAECNFPair(
        aednf=test_aednf,
        aecnf=test_aecnf,
        depth=1
    )
    
    print("原始公式:")
    display_pair_simple(test_phi)
    
    # 对代理 a 重新排序
    print("对代理 a 重新排序后:")
    reordered_a = reorder_formula_by_agent(test_phi, "a")
    display_pair_simple(reordered_a)
    
    # 对代理 b 重新排序
    print("对代理 b 重新排序后:")
    reordered_b = reorder_formula_by_agent(test_phi, "b")
    display_pair_simple(reordered_b)
    
    return reordered_a, reordered_b


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
    # 首先创建包含知识文字的项
    from archiv.models import AEDNFTerm, AEDNF, AECNF, AEDNFAECNFPair
    
    # 创建包含 K_a(p) 的项
    term_with_k_a_p = AEDNFTerm(
        objective_part=v1.aednf.terms[0].objective_part,
        positive_literals=[k_a_p]
    )
    
    # 创建包含 K_b(q) 的项
    term_with_k_b_q = AEDNFTerm(
        objective_part=v1.aednf.terms[0].objective_part,
        positive_literals=[k_b_q]
    )
    
    # 构造复杂项
    complex_term = AEDNFTerm(
        objective_part=v1.aednf.terms[0].objective_part,
        positive_literals=[k_a_p, k_b_q]
    )
    
    print("原始项:")
    print(f"  Objective: {complex_term.objective_part}")
    print(f"  Positive literals: {[f'{lit.agent}({lit.formula})' for lit in complex_term.positive_literals]}")
    print(f"  Negative literals: {[f'¬{lit.agent}({lit.formula})' for lit in complex_term.negative_literals]}")
    print()
    
    # 对代理 a 分解
    omega_a, theta_a = decompose_term_by_agent(complex_term, "a")
    print("对代理 a 分解:")
    print(f"  Ω_a (客观部分):")
    print(f"    Objective: {omega_a.objective_part}")
    print(f"    Positive literals: {[f'{lit.agent}({lit.formula})' for lit in omega_a.positive_literals]}")
    print(f"    Negative literals: {[f'¬{lit.agent}({lit.formula})' for lit in omega_a.negative_literals]}")
    print(f"  Θ_a (主观部分):")
    if theta_a:
        print(f"    Objective: {theta_a.objective_part}")
        print(f"    Positive literals: {[f'{lit.agent}({lit.formula})' for lit in theta_a.positive_literals]}")
        print(f"    Negative literals: {[f'¬{lit.agent}({lit.formula})' for lit in theta_a.negative_literals]}")
    else:
        print("    None")
    print()
    
    # 对代理 b 分解
    omega_b, theta_b = decompose_term_by_agent(complex_term, "b")
    print("对代理 b 分解:")
    print(f"  Ω_b (客观部分):")
    print(f"    Objective: {omega_b.objective_part}")
    print(f"    Positive literals: {[f'{lit.agent}({lit.formula})' for lit in omega_b.positive_literals]}")
    print(f"    Negative literals: {[f'¬{lit.agent}({lit.formula})' for lit in omega_b.negative_literals]}")
    print(f"  Θ_b (主观部分):")
    if theta_b:
        print(f"    Objective: {theta_b.objective_part}")
        print(f"    Positive literals: {[f'{lit.agent}({lit.formula})' for lit in theta_b.positive_literals]}")
        print(f"    Negative literals: {[f'¬{lit.agent}({lit.formula})' for lit in theta_b.negative_literals]}")
    else:
        print("    None")
    print()


if __name__ == "__main__":
    # 运行所有测试
    test_decompose_function()
    test_reorder_function()
    test_simple_deintrospective()
    test_complex_deintrospective()
