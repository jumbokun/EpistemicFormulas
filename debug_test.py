from deintrospective import deintrospective_k, decompose_term_by_agent, reorder_formula_by_agent, find_critical_index_in_reordered
from archiv.models import create_objective_pair, KnowledgeLiteral, AEDNFTerm, AEDNF, AECNF, AEDNFAECNFPair
from archiv.logical_operations import know, lnot
from archiv.models import AECNFClause


def test_simple_case():
    """
    测试一个非常简单的案例
    """
    print("=== 测试简单案例 ===")
    
    # 创建原子命题
    v1 = create_objective_pair("v1")
    p = create_objective_pair("p")
    
    # 创建知识文字
    k_a_p = KnowledgeLiteral(agent="a", formula=p, negated=False, depth=1)
    
    # 构造项：v1 ∧ K_a(p)
    term = AEDNFTerm(
        objective_part=v1.aednf.terms[0].objective_part,
        positive_literals=[k_a_p]
    )
    
    # 构造 AEDNF
    test_aednf = AEDNF(
        terms=[term],
        depth=1
    )
    
    # 构造 AECNF
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
    print(f"  AEDNF: {len(test_phi.aednf.terms)} terms")
    print(f"  AECNF: {len(test_phi.aecnf.clauses)} clauses")
    
    # 检查重新排序
    print("\n重新排序:")
    reordered = reorder_formula_by_agent(test_phi, "a")
    print(f"  重新排序后: {len(reordered.aednf.terms)} terms")
    
    # 检查临界点
    critical = find_critical_index_in_reordered(reordered, "a")
    print(f"  临界点: {critical}")
    
    # 尝试去内省
    print("\n尝试去内省:")
    try:
        result = deintrospective_k(test_phi, "a")
        print(f"  成功！结果: {len(result.aednf.terms)} terms")
    except Exception as e:
        print(f"  失败！错误: {e}")
    
    return test_phi


def test_decompose_only():
    """
    只测试项分解函数
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


if __name__ == "__main__":
    test_decompose_only()
    test_simple_case()
