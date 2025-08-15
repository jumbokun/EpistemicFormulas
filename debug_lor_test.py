from deintrospective import create_vm_minus_1, create_omega_disjunction, find_critical_index_in_reordered
from archiv.models import create_objective_pair, KnowledgeLiteral, AEDNFTerm, AEDNF, AECNF, AEDNFAECNFPair
from archiv.logical_operations import know, lor
from archiv.models import AECNFClause


def test_lor_behavior():
    """
    测试 lor 函数的行为
    """
    print("=== 测试 lor 函数行为 ===")
    
    # 创建原子命题
    v1 = create_objective_pair("v1")
    v2 = create_objective_pair("v2")
    p = create_objective_pair("p")
    
    # 创建知识文字
    k_a_p = KnowledgeLiteral(agent="a", formula=p, negated=False, depth=1)
    
    # 构造项
    term1 = v1.aednf.terms[0]  # v1
    term2 = AEDNFTerm(
        objective_part=v2.aednf.terms[0].objective_part,
        positive_literals=[k_a_p]
    )  # v2 ∧ K_a(p)
    
    # 构造 AEDNF
    test_aednf = AEDNF(
        terms=[term1, term2],
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
    
    # 测试 create_vm_minus_1
    print("\n测试 create_vm_minus_1 (m=1):")
    vm_minus_1 = create_vm_minus_1(test_phi, "a", 1)
    print(f"  V_0: {len(vm_minus_1.aednf.terms)} terms")
    
    # 测试 create_omega_disjunction
    print("\n测试 create_omega_disjunction (start=1, end=1):")
    omega_disj = create_omega_disjunction(test_phi, "a", 1, 1)
    print(f"  ⋁_{{i=1}}^1 Ω_i: {len(omega_disj.aednf.terms)} terms")
    
    # 测试 lor 函数
    print("\n测试 lor 函数:")
    result = lor(vm_minus_1, omega_disj)
    print(f"  lor 结果: {len(result.aednf.terms)} terms")
    
    # 检查临界点
    critical = find_critical_index_in_reordered(result, "a")
    print(f"  临界点: {critical}")
    
    return result


if __name__ == "__main__":
    test_lor_behavior()
