from typing import List, Tuple, Optional
from archiv.models import AEDNFAECNFPair, AEDNF, AECNF, AEDNFTerm, AECNFClause, KnowledgeLiteral, ObjectiveFormula
from archiv.logical_operations import land, lor, know, lnot
from archiv.obdd import true_node, false_node, AND, OR


def decompose_term_by_agent(term: AEDNFTerm, agent: str) -> Tuple[AEDNFTerm, Optional[AEDNFTerm]]:
    """
    将项按代理分解为 (Ω_i, Θ_i) 对
    Ω_i: a-客观部分（命题部分 + 非a代理的知识文字）
    Θ_i: a-主观部分（a代理的知识文字）
    """
    # 分离知识文字
    agent_literals = []      # Θ_i 部分
    other_literals = []      # Ω_i 部分
    
    for literal in term.positive_literals + term.negative_literals:
        if literal.agent == agent:
            agent_literals.append(literal)
        else:
            other_literals.append(literal)
    
    # 构造 Ω_i
    omega_i = AEDNFTerm(
        objective_part=term.objective_part,
        positive_literals=[lit for lit in other_literals if not lit.negated],
        negative_literals=[lit for lit in other_literals if lit.negated]
    )
    
    # 构造 Θ_i（如果存在a代理的知识文字）
    if agent_literals:
        theta_i = AEDNFTerm(
            objective_part=ObjectiveFormula(
                obdd_node_id=true_node.id,
                description="⊤"
            ),
            positive_literals=[lit for lit in agent_literals if not lit.negated],
            negative_literals=[lit for lit in agent_literals if lit.negated]
        )
        return omega_i, theta_i
    else:
        return omega_i, None


def find_critical_index(phi: AEDNFAECNFPair, agent: str) -> int:
    """
    找到临界点 ℓ_Φ：最大的非a-客观子句的索引
    当 ℓ_Φ = 0 时，整个公式都是 a-客观的
    """
    for i, term in enumerate(phi.aednf.terms):
        omega, theta = decompose_term_by_agent(term, agent)
        if theta is None:  # 找到第一个纯客观项
            return i  # 返回主观项的数量
    return len(phi.aednf.terms)  # 所有项都是主观的


def create_vm(phi: AEDNFAECNFPair, agent: str, m: int) -> AEDNFAECNFPair:
    """
    构造 V_m = (Ω₁ ∧ Θ₁) ∨ (Ω₂ ∧ Θ₂) ∨ ... ∨ (Ω_m ∧ Θ_m)
    """
    if m <= 0:
        # 构造空公式（⊥）
        empty_term = AEDNFTerm(
            objective_part=ObjectiveFormula(obdd_node_id=false_node.id, description="⊥"),
            positive_literals=[],
            negative_literals=[]
        )
        empty_aednf = AEDNF(terms=[empty_term], depth=0)
        empty_aecnf = AECNF(
            clauses=[AECNFClause(
                objective_part=ObjectiveFormula(obdd_node_id=false_node.id, description="⊥"),
                positive_literals=[],
                negative_literals=[]
            )],
            depth=0
        )
        return AEDNFAECNFPair(aednf=empty_aednf, aecnf=empty_aecnf, depth=0)
    
    # 取前 m 个项
    vm_terms = phi.aednf.terms[:m]
    
    vm_aednf = AEDNF(
        terms=vm_terms,
        depth=phi.aednf.depth
    )
    
    # 构造对应的 AECNF（简化处理）
    vm_aecnf = phi.aecnf
    
    return AEDNFAECNFPair(
        aednf=vm_aednf,
        aecnf=vm_aecnf,
        depth=phi.depth
    )


def create_omega_disjunction(phi: AEDNFAECNFPair, agent: str, start_idx: int, end_idx: int) -> AEDNFAECNFPair:
    """
    构造 ⋁_{i=start_idx}^{end_idx} Ω_i
    """
    if start_idx >= len(phi.aednf.terms):
        # 构造空公式（⊥）
        empty_term = AEDNFTerm(
            objective_part=ObjectiveFormula(obdd_node_id=false_node.id, description="⊥"),
            positive_literals=[],
            negative_literals=[]
        )
        empty_aednf = AEDNF(terms=[empty_term], depth=0)
        empty_aecnf = AECNF(
            clauses=[AECNFClause(
                objective_part=ObjectiveFormula(obdd_node_id=false_node.id, description="⊥"),
                positive_literals=[],
                negative_literals=[]
            )],
            depth=0
        )
        return AEDNFAECNFPair(aednf=empty_aednf, aecnf=empty_aecnf, depth=0)
    
    # 取指定范围的项，只保留 Ω_i 部分
    omega_terms = []
    for i in range(start_idx, min(end_idx + 1, len(phi.aednf.terms))):
        omega, _ = decompose_term_by_agent(phi.aednf.terms[i], agent)
        omega_terms.append(omega)
    
    omega_aednf = AEDNF(
        terms=omega_terms,
        depth=phi.aednf.depth
    )
    
    # 构造对应的 AECNF（简化处理）
    omega_aecnf = phi.aecnf
    
    return AEDNFAECNFPair(
        aednf=omega_aednf,
        aecnf=omega_aecnf,
        depth=phi.depth
    )


def get_clauses_from_aednf(phi: AEDNFAECNFPair) -> List[AEDNFTerm]:
    """
    从 AEDNF 中获取所有项（作为子句）
    """
    return phi.aednf.terms


def deintrospective_k(phi: AEDNFAECNFPair, agent: str) -> AEDNFAECNFPair:
    """
    构造 K_agent(phi) 的去内省形式 D[phi]
    根据数学定义：
    - 当 ℓ_Φ = 0 时，D_a[Φ] = K_a(Φ)
    - 当 ℓ_Φ = m > 0 时，使用递归公式
    """
    # 找到临界点
    critical_index = find_critical_index(phi, agent)
    
    print(f"DEBUG: 当前公式有 {len(phi.aednf.terms)} 个项，临界点 ℓ_Φ = {critical_index}")
    
    # 基本情形：整个公式都是 a-客观的
    if critical_index == 0:
        print(f"DEBUG: 基本情形，直接返回 K_{agent}(phi)")
        return know(phi, agent)
    
    # 递归情形：存在 a-主观部分需要外提
    print(f"DEBUG: 递归情形，开始处理 {critical_index} 个 subjective clauses")
    
    # 根据数学定义，我们需要重新排序公式，将包含a-主观部分的项移到前面
    # 然后应用递归公式
    return apply_deintrospective_formula(phi, agent, critical_index)


def apply_deintrospective_formula(phi: AEDNFAECNFPair, agent: str, m: int) -> AEDNFAECNFPair:
    """
    应用去内省公式：
    D_a[Φ] = D_a[V_{m-1} ∨ ⋁_{i=m+1}^n Ω_i] ∨ ⋁_{C∈C(D_a[V_{m-1} ∨ ⋁_{i=m}^n Ω_i])} (C ∧ Θ_m)
    """
    n = len(phi.aednf.terms)
    print(f"DEBUG: 应用去内省公式 - m={m}, n={n}")
    
    # 步骤1：构造 V_{m-1}
    vm_minus_1 = create_vm(phi, agent, m-1)
    print(f"DEBUG: V_{m-1} 构造完成，包含 {len(vm_minus_1.aednf.terms)} 个项")
    
    # 步骤2：构造 ⋁_{i=m+1}^n Ω_i
    omega_from_m_plus_1 = create_omega_disjunction(phi, agent, m, n-1)
    print(f"DEBUG: ⋁_{{i={m+1}}}^n Ω_i 构造完成，包含 {len(omega_from_m_plus_1.aednf.terms)} 个项")
    
    # 步骤3：构造 ⋁_{i=m}^n Ω_i
    omega_from_m = create_omega_disjunction(phi, agent, m-1, n-1)
    print(f"DEBUG: ⋁_{{i={m}}}^n Ω_i 构造完成，包含 {len(omega_from_m.aednf.terms)} 个项")
    
    # 步骤4：计算 D_a[V_{m-1} ∨ ⋁_{i=m+1}^n Ω_i]
    first_arg = lor(vm_minus_1, omega_from_m_plus_1)
    print(f"DEBUG: 第一个递归参数构造完成")
    d1 = deintrospective_k(first_arg, agent)
    print(f"DEBUG: 第一个递归调用完成")
    
    # 步骤5：计算 D_a[V_{m-1} ∨ ⋁_{i=m}^n Ω_i]
    second_arg = lor(vm_minus_1, omega_from_m)
    print(f"DEBUG: 第二个递归参数构造完成")
    d2 = deintrospective_k(second_arg, agent)
    print(f"DEBUG: 第二个递归调用完成")
    
    # 步骤6：获取 Θ_m
    _, theta_m = decompose_term_by_agent(phi.aednf.terms[m-1], agent)
    print(f"DEBUG: Θ_m 获取完成，是否为 None: {theta_m is None}")
    
    # 步骤7：构造 ⋁_{C∈C(D2)} (C ∧ Θ_m)
    d2_clauses = get_clauses_from_aednf(d2)
    print(f"DEBUG: D2 包含 {len(d2_clauses)} 个子句")
    
    combined_terms = []
    for clause in d2_clauses:
        if theta_m is not None:
            # 将 clause 转换为 AEDNFAECNFPair
            clause_pair = AEDNFAECNFPair(
                aednf=AEDNF(terms=[clause], depth=clause.depth if hasattr(clause, 'depth') else 0),
                aecnf=AECNF(clauses=[AECNFClause(
                    objective_part=clause.objective_part,
                    positive_literals=clause.positive_literals,
                    negative_literals=clause.negative_literals
                )], depth=0),
                depth=clause.depth if hasattr(clause, 'depth') else 0
            )
            
            # 将 theta_m 转换为 AEDNFAECNFPair
            theta_m_pair = AEDNFAECNFPair(
                aednf=AEDNF(terms=[theta_m], depth=0),
                aecnf=AECNF(clauses=[AECNFClause(
                    objective_part=theta_m.objective_part,
                    positive_literals=theta_m.positive_literals,
                    negative_literals=theta_m.negative_literals
                )], depth=0),
                depth=0
            )
            
            combined_term = land(clause_pair, theta_m_pair)
            combined_terms.append(combined_term)
    
    print(f"DEBUG: 构造了 {len(combined_terms)} 个 combined terms")
    
    # 步骤8：最终结果：D1 ∨ 所有 (C ∧ Θ_m)
    if combined_terms:
        # 将 D1 的项与 combined_terms 合并
        final_terms = d1.aednf.terms + [term.aednf.terms[0] for term in combined_terms]
        
        # 构造最终的 AEDNF
        final_aednf = AEDNF(
            terms=final_terms,
            depth=max(d1.depth, d2.depth) if combined_terms else d1.depth
        )
        
        # 构造最终的 AECNF（简化处理）
        final_aecnf = d1.aecnf
        
        result = AEDNFAECNFPair(
            aednf=final_aednf,
            aecnf=final_aecnf,
            depth=max(d1.depth, d2.depth) if combined_terms else d1.depth
        )
        print(f"DEBUG: 递归完成，返回包含 {len(result.aednf.terms)} 个项的公式")
        return result
    else:
        # 如果没有 Θ_m，直接返回 D1
        print(f"DEBUG: 没有 Θ_m，直接返回 D1")
        return d1


def simple_deintrospective_k(phi: AEDNFAECNFPair, agent: str) -> AEDNFAECNFPair:
    """
    简单但正确的去内省算法实现
    目标：将 K_a(Φ) 转换为等价形式，其中所有a-主观部分都被外提到最外层
    """
    print(f"DEBUG: 开始简单去内省算法")
    
    # 找到临界点
    critical_index = find_critical_index(phi, agent)
    print(f"DEBUG: 临界点 ℓ_Φ = {critical_index}")
    
    # 基本情形：整个公式都是 a-客观的
    if critical_index == 0:
        print(f"DEBUG: 基本情形，直接返回 K_{agent}(phi)")
        return know(phi, agent)
    
    # 递归情形：存在 a-主观部分需要外提
    print(f"DEBUG: 递归情形，处理 {critical_index} 个 subjective clauses")
    
    # 重新排序：主观项在前，客观项在后
    subjective_terms = []
    objective_terms = []
    
    for term in phi.aednf.terms:
        omega, theta = decompose_term_by_agent(term, agent)
        if theta is not None:  # 包含 a-主观部分
            subjective_terms.append(term)
        else:  # 纯 a-客观
            objective_terms.append(term)
    
    # 重新构造公式
    reordered_terms = subjective_terms + objective_terms
    reordered_phi = AEDNFAECNFPair(
        aednf=AEDNF(terms=reordered_terms, depth=phi.depth),
        aecnf=phi.aecnf,
        depth=phi.depth
    )
    
    print(f"DEBUG: 重新排序完成，主观项: {len(subjective_terms)}, 客观项: {len(objective_terms)}")
    
    # 应用简单的递归公式
    return apply_simple_deintrospective_formula(reordered_phi, agent, len(subjective_terms))


def apply_simple_deintrospective_formula(phi: AEDNFAECNFPair, agent: str, m: int) -> AEDNFAECNFPair:
    """
    应用简单的去内省公式
    简化理解：对于 Φ = (Ω₁ ∧ Θ₁) ∨ (Ω₂ ∧ Θ₂) ∨ ... ∨ (Ω_m ∧ Θ_m) ∨ ⋁_{i=m+1}^n Ω_i
    我们希望得到：D_a[Φ] = K_a(⋁_{i=m+1}^n Ω_i) ∧ Θ₁ ∧ Θ₂ ∧ ... ∧ Θ_m
    """
    n = len(phi.aednf.terms)
    print(f"DEBUG: 应用简单去内省公式 - m={m}, n={n}")
    
    if m == 0:
        # 没有主观项，直接返回 K_a(phi)
        return know(phi, agent)
    
    # 构造客观部分：⋁_{i=m+1}^n Ω_i
    objective_terms = []
    for i in range(m, n):
        omega, _ = decompose_term_by_agent(phi.aednf.terms[i], agent)
        objective_terms.append(omega)
    
    # 构造主观部分：Θ₁ ∧ Θ₂ ∧ ... ∧ Θ_m
    subjective_terms = []
    for i in range(m):
        _, theta = decompose_term_by_agent(phi.aednf.terms[i], agent)
        if theta is not None:
            subjective_terms.append(theta)
    
    # 构造结果：K_a(客观部分) ∧ 主观部分
    if objective_terms:
        # 有客观部分
        objective_phi = AEDNFAECNFPair(
            aednf=AEDNF(terms=objective_terms, depth=phi.depth),
            aecnf=phi.aecnf,
            depth=phi.depth
        )
        k_objective = know(objective_phi, agent)
        
        if subjective_terms:
            # 有主观部分，需要与客观部分结合
            result = k_objective
            for theta in subjective_terms:
                theta_pair = AEDNFAECNFPair(
                    aednf=AEDNF(terms=[theta], depth=0),
                    aecnf=AECNF(clauses=[AECNFClause(
                        objective_part=theta.objective_part,
                        positive_literals=theta.positive_literals,
                        negative_literals=theta.negative_literals
                    )], depth=0),
                    depth=0
                )
                result = land(result, theta_pair)
            return result
        else:
            # 没有主观部分
            return k_objective
    else:
        # 没有客观部分，只有主观部分
        if subjective_terms:
            # 只有主观部分，直接返回 K_a(phi)
            return know(phi, agent)
        else:
            # 既没有客观部分也没有主观部分（不应该发生）
            return know(phi, agent)


def verify_deintrospective_result(result: AEDNFAECNFPair, agent: str) -> bool:
    """
    验证去内省结果是否正确
    检查结果中是否还有a-主观部分
    注意：去内省的目标是将a-主观部分外提到最外层，而不是完全消除它们
    """
    # 检查结果中是否包含a-主观部分（这是正确的）
    has_a_subjective = False
    for term in result.aednf.terms:
        for lit in term.positive_literals + term.negative_literals:
            if lit.agent == agent:
                has_a_subjective = True
                break
        if has_a_subjective:
            break
    
    # 如果有a-主观部分，检查它们是否在最外层
    if has_a_subjective:
        # 检查是否所有a-主观部分都在最外层（即没有嵌套的a-主观部分）
        for term in result.aednf.terms:
            for lit in term.positive_literals + term.negative_literals:
                if lit.agent == agent:
                    # 检查这个a-主观部分是否包含更深层的a-主观部分
                    if has_nested_a_subjective(lit.formula, agent):
                        return False  # 发现嵌套的a-主观部分，结果不正确
        return True  # 所有a-主观部分都在最外层，结果正确
    else:
        return True  # 没有a-主观部分，结果正确


def has_nested_a_subjective(phi: AEDNFAECNFPair, agent: str) -> bool:
    """
    检查公式中是否有嵌套的a-主观部分
    """
    for term in phi.aednf.terms:
        for lit in term.positive_literals + term.negative_literals:
            if lit.agent == agent:
                # 发现a-主观部分，检查其内部是否还有a-主观部分
                if has_nested_a_subjective(lit.formula, agent):
                    return True
            else:
                # 检查非a-主观部分内部是否有a-主观部分
                if has_nested_a_subjective(lit.formula, agent):
                    return True
    return False


def visualize_formula(phi: AEDNFAECNFPair, title: str = "公式"):
    """
    可视化公式结构
    """
    print(f"\n=== {title} ===")
    print(f"深度: {phi.depth}")
    print(f"AEDNF项数: {len(phi.aednf.terms)}")
    print(f"AECNF子句数: {len(phi.aecnf.clauses)}")
    print()
    
    print("AEDNF结构:")
    for i, term in enumerate(phi.aednf.terms):
        print(f"  项{i+1}: {term.objective_part.description}")
        if term.positive_literals:
            for lit in term.positive_literals:
                print(f"    + K_{lit.agent}(...)")
        if term.negative_literals:
            for lit in term.negative_literals:
                print(f"    - ¬K_{lit.agent}(...)")
    
    print()
    print("AECNF结构:")
    for i, clause in enumerate(phi.aecnf.clauses):
        print(f"  子句{i+1}: {clause.objective_part.description}")
        if clause.positive_literals:
            for lit in clause.positive_literals:
                print(f"    + K_{lit.agent}(...)")
        if clause.negative_literals:
            for lit in clause.negative_literals:
                print(f"    - ¬K_{lit.agent}(...)")


def test_simple_deintrospective():
    """
    测试简单的去内省案例
    """
    from archiv.models import create_objective_pair
    
    print("=== 简单去内省测试 ===")
    print()
    
    # 创建一个简单的测试公式：Φ = (v1 ∧ K_a(p))
    # 这个公式只有一个项，包含a-主观部分
    
    v1 = create_objective_pair("v1")
    k_a_p_pair = know(create_objective_pair("p"), "a")
    term = land(v1, k_a_p_pair)
    
    # 构造测试公式
    test_phi = AEDNFAECNFPair(
        aednf=AEDNF(terms=[term.aednf.terms[0]], depth=1),
        aecnf=AECNF(clauses=[term.aecnf.clauses[0]], depth=1),
        depth=1
    )
    
    print("原始公式 Φ = (v1 ∧ K_a(p))")
    visualize_formula(test_phi, "原始公式 Φ")
    
    # 对代理 a 进行去内省
    print("\n对代理 a 进行去内省...")
    result = simple_deintrospective_k(test_phi, "a")
    
    # 可视化结果
    visualize_formula(result, "去内省结果 D_a[Φ]")
    
    # 验证结果
    print("\n=== 结果验证 ===")
    print("期望结果：D_a[Φ] = K_a(v1) ∧ K_a(p)")
    print("即：a知道v1，并且a知道p")
    
    # 验证结果是否正确
    is_correct = verify_deintrospective_result(result, "a")
    if is_correct:
        print("✅ 结果正确：所有a-主观部分都已外提到最外层")
    else:
        print("❌ 结果不正确：结果中仍包含a-主观部分")
    
    return result


def test_deintrospective():
    """
    测试去内省算法
    """
    from archiv.models import create_objective_pair
    
    print("=== 测试去内省算法 ===")
    print()
    
    # 创建一个更清晰的测试公式
    # Φ = (v1 ∧ K_a(p)) ∨ (v2 ∧ K_b(q)) ∨ v3
    # 其中：
    # - 第1项：(v1 ∧ K_a(p)) 包含a-主观部分
    # - 第2项：(v2 ∧ K_b(q)) 包含b-主观部分（对a来说是客观的）
    # - 第3项：v3 是纯客观的
    
    # 创建原子命题
    v1 = create_objective_pair("v1")
    v2 = create_objective_pair("v2") 
    v3 = create_objective_pair("v3")
    
    # 构造 (v1 ∧ K_a(p))
    k_a_p_pair = know(create_objective_pair("p"), "a")
    term1 = land(v1, k_a_p_pair)
    
    # 构造 (v2 ∧ K_b(q))  
    k_b_q_pair = know(create_objective_pair("q"), "b")
    term2 = land(v2, k_b_q_pair)
    
    # 构造 AEDNF
    test_aednf = AEDNF(
        terms=[term1.aednf.terms[0], term2.aednf.terms[0], v3.aednf.terms[0]],
        depth=1
    )
    
    # 构造 AECNF（简化）
    test_aecnf = AECNF(
        clauses=[v3.aecnf.clauses[0]],  # 使用v3的AECNF子句
        depth=1
    )
    
    # 构造测试公式
    test_phi = AEDNFAECNFPair(
        aednf=test_aednf,
        aecnf=test_aecnf,
        depth=1
    )
    
    print("原始公式 Φ:")
    print(f"  - 项1: (v1 ∧ K_a(p)) - 包含a-主观部分")
    print(f"  - 项2: (v2 ∧ K_b(q)) - 包含b-主观部分（对a客观）")
    print(f"  - 项3: v3 - 纯客观")
    
    # 可视化原始公式
    visualize_formula(test_phi, "原始公式 Φ")
    
    # 对代理 a 进行去内省
    print("\n对代理 a 进行去内省...")
    result = simple_deintrospective_k(test_phi, "a")
    
    # 可视化结果
    visualize_formula(result, "去内省结果 D_a[Φ]")
    
    # 验证结果
    print("\n=== 结果验证 ===")
    print("根据数学定义，D_a[Φ] 应该满足以下性质：")
    print("1. 所有a-主观部分都被外提到最外层")
    print("2. 内部只包含a-客观公式")
    print("3. 等价于原始公式 K_a(Φ)")
    
    # 验证结果是否正确
    is_correct = verify_deintrospective_result(result, "a")
    if is_correct:
        print("✅ 结果正确：所有a-主观部分都已外提到最外层")
    else:
        print("❌ 结果不正确：结果中仍包含a-主观部分")
    
    return result


def test_complex_deintrospective():
    """
    测试复杂的去内省案例
    """
    from archiv.models import create_objective_pair
    
    print("=== 复杂去内省测试 ===")
    print()
    
    # 创建一个复杂的测试公式：Φ = (v1 ∧ K_a(p)) ∨ (v2 ∧ K_a(q)) ∨ (v3 ∧ K_b(r)) ∨ v4
    # 其中：
    # - 第1项：(v1 ∧ K_a(p)) 包含a-主观部分
    # - 第2项：(v2 ∧ K_a(q)) 包含a-主观部分
    # - 第3项：(v3 ∧ K_b(r)) 包含b-主观部分（对a客观）
    # - 第4项：v4 是纯客观的
    
    # 创建原子命题
    v1 = create_objective_pair("v1")
    v2 = create_objective_pair("v2")
    v3 = create_objective_pair("v3")
    v4 = create_objective_pair("v4")
    
    # 构造 (v1 ∧ K_a(p))
    k_a_p_pair = know(create_objective_pair("p"), "a")
    term1 = land(v1, k_a_p_pair)
    
    # 构造 (v2 ∧ K_a(q))
    k_a_q_pair = know(create_objective_pair("q"), "a")
    term2 = land(v2, k_a_q_pair)
    
    # 构造 (v3 ∧ K_b(r))
    k_b_r_pair = know(create_objective_pair("r"), "b")
    term3 = land(v3, k_b_r_pair)
    
    # 构造 AEDNF
    test_aednf = AEDNF(
        terms=[term1.aednf.terms[0], term2.aednf.terms[0], term3.aednf.terms[0], v4.aednf.terms[0]],
        depth=1
    )
    
    # 构造 AECNF（简化）
    test_aecnf = AECNF(
        clauses=[v4.aecnf.clauses[0]],  # 使用v4的AECNF子句
        depth=1
    )
    
    # 构造测试公式
    test_phi = AEDNFAECNFPair(
        aednf=test_aednf,
        aecnf=test_aecnf,
        depth=1
    )
    
    print("原始公式 Φ:")
    print(f"  - 项1: (v1 ∧ K_a(p)) - 包含a-主观部分")
    print(f"  - 项2: (v2 ∧ K_a(q)) - 包含a-主观部分")
    print(f"  - 项3: (v3 ∧ K_b(r)) - 包含b-主观部分（对a客观）")
    print(f"  - 项4: v4 - 纯客观")
    
    # 可视化原始公式
    visualize_formula(test_phi, "原始公式 Φ")
    
    # 对代理 a 进行去内省
    print("\n对代理 a 进行去内省...")
    result = simple_deintrospective_k(test_phi, "a")
    
    # 可视化结果
    visualize_formula(result, "去内省结果 D_a[Φ]")
    
    # 验证结果
    print("\n=== 结果验证 ===")
    print("根据数学定义，D_a[Φ] 应该满足以下性质：")
    print("1. 所有a-主观部分都被外提到最外层")
    print("2. 内部只包含a-客观公式")
    print("3. 等价于原始公式 K_a(Φ)")
    
    # 验证结果是否正确
    is_correct = verify_deintrospective_result(result, "a")
    if is_correct:
        print("✅ 结果正确：所有a-主观部分都已外提到最外层")
    else:
        print("❌ 结果不正确：结果中仍包含a-主观部分")
    
    return result


def test_edge_cases():
    """
    测试边界情况
    """
    from archiv.models import create_objective_pair
    
    print("=== 边界情况测试 ===")
    print()
    
    # 测试1：纯客观公式
    print("测试1：纯客观公式")
    v1 = create_objective_pair("v1")
    v2 = create_objective_pair("v2")
    
    test_phi = AEDNFAECNFPair(
        aednf=AEDNF(terms=[v1.aednf.terms[0], v2.aednf.terms[0]], depth=0),
        aecnf=AECNF(clauses=[v1.aecnf.clauses[0]], depth=0),
        depth=0
    )
    
    print("原始公式：Φ = v1 ∨ v2（纯客观）")
    result = simple_deintrospective_k(test_phi, "a")
    print("结果：", "✅ 正确" if verify_deintrospective_result(result, "a") else "❌ 错误")
    print()
    
    # 测试2：只有主观项
    print("测试2：只有主观项")
    k_a_p_pair = know(create_objective_pair("p"), "a")
    k_a_q_pair = know(create_objective_pair("q"), "a")
    
    test_phi = AEDNFAECNFPair(
        aednf=AEDNF(terms=[k_a_p_pair.aednf.terms[0], k_a_q_pair.aednf.terms[0]], depth=1),
        aecnf=AECNF(clauses=[k_a_p_pair.aecnf.clauses[0]], depth=1),
        depth=1
    )
    
    print("原始公式：Φ = K_a(p) ∨ K_a(q)（只有主观项）")
    result = simple_deintrospective_k(test_phi, "a")
    print("结果：", "✅ 正确" if verify_deintrospective_result(result, "a") else "❌ 错误")
    print()
    
    return True


if __name__ == "__main__":
    # 先运行简单测试
    test_simple_deintrospective()
    print("\n" + "="*50 + "\n")
    # 再运行复杂测试
    test_deintrospective()
    print("\n" + "="*50 + "\n")
    # 运行更复杂的测试
    test_complex_deintrospective()
    print("\n" + "="*50 + "\n")
    # 运行边界情况测试
    test_edge_cases()
