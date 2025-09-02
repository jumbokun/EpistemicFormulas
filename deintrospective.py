from typing import List, Tuple, Optional
from archiv.models import AEDNFAECNFPair, AEDNF, AECNF, AEDNFTerm, AECNFClause, KnowledgeLiteral, ObjectiveFormula
from archiv.logical_operations import land, lor, know, normalize_pair
from archiv.obdd import true_node, false_node


class TraceLogger:
    """极简缩进式调试日志器，用于清晰展示递归/分步转换轨迹。"""
    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self._indent = 0

    def log(self, message: str):
        if not self.enabled:
            return
        print(f"TRACE: {'  ' * self._indent}{message}")

    def push(self, title: Optional[str] = None):
        if title:
            self.log(title)
        self._indent += 1

    def pop(self):
        self._indent = max(0, self._indent - 1)

    def block(self, title: str):
        class _Ctx:
            def __init__(self, logger: 'TraceLogger', title_msg: str):
                self.logger = logger
                self.title_msg = title_msg
            def __enter__(self):
                self.logger.push(self.title_msg)
            def __exit__(self, exc_type, exc, tb):
                self.logger.pop()
        return _Ctx(self, title)


def _is_obj_true(obj: ObjectiveFormula) -> bool:
    return obj.obdd_node_id == true_node.id or obj.description == "⊤"


def _is_obj_false(obj: ObjectiveFormula) -> bool:
    return obj.obdd_node_id == false_node.id or obj.description == "⊥"


def _is_trivially_true_pair(p: AEDNFAECNFPair) -> bool:
    try:
        if len(p.aednf.terms) == 1:
            t = p.aednf.terms[0]
            if _is_obj_true(t.objective_part) and not t.positive_literals and not t.negative_literals:
                return True
        if len(p.aecnf.clauses) == 1:
            c = p.aecnf.clauses[0]
            if _is_obj_true(c.objective_part) and not c.positive_literals and not c.negative_literals:
                return True
    except Exception:
        pass
    return False


def _is_trivially_false_pair(p: AEDNFAECNFPair) -> bool:
    try:
        if len(p.aednf.terms) == 1:
            t = p.aednf.terms[0]
            if _is_obj_false(t.objective_part) and not t.positive_literals and not t.negative_literals:
                return True
        if len(p.aecnf.clauses) == 1:
            c = p.aecnf.clauses[0]
            if _is_obj_false(c.objective_part) and not c.positive_literals and not c.negative_literals:
                return True
    except Exception:
        pass
    return False


def deep_simplify_pair(phi: AEDNFAECNFPair) -> AEDNFAECNFPair:
    """
    递归化简整个 Pair：
      - 先对所有嵌套在 K/¬K 内部的 formula 做深度化简
      - 再在当前层做常量折叠（删除 ⊤∧、⊥∨、K(⊥)、¬K(⊤) 等）
    """
    # 深化简 AEDNF
    new_terms: List[AEDNFTerm] = []
    for term in phi.aednf.terms:
        if _is_obj_false(term.objective_part):
            continue
        drop_term = False
        kept_pos: List[KnowledgeLiteral] = []
        kept_neg: List[KnowledgeLiteral] = []
        for lit in term.positive_literals:
            sub = deep_simplify_pair(lit.formula)
            # 应用字面值层常量规则
            if _is_trivially_false_pair(sub):
                drop_term = True
                break
            if _is_trivially_true_pair(sub):
                continue
            kept_pos.append(KnowledgeLiteral(agent=lit.agent, formula=sub, negated=False, depth=lit.depth))
        if drop_term:
            continue
        for lit in term.negative_literals:
            sub = deep_simplify_pair(lit.formula)
            if _is_trivially_true_pair(sub):
                drop_term = True
                break
            if _is_trivially_false_pair(sub):
                continue
            kept_neg.append(KnowledgeLiteral(agent=lit.agent, formula=sub, negated=True, depth=lit.depth))
        if drop_term:
            continue
        new_terms.append(AEDNFTerm(
            objective_part=term.objective_part,
            positive_literals=kept_pos,
            negative_literals=kept_neg
        ))
    if not new_terms:
        new_terms = [AEDNFTerm(objective_part=ObjectiveFormula(obdd_node_id=false_node.id, description="⊥"))]

    # 深化简 AECNF
    new_clauses: List[AECNFClause] = []
    for clause in phi.aecnf.clauses:
        if _is_obj_true(clause.objective_part):
            new_clauses.append(AECNFClause(objective_part=ObjectiveFormula(obdd_node_id=true_node.id, description="⊤")))
            continue
        clause_true = False
        kept_pos: List[KnowledgeLiteral] = []
        kept_neg: List[KnowledgeLiteral] = []
        for lit in clause.positive_literals:
            sub = deep_simplify_pair(lit.formula)
            if _is_trivially_true_pair(sub):
                clause_true = True
                break
            if _is_trivially_false_pair(sub):
                continue
            kept_pos.append(KnowledgeLiteral(agent=lit.agent, formula=sub, negated=False, depth=lit.depth))
        if clause_true:
            new_clauses.append(AECNFClause(objective_part=ObjectiveFormula(obdd_node_id=true_node.id, description="⊤")))
            continue
        for lit in clause.negative_literals:
            sub = deep_simplify_pair(lit.formula)
            if _is_trivially_false_pair(sub):
                clause_true = True
                break
            if _is_trivially_true_pair(sub):
                continue
            kept_neg.append(KnowledgeLiteral(agent=lit.agent, formula=sub, negated=True, depth=lit.depth))
        if clause_true:
            new_clauses.append(AECNFClause(objective_part=ObjectiveFormula(obdd_node_id=true_node.id, description="⊤")))
            continue
        new_clauses.append(AECNFClause(
            objective_part=clause.objective_part,
            positive_literals=kept_pos,
            negative_literals=kept_neg
        ))

    simplified = AEDNFAECNFPair(
        aednf=AEDNF(terms=new_terms, depth=phi.aednf.depth),
        aecnf=AECNF(clauses=new_clauses, depth=phi.aecnf.depth),
        depth=phi.depth
    )
    # 当前层再做一遍浅常量化简以去除可能新形成的项/子句
    return simplify_constants_in_pair(simplified)


def simplify_constants_in_pair(phi: AEDNFAECNFPair) -> AEDNFAECNFPair:
    """
    常量化简（浅层）：
    - AEDNF 项：
        - 目标部为 ⊥ -> 移除该项
        - 含 K_a(⊤) 或 ¬K_a(⊥) -> 删除这些总为真的文字
        - 含 K_a(⊥) 或 ¬K_a(⊤) -> 整项为假 -> 移除该项
    - AECNF 子句：
        - 目标部为 ⊤ -> 整子句为真
        - 删除 K_a(⊥) 或 ¬K_a(⊤) 这些总为假的析取项
        - 若含 K_a(⊤) 或 ¬K_a(⊥) -> 整子句为真
    返回同构的新 Pair。
    """
    # AEDNF 处理
    new_terms: List[AEDNFTerm] = []
    for term in phi.aednf.terms:
        if _is_obj_false(term.objective_part):
            continue
        drop_term = False
        kept_pos: List[KnowledgeLiteral] = []
        kept_neg: List[KnowledgeLiteral] = []
        for lit in term.positive_literals:
            if _is_trivially_false_pair(lit.formula):
                drop_term = True
                break
            if _is_trivially_true_pair(lit.formula):
                continue
            kept_pos.append(lit)
        if drop_term:
            continue
        for lit in term.negative_literals:
            if _is_trivially_true_pair(lit.formula):
                drop_term = True
                break
            if _is_trivially_false_pair(lit.formula):
                continue
            kept_neg.append(lit)
        if drop_term:
            continue
        new_terms.append(AEDNFTerm(
            objective_part=term.objective_part,
            positive_literals=kept_pos,
            negative_literals=kept_neg
        ))
    if not new_terms:
        new_terms = [AEDNFTerm(objective_part=ObjectiveFormula(obdd_node_id=false_node.id, description="⊥"))]

    # AECNF 处理
    new_clauses: List[AECNFClause] = []
    for clause in phi.aecnf.clauses:
        if _is_obj_true(clause.objective_part):
            new_clauses.append(AECNFClause(objective_part=ObjectiveFormula(obdd_node_id=true_node.id, description="⊤")))
            continue
        clause_true = False
        kept_pos: List[KnowledgeLiteral] = []
        kept_neg: List[KnowledgeLiteral] = []
        for lit in clause.positive_literals:
            if _is_trivially_true_pair(lit.formula):
                clause_true = True
                break
            if _is_trivially_false_pair(lit.formula):
                continue
            kept_pos.append(lit)
        if clause_true:
            new_clauses.append(AECNFClause(objective_part=ObjectiveFormula(obdd_node_id=true_node.id, description="⊤")))
            continue
        for lit in clause.negative_literals:
            if _is_trivially_false_pair(lit.formula):
                clause_true = True
                break
            if _is_trivially_true_pair(lit.formula):
                continue
            kept_neg.append(lit)
        if clause_true:
            new_clauses.append(AECNFClause(objective_part=ObjectiveFormula(obdd_node_id=true_node.id, description="⊤")))
            continue
        new_clauses.append(AECNFClause(
            objective_part=clause.objective_part,
            positive_literals=kept_pos,
            negative_literals=kept_neg
        ))

    return AEDNFAECNFPair(
        aednf=AEDNF(terms=new_terms, depth=phi.aednf.depth),
        aecnf=AECNF(clauses=new_clauses, depth=phi.aecnf.depth),
        depth=phi.depth
    )


def _fol_of_pair(phi: AEDNFAECNFPair, prefer_aednf: bool = True) -> str:
    """本地化的公式字符串化（仅用于日志）。"""
    def _fol_of_term(term: AEDNFTerm) -> str:
        conj = [term.objective_part.description or "⊤"]
        for lit in term.positive_literals:
            nested = _fol_of_pair(lit.formula, prefer_aednf=True)
            conj.append(f"K_{lit.agent}({nested})")
        for lit in term.negative_literals:
            nested = _fol_of_pair(lit.formula, prefer_aednf=True)
            conj.append(f"¬K_{lit.agent}({nested})")
        return " ∧ ".join(conj)

    def _fol_of_clause(clause: AECNFClause) -> str:
        disj: List[str] = []
        has_literals = bool(clause.negative_literals or clause.positive_literals)
        obj_desc = clause.objective_part.description
        # 打印客观部：若为 ⊤/⊥ 且存在其它析取项，则省略输出该常量
        if obj_desc == "⊤":
            if not has_literals:
                disj.append("⊤")
        elif obj_desc == "⊥":
            if not has_literals:
                disj.append("⊥")
        else:
            disj.append(obj_desc or "⊥")
        for lit in clause.negative_literals:
            nested = _fol_of_pair(lit.formula, prefer_aednf=False)
            disj.append(f"¬K_{lit.agent}({nested})")
        for lit in clause.positive_literals:
            nested = _fol_of_pair(lit.formula, prefer_aednf=False)
            disj.append(f"K_{lit.agent}({nested})")
        return " ∨ ".join(disj) if disj else "⊥"

    if prefer_aednf:
        if phi.aednf.terms:
            parts = [_fol_of_term(t) for t in phi.aednf.terms]
            return " ∨ ".join(parts) if len(parts) == 1 else " ∨ ".join(f"({p})" for p in parts)
        return "⊥"
    else:
        if phi.aecnf.clauses:
            parts = [_fol_of_clause(c) for c in phi.aecnf.clauses]
            return " ∧ ".join(parts) if len(parts) == 1 else " ∧ ".join(f"({p})" for p in parts)
        return "⊤"


def decompose_term_by_agent(term: AEDNFTerm, agent: str, logger: Optional['TraceLogger'] = None) -> Tuple[AEDNFTerm, Optional[AEDNFTerm]]:
    """
    将项按代理分解为 (Ω_i, Θ_i) 对
    Ω_i: a-客观部分（命题部分 + 非a代理的知识文字）
    Θ_i: a-主观部分（a代理的知识文字）
    """
    # 是否显示详细日志
    detailed_logging = False
    if logger and logger.enabled:
        detailed_logging = True
    else:
        # 兼容旧的调用栈检测（仅用于 validate_deintrospective_pipeline 的 step 4）
        import inspect
        frame = inspect.currentframe()
        while frame:
            if 'validate_deintrospective_pipeline.py' in frame.f_code.co_filename and 'step 4' in str(frame.f_locals.get('step', '')):
                detailed_logging = True
                break
            frame = frame.f_back
    
    # 分离知识文字
    agent_literals = []      # Θ_i 部分
    other_literals = []      # Ω_i 部分
    
    for literal in term.positive_literals + term.negative_literals:
        if literal.agent == agent:
            agent_literals.append(literal)
        else:
            other_literals.append(literal)
    
    if detailed_logging:
        msg = f"decompose_term_by_agent: 项包含 {len(agent_literals)} 个 {agent}-文字，{len(other_literals)} 个其他文字"
        if logger and logger.enabled:
            logger.log(msg)
        else:
            print(f"DEBUG: {msg}")
    
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
        if detailed_logging:
            msg = f"decompose_term_by_agent: 返回 (Ω_i, Θ_i)，其中 Θ_i 包含 {len(agent_literals)} 个 {agent}-文字"
            if logger and logger.enabled:
                logger.log(msg)
            else:
                print(f"DEBUG: {msg}")
        return omega_i, theta_i
    else:
        if detailed_logging:
            msg = f"decompose_term_by_agent: 返回 (Ω_i, None)，无 {agent}-文字"
            if logger and logger.enabled:
                logger.log(msg)
            else:
                print(f"DEBUG: {msg}")
        return omega_i, None

def find_critical_index(phi: AEDNFAECNFPair, agent: str, logger: Optional['TraceLogger'] = None) -> int:
    """
    找到临界点 ℓ_Φ：最大的非a-客观子句的索引
    当 ℓ_Φ = 0 时，整个公式都是 a-客观的
    """
    # 计算包含a-主观部分的项的数量
    subjective_count = 0
    for term in phi.aednf.terms:
        omega, theta = decompose_term_by_agent(term, agent, logger)
        if theta is not None:  # 包含a-主观部分
            subjective_count += 1
        else:  # 找到第一个纯客观项，停止计数
            break
    return subjective_count

def simple_deintrospective_k(phi: AEDNFAECNFPair, agent: str, logger: Optional['TraceLogger'] = None) -> AEDNFAECNFPair:
    """
    简单但正确的去内省算法实现
    目标：将 K_a(Φ) 转换为等价形式，其中所有a-主观部分都被外提到最外层
    """
    # 详细日志开关：优先使用传入的 logger，其次兼容旧方式
    detailed_logging = bool(logger and logger.enabled)
    if not detailed_logging:
        import inspect
        frame = inspect.currentframe()
        while frame:
            if 'validate_deintrospective_pipeline.py' in frame.f_code.co_filename and 'step 4' in str(frame.f_locals.get('step', '')):
                detailed_logging = True
                break
            frame = frame.f_back
    if detailed_logging and logger and logger.enabled:
        logger.log(f"开始简单去内省算法（K_{agent}；AEDNF 视图）")
    
    # 先做深度化简（删除 true ∧ / false ∨ 等冗余）
    phi = deep_simplify_pair(phi)
    # 然后重新排序：主观项在前，客观项在后
    subjective_terms = []
    objective_terms = []
    
    for idx, term in enumerate(phi.aednf.terms, start=1):
        # 打印该项的知识分布
        pos_agents = [lit.agent for lit in term.positive_literals]
        neg_agents = [lit.agent for lit in term.negative_literals]
        if detailed_logging:
            msg = f"项#{idx} obj={term.objective_part.description} +{pos_agents} -{neg_agents} 针对代理 {agent}"
            if logger and logger.enabled:
                logger.log(msg)
            else:
                print(f"DEBUG: {msg}")
        omega, theta = decompose_term_by_agent(term, agent, logger)
        if theta is not None:  # 包含 a-主观部分
            if detailed_logging:
                msg = f"项#{idx} 判定为 subjective (含 {agent} 的知识文字)"
                if logger and logger.enabled:
                    logger.log(msg)
                else:
                    print(f"DEBUG: {msg}")
            subjective_terms.append(term)
        else:  # 纯 a-客观
            if detailed_logging:
                msg = f"项#{idx} 判定为 objective (不含 {agent} 的知识文字)"
                if logger and logger.enabled:
                    logger.log(msg)
                else:
                    print(f"DEBUG: {msg}")
            objective_terms.append(term)
    
    # 重新构造公式
    reordered_terms = subjective_terms + objective_terms
    reordered_phi = AEDNFAECNFPair(
        aednf=AEDNF(terms=reordered_terms, depth=phi.depth),
        aecnf=phi.aecnf,
        depth=phi.depth
    )
    
    if detailed_logging:
        msg = f"重新排序完成，主观项: {len(subjective_terms)}, 客观项: {len(objective_terms)}"
        if logger and logger.enabled:
            logger.log(msg)
        else:
            print(f"DEBUG: {msg}")
    
    # 找到重新排序后的临界点
    critical_index = find_critical_index(reordered_phi, agent, logger)
    if detailed_logging:
        msg = f"临界点 ℓ_Φ = {critical_index}"
        if logger and logger.enabled:
            logger.log(msg)
        else:
            print(f"DEBUG: {msg}")
    
    # 基本情形：整个公式都是 a-客观的
    if critical_index == 0:
        if detailed_logging:
            msg = f"基本情形，直接返回 K_{agent}(Φ)"
            if logger and logger.enabled:
                logger.log(msg)
            else:
                print(f"DEBUG: {msg}")
        return know(phi, agent)
    
    # 递归情形：存在 a-主观部分需要外提
    if detailed_logging:
        msg = f"递归情形，处理 {critical_index} 个 subjective clauses"
        if logger and logger.enabled:
            logger.log(msg)
        else:
            print(f"DEBUG: {msg}")
    
    # 应用简单的递归公式
    if detailed_logging and logger and logger.enabled:
        with logger.block("应用递归公式 D_a[Φ]"):
            result = apply_simple_deintrospective_formula(reordered_phi, agent, len(subjective_terms), logger)
            logger.log(f"结果（AEDNF）: {_fol_of_pair(result, prefer_aednf=True)}")
            return result
    else:
        return apply_simple_deintrospective_formula(reordered_phi, agent, len(subjective_terms), logger)

def apply_simple_deintrospective_formula(phi: AEDNFAECNFPair, agent: str, m: int, logger: Optional['TraceLogger'] = None) -> AEDNFAECNFPair:
    """
    递归实现 D_a[Φ]（AEDNF 视图），符合你给出的定义：
      若 ℓ_Φ = 0，则 D_a[Φ] := K_a(Φ)
      若 ℓ_Φ = m > 0，则
        D_a[Φ] := D_a[V_{m-1} ∨ ⋁_{i=m+1}^n Ω_i]
                 ∨ ⋁_{C∈C(D_a[V_{m-1} ∨ ⋁_{i=m}^n Ω_i])} (C ∧ Θ_m)

    其中：
      V_{k} = ⋁_{i=1}^{k} (Ω_i ∧ Θ_i)（按 a-主观项计数后的前缀）
      Ω_i/Θ_i 为针对代理 a 的项分解。
    """
    n = len(phi.aednf.terms)
    detailed_logging = bool(logger and logger.enabled)
    if detailed_logging:
        logger.log(f"应用递归 D_{agent}[Φ]：m={m}, n={n}")

    # 基本情形
    if m == 0:
        if detailed_logging:
            logger.log(f"ℓ_Φ=0：返回 K_{agent}(Φ)")
        return know(phi, agent)

    # 取第 m 项的分解（注意：phi 已经按主观在前排序）
    omega_m, theta_m = decompose_term_by_agent(phi.aednf.terms[m - 1], agent, logger)

    # 收集 i∈[m+1, n] 的 Ω_i（phi 中索引为 m..n-1 的项都是 a-客观项）
    omegas_after_m: List[AEDNFTerm] = []
    for i in range(m, n):
        omega_i, _ = decompose_term_by_agent(phi.aednf.terms[i], agent, logger)
        omegas_after_m.append(omega_i)

    # 构造 Left 输入：V_{m-1} ∨ ⋁_{i=m+1}^n Ω_i
    left_terms: List[AEDNFTerm] = list(phi.aednf.terms[:m - 1]) + omegas_after_m
    left_input: Optional[AEDNFAECNFPair] = None
    if left_terms:
        left_input = AEDNFAECNFPair(
            aednf=AEDNF(terms=left_terms, depth=phi.depth),
            aecnf=phi.aecnf,
            depth=phi.depth
        )

    # 构造 Right 输入：V_{m-1} ∨ Ω_m ∨ ⋁_{i=m+1}^n Ω_i
    right_terms: List[AEDNFTerm] = list(phi.aednf.terms[:m - 1]) + [omega_m] + omegas_after_m
    right_input = AEDNFAECNFPair(
        aednf=AEDNF(terms=right_terms, depth=phi.depth),
        aecnf=phi.aecnf,
        depth=phi.depth
    )

    left_da: Optional[AEDNFAECNFPair] = None
    if left_input is not None:
        if detailed_logging and logger and logger.enabled:
            with logger.block("Left 分支：D_a[V_{m-1} ∨ ⋁ Ω_{>m}]"):
                logger.log(f"Left 输入（AEDNF）= {_fol_of_pair(left_input, prefer_aednf=True)}")
        left_da = simple_deintrospective_k(left_input, agent, logger)
    else:
        if detailed_logging:
            logger.log("Left 分支为空（m=1 且无 Ω_{>m}），按 false 处理并跳过")

    if detailed_logging and logger and logger.enabled:
        with logger.block("Right 分支：D_a[V_{m-1} ∨ Ω_m ∨ ⋁ Ω_{>m}] 与 Θ_m 合取"):
            logger.log(f"Right 输入（AEDNF）= {_fol_of_pair(right_input, prefer_aednf=True)}")
    right_da = simple_deintrospective_k(right_input, agent, logger)

    # 将 Θ_m 转成 Pair，深度与 right_da 对齐以便合取时按高阶规则合并知识文字
    theta_pair = AEDNFAECNFPair(
        aednf=AEDNF(terms=[theta_m], depth=right_da.depth),
        aecnf=AECNF(clauses=[AECNFClause(
            objective_part=theta_m.objective_part,
            positive_literals=theta_m.positive_literals,
            negative_literals=theta_m.negative_literals
        )], depth=right_da.depth),
        depth=right_da.depth
    )

    right_conj = land(right_da, theta_pair)
    # 化简 Right 分支
    right_conj = normalize_pair(right_conj)

    # 汇总：Left ∨ (Right ∧ Θ_m)

    result = right_conj if left_da is None else lor(left_da, right_conj)
    # 最终结果化简
    result = normalize_pair(result)

    if detailed_logging:
        logger.log(f"完成一步：D_{agent}[Φ] = {_fol_of_pair(result, prefer_aednf=True)}")
    return simplify_constants_in_pair(result)

def extract_atomic_propositions(phi: AEDNFAECNFPair) -> List[str]:
    """
    从公式中提取所有原子命题
    """
    atoms = set()
    
    # 从AEDNF项中提取
    for term in phi.aednf.terms:
        # 从客观部分提取
        if hasattr(term.objective_part, 'description'):
            desc = term.objective_part.description
            if desc not in ['⊤', '⊥'] and desc.isalnum():
                atoms.add(desc)
        
        # 从知识文字中提取
        for lit in term.positive_literals + term.negative_literals:
            if hasattr(lit.formula, 'aednf'):
                for sub_term in lit.formula.aednf.terms:
                    if hasattr(sub_term.objective_part, 'description'):
                        desc = sub_term.objective_part.description
                        if desc not in ['⊤', '⊥'] and desc.isalnum():
                            atoms.add(desc)
    
    # 从AECNF子句中提取
    for clause in phi.aecnf.clauses:
        if hasattr(clause.objective_part, 'description'):
            desc = clause.objective_part.description
            if desc not in ['⊤', '⊥'] and desc.isalnum():
                atoms.add(desc)
    
    return list(atoms)

def is_a_objective_clause(clause: AECNFClause, agent: str) -> bool:
    """
    判断单个 AECNF 子句是否对代理 agent 客观（不含 agent 的知识文字）
    """
    for lit in clause.positive_literals + clause.negative_literals:
        if lit.agent == agent:
            return False
    return True

def build_pair_from_clause_parts(obj: ObjectiveFormula, pos_lits: List[KnowledgeLiteral], neg_lits: List[KnowledgeLiteral], depth: int) -> AEDNFAECNFPair:
    """
    由一个客观部分与知识文字列表，构造一个包含单个子句/项的 Pair。
    - AECNF: 单个子句 obj ∨ ⋁ pos ∨ ⋁ neg
    - AEDNF: 单个项   obj ∧ ⋀ pos ∧ ⋀ ¬neg
    """
    term = AEDNFTerm(
        objective_part=obj,
        positive_literals=[lit for lit in pos_lits if not lit.negated],
        negative_literals=[lit for lit in neg_lits if lit.negated]
    )
    clause = AECNFClause(
        objective_part=obj,
        positive_literals=pos_lits,
        negative_literals=neg_lits
    )
    aednf = AEDNF(terms=[term], depth=depth)
    aecnf = AECNF(clauses=[clause], depth=depth)
    return AEDNFAECNFPair(aednf=aednf, aecnf=aecnf, depth=depth)

def deintrospective_clause_a(clause: AECNFClause, agent: str, overall_depth: int) -> AEDNFAECNFPair:
    """
    对单个 AECNF 子句执行去内省：
    C_a[ α ∨ (⋁_{b} (¬K_b φ_b ∨ ⋁ K_b ψ_{b,j})) ]
      = (⋁ 现有的 a-文字) ∨ K_a( α ∨ (⋁_{b≠a} (¬K_b φ_b ∨ ⋁ K_b ψ_{b,j})) )
    返回一个只包含一个子句的 Pair。
    """
    # 划分 a 和 非 a 的知识文字
    a_pos: List[KnowledgeLiteral] = []
    a_neg: List[KnowledgeLiteral] = []
    other_pos: List[KnowledgeLiteral] = []
    other_neg: List[KnowledgeLiteral] = []

    for lit in clause.positive_literals:
        if lit.agent == agent:
            a_pos.append(lit)
        else:
            other_pos.append(lit)
    for lit in clause.negative_literals:
        if lit.agent == agent:
            a_neg.append(lit)
        else:
            other_neg.append(lit)

    # 残式：α ∨ (其它代理的知识文字)
    residual_pair = build_pair_from_clause_parts(
        obj=clause.objective_part,
        pos_lits=other_pos,
        neg_lits=other_neg,
        depth=overall_depth
    )

    # 构造 K_a(residual)
    ka_residual = KnowledgeLiteral(
        agent=agent,
        formula=residual_pair,
        negated=False,
        depth=residual_pair.depth + 1
    )

    # 新子句： (a-负文字) ∨ (a-正文字) ∨ K_a(residual)
    new_positive = list(a_pos) + [ka_residual]
    new_negative = list(a_neg)

    # AECNF 子句外部客观部分为空（⊥）
    new_clause = AECNFClause(
        objective_part=ObjectiveFormula(obdd_node_id=false_node.id, description="⊥"),
        positive_literals=new_positive,
        negative_literals=new_negative
    )

    # 构造单子句 Pair；
    # AEDNF 用 ⊤ 作客观部分，携带相同的知识文字，保证结构验证通过
    new_term = AEDNFTerm(
        objective_part=ObjectiveFormula(obdd_node_id=true_node.id, description="⊤"),
        positive_literals=new_positive,
        negative_literals=new_negative
    )

    new_aednf = AEDNF(terms=[new_term], depth=overall_depth + 1)
    new_aecnf = AECNF(clauses=[new_clause], depth=overall_depth + 1)
    return AEDNFAECNFPair(aednf=new_aednf, aecnf=new_aecnf, depth=overall_depth + 1)

def simple_deintrospective_c(phi: AEDNFAECNFPair, agent: str) -> AEDNFAECNFPair:
    """
    AECNF 版的去内省：对 K_a(Φ) 的子句级外提。
    输入为 Φ（AECNF 视图），输出 C_a[Φ]（仍以 Pair 表示）。
    结果深度 = phi.depth + 1。
    """
    # 先将 non-alternating（含 a-知识文字）子句放前面
    def is_alternating_for_agent(c: AECNFClause, a: str) -> bool:
        for lit in c.positive_literals + c.negative_literals:
            if lit.agent == a:
                return False
        return True

    non_alt = [c for c in phi.aecnf.clauses if not is_alternating_for_agent(c, agent)]
    alt = [c for c in phi.aecnf.clauses if is_alternating_for_agent(c, agent)]
    ordered = non_alt + alt

    result_pair: Optional[AEDNFAECNFPair] = None
    for idx, clause in enumerate(ordered):
        processed = deintrospective_clause_a(clause, agent, phi.depth)
        if result_pair is None:
            result_pair = processed
        else:
            result_pair = land(result_pair, processed)

    assert result_pair is not None, "AECNF 至少应有一个子句"
    return simplify_constants_in_pair(normalize_pair(result_pair))

