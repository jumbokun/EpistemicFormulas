from typing import List, Tuple, Optional
from archiv.models import AEDNFAECNFPair, AEDNF, AECNF, AEDNFTerm, AECNFClause, KnowledgeLiteral, ObjectiveFormula
from archiv.logical_operations import land, know
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
        disj = [clause.objective_part.description or "⊥"]
        for lit in clause.negative_literals:
            nested = _fol_of_pair(lit.formula, prefer_aednf=False)
            disj.append(f"¬K_{lit.agent}({nested})")
        for lit in clause.positive_literals:
            nested = _fol_of_pair(lit.formula, prefer_aednf=False)
            disj.append(f"K_{lit.agent}({nested})")
        return " ∨ ".join(disj)

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
    
    # 首先重新排序：主观项在前，客观项在后
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
    应用简单的去内省公式
    简化理解：对于 Φ = (Ω₁ ∧ Θ₁) ∨ (Ω₂ ∧ Θ₂) ∨ ... ∨ (Ω_m ∧ Θ_m) ∨ ⋁_{i=m+1}^n Ω_i
    我们希望得到：D_a[Φ] = K_a(⋁_{i=m+1}^n Ω_i) ∧ Θ₁ ∧ Θ₂ ∧ ... ∧ Θ_m
    """
    n = len(phi.aednf.terms)
    detailed_logging = bool(logger and logger.enabled)
    if detailed_logging:
        logger.log(f"应用简单去内省公式 - m={m}, n={n}")
    
    if m == 0:
        # 没有主观项，直接返回 K_a(phi)
        if detailed_logging:
            logger.log(f"m=0，没有主观项，直接返回 K_{agent}(Φ)")
        return know(phi, agent)
    
    # 构造客观部分：⋁_{i=m+1}^n Ω_i
    objective_terms = []
    if detailed_logging and logger and logger.enabled:
        with logger.block(f"Step 1: 构造客观部分 O = ⋁ Ω_i (i∈[{m+1}, {n}])"):
            for i in range(m, n):
                omega, _ = decompose_term_by_agent(phi.aednf.terms[i], agent, logger)
                logger.log(f"Ω_{i+1}: obj={omega.objective_part.description} +{[lit.agent for lit in omega.positive_literals]} -{[lit.agent for lit in omega.negative_literals]}")
                objective_terms.append(omega)
    else:
        if detailed_logging:
            logger.log(f"开始构造客观部分，范围: [{m}, {n})")
        for i in range(m, n):
            omega, _ = decompose_term_by_agent(phi.aednf.terms[i], agent, logger)
            if detailed_logging:
                logger.log(f"项 {i+1} 的客观部分: {omega.objective_part.description} +{[lit.agent for lit in omega.positive_literals]} -{[lit.agent for lit in omega.negative_literals]}")
            objective_terms.append(omega)
    
    # 如果 m=n，说明所有项都是主观的，我们需要从主观项中提取客观部分
    if m == n:
        if detailed_logging and logger and logger.enabled:
            with logger.block(f"m=n={m}：所有项主观，从主观项提取 Ω_i"):
                for i in range(m):
                    omega, _ = decompose_term_by_agent(phi.aednf.terms[i], agent, logger)
                    if omega is not None:
                        logger.log(f"Ω_{i+1}: obj={omega.objective_part.description} +{[lit.agent for lit in omega.positive_literals]} -{[lit.agent for lit in omega.negative_literals]}")
                        objective_terms.append(omega)
        else:
            if detailed_logging:
                logger.log(f"m=n={m}，所有项都是主观的，从主观项中提取客观部分")
            for i in range(m):
                omega, _ = decompose_term_by_agent(phi.aednf.terms[i], agent, logger)
                if omega is not None:
                    if detailed_logging:
                        logger.log(f"主观项 {i+1} 的客观部分: {omega.objective_part.description} +{[lit.agent for lit in omega.positive_literals]} -{[lit.agent for lit in omega.negative_literals]}")
                    objective_terms.append(omega)
    
    if detailed_logging:
        logger.log(f"客观部分 O 构造完成，共 {len(objective_terms)} 项")
        # 打印 O 的展开
        o_phi = AEDNFAECNFPair(
            aednf=AEDNF(terms=objective_terms if objective_terms else [AEDNFTerm(objective_part=ObjectiveFormula(obdd_node_id=false_node.id, description="⊥"))], depth=phi.depth),
            aecnf=phi.aecnf,
            depth=phi.depth
        )
        logger.log(f"O（AEDNF）= {_fol_of_pair(o_phi, prefer_aednf=True)}")
    
    # 构造主观部分：Θ₁ ∧ Θ₂ ∧ ... ∧ Θ_m
    subjective_terms = []
    if detailed_logging and logger and logger.enabled:
        with logger.block(f"Step 2: 构造主观部分 T = ⋀ Θ_i (i∈[1, {m}])"):
            for i in range(m):
                _, theta = decompose_term_by_agent(phi.aednf.terms[i], agent, logger)
                if theta is not None:
                    logger.log(f"Θ_{i+1}: obj={theta.objective_part.description} +{[lit.agent for lit in theta.positive_literals]} -{[lit.agent for lit in theta.negative_literals]}")
                    subjective_terms.append(theta)
            logger.log(f"T 项数 = {len(subjective_terms)}")
    else:
        if detailed_logging:
            logger.log(f"开始构造主观部分，范围: [0, {m})")
        for i in range(m):
            _, theta = decompose_term_by_agent(phi.aednf.terms[i], agent, logger)
            if theta is not None:
                if detailed_logging:
                    logger.log(f"项 {i+1} 的主观部分: {theta.objective_part.description} +{[lit.agent for lit in theta.positive_literals]} -{[lit.agent for lit in theta.negative_literals]}")
                subjective_terms.append(theta)
    
    if detailed_logging:
        logger.log(f"主观部分构造完成，共 {len(subjective_terms)} 项")
    
    # 构造结果：K_a(客观部分) ∧ 主观部分
    if objective_terms:
        # 有客观部分
        if detailed_logging:
            logger.log(f"Step 3: 构造 K_{agent}(O)")
        objective_phi = AEDNFAECNFPair(
            aednf=AEDNF(terms=objective_terms, depth=phi.depth),
            aecnf=phi.aecnf,
            depth=phi.depth
        )
        k_objective = know(objective_phi, agent)
        if detailed_logging:
            logger.log(f"K_{agent}(O) 构造完成，深度: {k_objective.depth}")
            logger.log(f"O（AEDNF）= {_fol_of_pair(objective_phi, prefer_aednf=True)}")
        
        if subjective_terms:
            # 有主观部分，需要与客观部分结合
            if detailed_logging:
                logger.log(f"Step 4: 合并 R = K_{agent}(O) ∧ T")
            result = k_objective
            for idx, theta in enumerate(subjective_terms):
                if detailed_logging:
                    logger.log(f"合并 Θ_{idx+1}/{len(subjective_terms)}")
                theta_pair = AEDNFAECNFPair(
                    aednf=AEDNF(terms=[theta], depth=0),
                    aecnf=AECNF(clauses=[AECNFClause(
                        objective_part=theta.objective_part,
                        positive_literals=theta.positive_literals,
                        negative_literals=theta.negative_literals
                    )], depth=0),
                    depth=0
                )
                if detailed_logging:
                    logger.log(f"主观部分 {idx+1} 转换为 Pair，深度: {theta_pair.depth}")
                result = land(result, theta_pair)
                if detailed_logging:
                    logger.log(f"当前 R（AEDNF）= {_fol_of_pair(result, prefer_aednf=True)}；depth={result.depth}")
            if detailed_logging:
                logger.log(f"完成：D_{agent}[Φ]（AEDNF）= {_fol_of_pair(result, prefer_aednf=True)}")
            return result
        else:
            # 没有主观部分
            if detailed_logging:
                logger.log(f"没有主观部分，直接返回 K_{agent}(客观部分)")
            return k_objective
    else:
        # 没有客观部分，只有主观部分
        if detailed_logging:
            logger.log(f"没有客观部分，只有主观部分")
        if subjective_terms:
            # 只有主观部分，直接返回 K_a(phi)
            if detailed_logging:
                logger.log(f"只有主观部分，直接返回 K_{agent}(Φ)")
            return know(phi, agent)
        else:
            # 既没有客观部分也没有主观部分（不应该发生）
            if detailed_logging:
                logger.log(f"既没有客观部分也没有主观部分（不应该发生），返回 K_{agent}(Φ)")
            return know(phi, agent)

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
    return result_pair

