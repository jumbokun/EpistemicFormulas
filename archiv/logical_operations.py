from .models import AEDNFAECNFPair, AEDNF, AECNF, AEDNFTerm, AECNFClause, KnowledgeLiteral, ObjectiveFormula
from .obdd import negate, AND, NOT, OR, implies, true_node, false_node, conjoin, disjoin, reset_cache, Node, branch_cache, nodeID_2_key

def get_node_from_id(node_id: int) -> Node:
    """从节点ID获取Node对象"""
    if node_id == 0:  # false_node
        return false_node
    elif node_id == 1:  # true_node
        return true_node
    else:
        # 尝试从 nodeID_2_key 获取
        if node_id in nodeID_2_key:
            return branch_cache[nodeID_2_key[node_id]]
        
        # 如果不在 nodeID_2_key 中，尝试从 Node._instances 获取
        if node_id < len(Node._instances):
            return Node._instances[node_id]
        
        # 如果都不行，尝试从 branch_cache 的值中查找
        for node in branch_cache.values():
            if node.id == node_id:
                return node
        
        # 最后返回 false_node 作为默认值
        print(f"警告: 节点ID {node_id} 无法找到，使用 false_node 作为默认值")
        return false_node

def lnot(formula: AEDNFAECNFPair) -> AEDNFAECNFPair:
    """
    逻辑非
    新的AEDNF应该是原先的AECNF negate之后的结果
    """
    
    # 新的AEDNF = 原先AECNF的negate
    # 对于AECNF中的每个clause，我们需要将其转换为AEDNF的term
    new_terms = []
    
    for clause in formula.aecnf.clauses:
        # 将AECNF的clause转换为AEDNF的term
        # 对于clause: α ∨ ⋁_{a∈A} (¬K_a φ_a ∨ ⋁_j K_a ψ_{a,j})
        # 转换为term: ¬α ∧ ⋀_{a∈A} (K_a φ_a ∧ ⋀_j ¬K_a ψ_{a,j})
        
        # 处理客观部分：¬α
        # 获取原始OBDD节点并否定
        original_node = Node._instances[clause.objective_part.obdd_node_id] if clause.objective_part.obdd_node_id < len(Node._instances) else None
        if original_node:
            negated_node = negate(original_node)
            negated_objective = ObjectiveFormula(
                obdd_node_id=negated_node.id,
                description=f"¬({clause.objective_part.description})" if clause.objective_part.description else None
            )
        else:
            # 如果找不到节点，使用原始ID（fallback）
            negated_objective = ObjectiveFormula(
                obdd_node_id=clause.objective_part.obdd_node_id,
                description=f"¬({clause.objective_part.description})" if clause.objective_part.description else None
            )
        
        positive_literals = []
        negative_literals = []
        
        for lit in clause.positive_literals:
            negated_lit = KnowledgeLiteral(
                agent=lit.agent,
                formula=lit.formula,
                negated=True,
                depth=lit.depth
            )
            negative_literals.append(negated_lit)
        
        for lit in clause.negative_literals:
            negated_lit = KnowledgeLiteral(
                agent=lit.agent,
                formula=lit.formula,
                negated=False,
                depth=lit.depth
            )
            positive_literals.append(negated_lit)
        
        new_term = AEDNFTerm(
            objective_part=negated_objective,
            positive_literals=positive_literals,
            negative_literals=negative_literals
        )
        new_terms.append(new_term)
    
    new_aednf = AEDNF(
        terms=new_terms,
        depth=formula.depth # lnot 不会改变深度
    )
    
    new_clauses = []
    
    for term in formula.aednf.terms:
        original_node = Node._instances[term.objective_part.obdd_node_id] if term.objective_part.obdd_node_id < len(Node._instances) else None
        if original_node:
            negated_node = negate(original_node)
            negated_objective = ObjectiveFormula(
                obdd_node_id=negated_node.id,
                description=f"¬({term.objective_part.description})" if term.objective_part.description else None
            )
        else:
            # 如果找不到节点，使用原始ID（fallback）
            negated_objective = ObjectiveFormula(
                obdd_node_id=term.objective_part.obdd_node_id,
                description=f"¬({term.objective_part.description})" if term.objective_part.description else None
            )
        
        positive_literals = []
        negative_literals = []
        
        for lit in term.positive_literals:
            negated_lit = KnowledgeLiteral(
                agent=lit.agent,
                formula=lit.formula,
                negated=True,
                depth=lit.depth
            )
            negative_literals.append(negated_lit)
        
        for lit in term.negative_literals:
            negated_lit = KnowledgeLiteral(
                agent=lit.agent,
                formula=lit.formula,
                negated=False,
                depth=lit.depth
            )
            positive_literals.append(negated_lit)
        
        new_clause = AECNFClause(
            objective_part=negated_objective,
            positive_literals=positive_literals,
            negative_literals=negative_literals
        )
        new_clauses.append(new_clause)
    
    new_aecnf = AECNF(
        clauses=new_clauses,
        depth=formula.aednf.depth
    )
    
    return AEDNFAECNFPair(
        aednf=new_aednf,
        aecnf=new_aecnf,
        depth=formula.depth
    )

def land(formula1: AEDNFAECNFPair, formula2: AEDNFAECNFPair) -> AEDNFAECNFPair:
    """
    逻辑与
    """
    if formula1.depth == 0 and formula2.depth == 0:
        # Both "propositional". Simply do a obdd and.
        new_aednf = AEDNF(
            terms=[AEDNFTerm(
                objective_part=ObjectiveFormula(
                    obdd_node_id=AND(get_node_from_id(formula1.aednf.terms[0].objective_part.obdd_node_id), get_node_from_id(formula2.aednf.terms[0].objective_part.obdd_node_id)).id,
                    description=f"({formula1.aednf.terms[0].objective_part.description} ∧ {formula2.aednf.terms[0].objective_part.description})"
                ),
                positive_literals=[],
                negative_literals=[]
            )],
            depth=0
        )
        new_aecnf = AECNF(
            clauses=[AECNFClause(
                objective_part=ObjectiveFormula(
                    obdd_node_id=AND(get_node_from_id(formula1.aecnf.clauses[0].objective_part.obdd_node_id), get_node_from_id(formula2.aecnf.clauses[0].objective_part.obdd_node_id)).id,
                    description=f"({formula1.aecnf.clauses[0].objective_part.description} ∧ {formula2.aecnf.clauses[0].objective_part.description})"
                ),
                positive_literals=[],
                negative_literals=[]
            )],
            depth=0
        )
        return AEDNFAECNFPair(aednf=new_aednf, aecnf=new_aecnf, depth=0)
    
    if formula2.depth == 0 and formula1.depth > 0:
        return land(formula2, formula1)
    
    if formula1.depth == 0 and formula2.depth > 0:
        new_terms = []
        for term in formula2.aednf.terms:
            new_terms.append(AEDNFTerm(
                objective_part= ObjectiveFormula(
                    obdd_node_id=AND(get_node_from_id(formula1.aednf.terms[0].objective_part.obdd_node_id), get_node_from_id(term.objective_part.obdd_node_id)).id,
                    description=f"({formula1.aednf.terms[0].objective_part.description} ∧ {term.objective_part.description})"
                ),
                positive_literals=term.positive_literals,
                negative_literals=term.negative_literals
            ))
        new_aednf = AEDNF(terms=new_terms, depth=formula2.depth)
        
        new_aecnf = AECNF(clauses=formula1.aecnf.clauses + formula2.aecnf.clauses, depth=max(formula1.depth, formula2.depth))
        return AEDNFAECNFPair(aednf=new_aednf, aecnf=new_aecnf, depth=max(formula1.depth, formula2.depth))
    
    if formula1.depth > 0 and formula2.depth > 0:
        new_aecnf = AECNF(clauses=formula1.aecnf.clauses + formula2.aecnf.clauses, depth=max(formula1.depth, formula2.depth))
        
        new_terms = []
        for term1 in formula1.aednf.terms:
            for term2 in formula2.aednf.terms:
                # 对每一对term1和term2,我们需要:
                # 1. 合并它们的objective_part (用AND)
                new_objective = ObjectiveFormula(
                    obdd_node_id=AND(get_node_from_id(term1.objective_part.obdd_node_id), get_node_from_id(term2.objective_part.obdd_node_id)).id,
                    description=f"({term1.objective_part.description} ∧ {term2.objective_part.description})"
                )
                
                positive_literals = []
                agent_to_formulas = {}
                for lit in term1.positive_literals + term2.positive_literals:
                    if lit.agent not in agent_to_formulas:
                        agent_to_formulas[lit.agent] = []
                    agent_to_formulas[lit.agent].append(lit.formula)
                
                # 对每个agent,合并其所有的positive formulas
                for agent, formulas in agent_to_formulas.items():
                    if len(formulas) == 1:
                        positive_literals.append(KnowledgeLiteral(
                            agent=agent,
                            formula=formulas[0],
                            negated=False,
                            depth=formulas[0].depth + 1
                        ))
                    else:
                        # merge formulas with land; depth = merged depth + 1
                        merged_formula = formulas[0]
                        for f in formulas[1:]:
                            merged_formula = land(merged_formula, f)
                        positive_literals.append(KnowledgeLiteral(
                            agent=agent,
                            formula=merged_formula,
                            negated=False,
                            depth=merged_formula.depth + 1
                        ))
                # 对于negative literals 只需要简单地堆叠
                negative_literals = term1.negative_literals + term2.negative_literals
                
                new_term = AEDNFTerm(
                    objective_part=new_objective,
                    positive_literals=positive_literals,
                    negative_literals=negative_literals
                )
                new_terms.append(new_term)
            
        new_aednf = AEDNF(terms=new_terms, depth=max(formula1.depth, formula2.depth))

        return AEDNFAECNFPair(aednf=new_aednf, aecnf=new_aecnf, depth=max(formula1.depth, formula2.depth))
    
    raise Exception("Invalid input")

def lor(formula1: AEDNFAECNFPair, formula2: AEDNFAECNFPair) -> AEDNFAECNFPair:
    """
    逻辑或
    """

    if formula1.depth == 0 and formula2.depth == 0:
        # Both "propositional". Simply do a obdd or.
        new_aednf = AEDNF(
            terms=[AEDNFTerm(
                objective_part=ObjectiveFormula(
                    obdd_node_id=OR(get_node_from_id(formula1.aednf.terms[0].objective_part.obdd_node_id), get_node_from_id(formula2.aednf.terms[0].objective_part.obdd_node_id)).id,
                    description=f"({formula1.aednf.terms[0].objective_part.description} ∨ {formula2.aednf.terms[0].objective_part.description})"
                ),
                positive_literals=[],
                negative_literals=[]
            )],
            depth=0
        )
        new_aecnf = AECNF(
            clauses=[AECNFClause(
                objective_part=ObjectiveFormula(
                    obdd_node_id=OR(get_node_from_id(formula1.aecnf.clauses[0].objective_part.obdd_node_id), get_node_from_id(formula2.aecnf.clauses[0].objective_part.obdd_node_id)).id,
                    description=f"({formula1.aecnf.clauses[0].objective_part.description} ∨ {formula2.aecnf.clauses[0].objective_part.description})"
                ),
                positive_literals=[],
                negative_literals=[]
            )],
            depth=0
        )
        return AEDNFAECNFPair(aednf=new_aednf, aecnf=new_aecnf, depth=0)
    
    if formula1.depth > 0 and formula2.depth == 0:
        return lor(formula2, formula1)
    
    if formula1.depth == 0 and formula2.depth > 0:
        new_aednf = AEDNF(terms=formula1.aednf.terms + formula2.aednf.terms, depth=formula2.depth)
        new_clauses = []
        
        for clause in formula2.aecnf.clauses:
            new_clauses.append(AECNFClause(
                objective_part=ObjectiveFormula(
                    obdd_node_id=OR(get_node_from_id(formula1.aednf.terms[0].objective_part.obdd_node_id), get_node_from_id(clause.objective_part.obdd_node_id)).id,
                    description=f"({formula1.aednf.terms[0].objective_part.description} ∨ {clause.objective_part.description})"
                ),
                positive_literals=clause.positive_literals,
                negative_literals=clause.negative_literals
            ))
        new_aecnf = AECNF(clauses=new_clauses, depth=formula2.depth)
        return AEDNFAECNFPair(aednf=new_aednf, aecnf=new_aecnf, depth=formula2.depth)
    
    if formula1.depth > 0 and formula2.depth > 0:
        new_aednf = AEDNF(terms=formula1.aednf.terms + formula2.aednf.terms, depth=max(formula1.depth, formula2.depth))
        
        new_clauses = []
        for clause1 in formula1.aecnf.clauses:
            for clause2 in formula2.aecnf.clauses:
                # 对每一对clause1和clause2,我们需要:
                # 1. 合并它们的objective_part (用OR)
                new_objective = ObjectiveFormula(
                    obdd_node_id=OR(get_node_from_id(clause1.objective_part.obdd_node_id), get_node_from_id(clause2.objective_part.obdd_node_id)).id,
                    description=f"({clause1.objective_part.description} ∨ {clause2.objective_part.description})"
                )
                
                # 合并负知识文字（来自两个子句的 negative_literals）
                agent_to_formulas = {}
                for lit in clause1.negative_literals + clause2.negative_literals:
                    if lit.agent not in agent_to_formulas:
                        agent_to_formulas[lit.agent] = []
                    agent_to_formulas[lit.agent].append(lit.formula)
                
                # 对每个 agent，合并其所有的负知识里的公式（用 OR），并设置正确的深度与否定标记
                merged_negative_literals = []
                for agent, formulas in agent_to_formulas.items():
                    if len(formulas) == 1:
                        merged_negative_literals.append(KnowledgeLiteral(
                            agent=agent,
                            formula=formulas[0],
                            negated=True,
                            depth=formulas[0].depth + 1
                        ))
                    else:
                        merged_formula = formulas[0]
                        for f in formulas[1:]:
                            merged_formula = lor(merged_formula, f)
                        merged_negative_literals.append(KnowledgeLiteral(
                            agent=agent,
                            formula=merged_formula,
                            negated=True,
                            depth=merged_formula.depth + 1
                        ))

                # 正知识直接拼接
                positive_literals = clause1.positive_literals + clause2.positive_literals
                negative_literals = merged_negative_literals
                
                new_clause = AECNFClause(
                    objective_part=new_objective,
                    positive_literals=positive_literals,
                    negative_literals=negative_literals
                )
                new_clauses.append(new_clause)
            
        new_aecnf = AECNF(clauses=new_clauses, depth=max(formula1.depth, formula2.depth))
        return AEDNFAECNFPair(aednf=new_aednf, aecnf=new_aecnf, depth=max(formula1.depth, formula2.depth))
    
def know(formula: AEDNFAECNFPair, agent: str) -> AEDNFAECNFPair:
    """
    逻辑知道
    K_agent(formula) 表示 agent 知道 formula
    """
    new_literal = KnowledgeLiteral(
        agent=agent,
        formula=formula,
        negated=False,
        depth=formula.depth + 1
    )
    
    # 对于AEDNF，K_agent(formula) 表示为 (⊤ ∧ K_agent(formula))
    new_term = AEDNFTerm(
        objective_part=ObjectiveFormula(obdd_node_id=true_node.id, description="⊤"),
        positive_literals=[new_literal],
        negative_literals=[]
    )
    
    # 对于AECNF，K_agent(formula) 表示为 (⊥ ∨ K_agent(formula))
    new_clause = AECNFClause(
        objective_part=ObjectiveFormula(obdd_node_id=false_node.id, description="⊥"),
        positive_literals=[new_literal],
        negative_literals=[]
    )
    
    new_aednf = AEDNF(terms=[new_term], depth=formula.depth + 1)
    new_aecnf = AECNF(clauses=[new_clause], depth=formula.depth + 1)
    return AEDNFAECNFPair(aednf=new_aednf, aecnf=new_aecnf, depth=formula.depth + 1)


        