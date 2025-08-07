from models import AEDNFAECNFPair, AEDNF, AECNF, AEDNFTerm, AECNFClause, KnowledgeLiteral, ObjectiveFormula
from obdd import negate, AND, NOT, OR, IMPLIES, EQUIV, true_node, false_node, conjoin, disjoin, implies, equiv, reset_cache, Node
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
                negated=True
            )
            negative_literals.append(negated_lit)
        
        for lit in clause.negative_literals:
            negated_lit = KnowledgeLiteral(
                agent=lit.agent,
                formula=lit.formula,
                negated=False
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
                negated=True
            )
            negative_literals.append(negated_lit)
        
        for lit in term.negative_literals:
            negated_lit = KnowledgeLiteral(
                agent=lit.agent,
                formula=lit.formula,
                negated=False
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
    if formula1.depth and formula2.depth == 0:
        # Both "propositional". Simply do a obdd and.
        new_aednf = AEDNF(
            terms=[AEDNFTerm(
                objective_part=ObjectiveFormula(
                    obdd_node_id=AND(formula1.aednf.terms[0].objective_part.obdd_node_id, formula2.aednf.terms[0].objective_part.obdd_node_id).id,
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
                    obdd_node_id=AND(formula1.aecnf.clauses[0].objective_part.obdd_node_id, formula2.aecnf.clauses[0].objective_part.obdd_node_id).id,
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
                objective_part= AND(formula1.aednf.terms[0].objective_part.obdd_node_id, term.objective_part.obdd_node_id),
                positive_literals=term.positive_literals,
                negative_literals=term.negative_literals
            ))
        new_aednf = AEDNF(terms=new_terms, depth=formula2.depth)
        
        new_aecnf = AECNF(formula1.aecnf.clauses + formula2.aecnf.clauses, depth=max(formula1.depth, formula2.depth))
        return AEDNFAECNFPair(aednf=new_aednf, aecnf=new_aecnf, depth=max(formula1.depth, formula2.depth))
    
    if formula1.depth > 0 and formula2.depth > 0:
        new_aecnf = AECNF(clauses=formula1.aecnf.clauses + formula2.aecnf.clauses, depth=max(formula1.depth, formula2.depth))
        
        new_terms = []
        for term1 in formula1.aednf.terms:
            for term2 in formula2.aednf.terms:
                # 对每一对term1和term2,我们需要:
                # 1. 合并它们的objective_part (用AND)
                new_objective = ObjectiveFormula(
                    obdd_node_id=AND(term1.objective_part.obdd_node_id, term2.objective_part.obdd_node_id).id,
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
                            negated=False
                        ))
                    else:
                        # merge formulas with land
                        merged_formula = formulas[0]
                        for f in formulas[1:]:
                            merged_formula = land(merged_formula, f)
                        positive_literals.append(KnowledgeLiteral(
                            agent=agent,
                            formula=merged_formula,
                            negated=False   
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
    
    return Exception("Invalid input")

def lor(formula1: AEDNFAECNFPair, formula2: AEDNFAECNFPair) -> AEDNFAECNFPair:
    """
    逻辑或
    """

    if formula1.depth == 0 and formula2.depth == 0:
        # Both "propositional". Simply do a obdd or.
        new_aednf = AEDNF(
            terms=[AEDNFTerm(
                objective_part=ObjectiveFormula(
                    obdd_node_id=OR(formula1.aednf.terms[0].objective_part.obdd_node_id, formula2.aednf.terms[0].objective_part.obdd_node_id).id,
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
                    obdd_node_id=AND(formula1.aecnf.clauses[0].objective_part.obdd_node_id, formula2.aecnf.clauses[0].objective_part.obdd_node_id).id,
                    description=f"({formula1.aecnf.clauses[0].objective_part.description} ∧ {formula2.aecnf.clauses[0].objective_part.description})"
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
                objective_part= OR(formula1.aednf.terms[0].objective_part.obdd_node_id, clause.objective_part.obdd_node_id),
                positive_literals=clause.positive_literals,
                negative_literals=clause.negative_literals
            ))
        new_aecnf = AEDNF(terms=new_clauses, depth=formula2.depth)
        return AEDNFAECNFPair(aednf=new_aednf, aecnf=new_aecnf, depth=formula2.depth)
    
    if formula1.depth > 0 and formula2.depth > 0:
        new_aednf = AEDNF(terms=formula1.aednf.terms + formula2.aednf.terms, depth=max(formula1.depth, formula2.depth))
        
        new_clauses = []
        for clause1 in formula1.aecnf.clauses:
            for clause2 in formula2.aecnf.clauses:
                # 对每一对term1和term2,我们需要:
                # 1. 合并它们的objective_part (用AND)
                new_objective = ObjectiveFormula(
                    obdd_node_id=OR(clause1.objective_part.obdd_node_id, clause2.objective_part.obdd_node_id).id,
                    description=f"({clause1.objective_part.description} ∨ {clause2.objective_part.description})"
                )
                
                negative_literals = []
                agent_to_formulas = {}
                for lit in clause1.positive_literals + clause2.positive_literals:
                    if lit.agent not in agent_to_formulas:
                        agent_to_formulas[lit.agent] = []
                    agent_to_formulas[lit.agent].append(lit.formula)
                
                # 对每个agent,合并其所有的negative formulas
                for agent, formulas in agent_to_formulas.items():
                    if len(formulas) == 1:
                        negative_literals.append(KnowledgeLiteral(
                            agent=agent,
                            formula=formulas[0],
                            negated=False
                        ))
                    else:
                        # merge formulas with land
                        merged_formula = formulas[0]
                        for f in formulas[1:]:
                            merged_formula = lor(merged_formula, f)
                        negative_literals.append(KnowledgeLiteral(
                            agent=agent,
                            formula=merged_formula,
                            negated=False   
                        ))
                        
                positive_literals = clause1.positive_literals + clause2.positive_literals
                
                new_clause = AECNFClause(
                    objective_part=new_objective,
                    positive_literals=positive_literals,
                    negative_literals=negative_literals
                )
                new_clauses.append(new_clause)
            
        new_aecnf = AECNF(clauses=new_clauses, depth=max(formula1.depth, formula2.depth))

def know(formula: AEDNFAECNFPair, agent: str) -> AEDNFAECNFPair:
    """
    逻辑知道
    """
    if formula.depth == 0:
        new_literal = KnowledgeLiteral(
            agent=agent,
            formula=formula.aednf.terms[0].objective_part,
            negated=False
        )
        new_clause  = AECNFClause(
            objective_part=ObjectiveFormula(),
            positive_literals=[new_literal],
            negative_literals=[]
        )
        new_term = AEDNFTerm(
            objective_part=ObjectiveFormula(),
            positive_literals=[new_literal],
            negative_literals=[]
        )
        new_aednf = AEDNF(terms=[new_term], depth=1)
        new_aecnf = AECNF(clauses=[new_clause], depth=1)
        return AEDNFAECNFPair(aednf=new_aednf, aecnf=new_aecnf, depth=1)
    
        