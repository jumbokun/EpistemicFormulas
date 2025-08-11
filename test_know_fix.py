from archiv.models import AEDNFAECNFPair, AEDNF, AECNF, AEDNFTerm, AECNFClause, KnowledgeLiteral, ObjectiveFormula
from archiv.logical_operations import know
from archiv.obdd import V, true_node, false_node, reset_cache

def test_know_function():
    """测试know函数"""
    reset_cache()
    
    # 创建一个简单的原子命题 v1
    var_node = V("v1")
    aednf_term = AEDNFTerm(
        objective_part=ObjectiveFormula(obdd_node_id=var_node.id, description="v1"),
        positive_literals=[],
        negative_literals=[]
    )
    aecnf_clause = AECNFClause(
        objective_part=ObjectiveFormula(obdd_node_id=var_node.id, description="v1"),
        positive_literals=[],
        negative_literals=[]
    )
    
    formula = AEDNFAECNFPair(
        aednf=AEDNF(terms=[aednf_term], depth=0),
        aecnf=AECNF(clauses=[aecnf_clause], depth=0),
        depth=0
    )
    
    print("原始公式:")
    print(f"  AEDNF: {formula.aednf.terms[0].objective_part.description}")
    print(f"  AECNF: {formula.aecnf.clauses[0].objective_part.description}")
    print()
    
    # 应用知识算子
    result = know(formula, "a1")
    
    print("应用 K_a1 后:")
    print(f"  AEDNF: {result.aednf.terms[0].objective_part.description} ∧ K_a1(φ_1)")
    print(f"  AECNF: {result.aecnf.clauses[0].objective_part.description} ∨ K_a1(φ_1)")
    print(f"  深度: {result.depth}")
    print()
    
    # 检查知识文字
    if result.aednf.terms[0].positive_literals:
        lit = result.aednf.terms[0].positive_literals[0]
        print(f"知识文字: K_{lit.agent}(φ_{lit.depth})")
        print(f"  代理: {lit.agent}")
        print(f"  深度: {lit.depth}")
        print(f"  否定: {lit.negated}")

if __name__ == "__main__":
    test_know_function()
