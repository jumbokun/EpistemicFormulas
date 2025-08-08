from archiv.models import AEDNFAECNFPair, AEDNF, AECNF, AEDNFTerm, AECNFClause, KnowledgeLiteral, ObjectiveFormula, create_objective_pair
from archiv.logical_operations import land, lor, lnot, know
from archiv.obdd import V, true_node, false_node, reset_cache
import random
from typing import List, Dict, Any
import json

class EpistemicFormulaGenerator:
    """认知逻辑公式生成器"""
    
    def __init__(self):
        self.agents = ["Alice", "Bob", "Charlie"]
        self.variables = ["p", "q", "r", "s", "t"]
        self.reset_cache()
    
    def reset_cache(self):
        """重置OBDD缓存"""
        reset_cache()
    
    def generate_depth_0_formulas(self, count: int = 5) -> List[AEDNFAECNFPair]:
        """生成深度为0的公式（纯命题逻辑）"""
        formulas = []
        
        # 基本变量
        for var in self.variables[:3]:
            formulas.append(create_objective_pair(var))
        
        # 真值和假值
        formulas.append(create_objective_pair("⊤"))
        formulas.append(create_objective_pair("⊥"))
        
        # 简单组合
        if count > 5:
            # 生成一些简单的组合
            p = create_objective_pair("p")
            q = create_objective_pair("q")
            r = create_objective_pair("r")
            
            formulas.append(land(p, q))
            formulas.append(lor(p, q))
            formulas.append(lnot(p))
            formulas.append(land(lor(p, q), r))
        
        return formulas[:count]
    
    def generate_depth_1_formulas(self, count: int = 5) -> List[AEDNFAECNFPair]:
        """生成深度为1的公式（包含一层认知算子）"""
        formulas = []
        
        # 从深度0的公式开始
        depth_0_formulas = self.generate_depth_0_formulas(3)
        
        for agent in self.agents:
            for base_formula in depth_0_formulas:
                # K_a(φ)
                formulas.append(know(base_formula, agent))
                
                if len(formulas) >= count:
                    break
            if len(formulas) >= count:
                break
        
        # 生成一些组合
        if len(formulas) < count:
            p = create_objective_pair("p")
            q = create_objective_pair("q")
            
            # K_Alice(p) ∧ K_Bob(q)
            k_alice_p = know(p, "Alice")
            k_bob_q = know(q, "Bob")
            formulas.append(land(k_alice_p, k_bob_q))
            
            # K_Alice(p) ∨ K_Bob(q)
            formulas.append(lor(k_alice_p, k_bob_q))
            
            # ¬K_Alice(p)
            formulas.append(lnot(k_alice_p))
        
        return formulas[:count]
    
    def generate_depth_2_formulas(self, count: int = 5) -> List[AEDNFAECNFPair]:
        """生成深度为2的公式（包含两层认知算子）"""
        formulas = []
        
        # 从深度1的公式开始
        depth_1_formulas = self.generate_depth_1_formulas(3)
        
        for agent in self.agents:
            for base_formula in depth_1_formulas:
                # K_a(K_b(φ))
                formulas.append(know(base_formula, agent))
                
                if len(formulas) >= count:
                    break
            if len(formulas) >= count:
                break
        
        # 生成一些复杂的组合
        if len(formulas) < count:
            p = create_objective_pair("p")
            q = create_objective_pair("q")
            
            # K_Alice(K_Bob(p))
            k_bob_p = know(p, "Bob")
            k_alice_k_bob_p = know(k_bob_p, "Alice")
            formulas.append(k_alice_k_bob_p)
            
            # K_Alice(p) ∧ K_Bob(K_Charlie(q))
            k_alice_p = know(p, "Alice")
            k_charlie_q = know(q, "Charlie")
            k_bob_k_charlie_q = know(k_charlie_q, "Bob")
            formulas.append(land(k_alice_p, k_bob_k_charlie_q))
            
            # K_Alice(p ∨ K_Bob(q))
            k_bob_q = know(q, "Bob")
            p_or_k_bob_q = lor(p, k_bob_q)
            formulas.append(know(p_or_k_bob_q, "Alice"))
        
        return formulas[:count]
    
    def generate_all_formulas(self, count_per_depth: int = 3) -> Dict[str, List[AEDNFAECNFPair]]:
        """生成所有深度的公式"""
        return {
            "depth_0": self.generate_depth_0_formulas(count_per_depth),
            "depth_1": self.generate_depth_1_formulas(count_per_depth),
            "depth_2": self.generate_depth_2_formulas(count_per_depth)
        }
    
    def formula_to_dict(self, formula: AEDNFAECNFPair) -> Dict[str, Any]:
        """将公式转换为字典格式"""
        def term_to_dict(term):
            return {
                "objective_part": {
                    "obdd_node_id": term.objective_part.obdd_node_id,
                    "description": term.objective_part.description
                },
                "positive_literals": [
                    {
                        "agent": lit.agent,
                        "negated": lit.negated,
                        "depth": lit.depth,
                        "formula_depth": lit.formula.depth
                    } for lit in term.positive_literals
                ],
                "negative_literals": [
                    {
                        "agent": lit.agent,
                        "negated": lit.negated,
                        "depth": lit.depth,
                        "formula_depth": lit.formula.depth
                    } for lit in term.negative_literals
                ]
            }
        
        def clause_to_dict(clause):
            return {
                "objective_part": {
                    "obdd_node_id": clause.objective_part.obdd_node_id,
                    "description": clause.objective_part.description
                },
                "positive_literals": [
                    {
                        "agent": lit.agent,
                        "negated": lit.negated,
                        "depth": lit.depth,
                        "formula_depth": lit.formula.depth
                    } for lit in clause.positive_literals
                ],
                "negative_literals": [
                    {
                        "agent": lit.agent,
                        "negated": lit.negated,
                        "depth": lit.depth,
                        "formula_depth": lit.formula.depth
                    } for lit in clause.negative_literals
                ]
            }
        
        return {
            "depth": formula.depth,
            "aednf": {
                "terms": [term_to_dict(term) for term in formula.aednf.terms],
                "depth": formula.aednf.depth
            },
            "aecnf": {
                "clauses": [clause_to_dict(clause) for clause in formula.aecnf.clauses],
                "depth": formula.aecnf.depth
            }
        }
    
    def display_formula(self, formula: AEDNFAECNFPair, index: int = None) -> str:
        """以可读格式显示公式"""
        prefix = f"Formula {index}: " if index is not None else ""
        
        # 尝试生成描述性文本
        description = self._generate_description(formula)
        
        return f"{prefix}Depth {formula.depth} - {description}"
    
    def _generate_description(self, formula: AEDNFAECNFPair) -> str:
        """生成公式的描述性文本"""
        if formula.depth == 0:
            # 深度0：纯命题逻辑
            if formula.aednf.terms[0].objective_part.description:
                return f"Propositional: {formula.aednf.terms[0].objective_part.description}"
            else:
                return "Propositional formula"
        
        elif formula.depth == 1:
            # 深度1：一层认知算子
            if formula.aednf.terms[0].positive_literals:
                lit = formula.aednf.terms[0].positive_literals[0]
                return f"K_{lit.agent}(φ) where φ is depth {lit.formula.depth}"
            elif formula.aednf.terms[0].negative_literals:
                lit = formula.aednf.terms[0].negative_literals[0]
                return f"¬K_{lit.agent}(φ) where φ is depth {lit.formula.depth}"
            else:
                return "Depth 1 epistemic formula"
        
        elif formula.depth == 2:
            # 深度2：两层认知算子
            if formula.aednf.terms[0].positive_literals:
                lit = formula.aednf.terms[0].positive_literals[0]
                return f"K_{lit.agent}(φ) where φ is depth {lit.formula.depth}"
            elif formula.aednf.terms[0].negative_literals:
                lit = formula.aednf.terms[0].negative_literals[0]
                return f"¬K_{lit.agent}(φ) where φ is depth {lit.formula.depth}"
            else:
                return "Depth 2 epistemic formula"
        
        return f"Epistemic formula of depth {formula.depth}"

def main():
    """主函数：生成并展示公式"""
    generator = EpistemicFormulaGenerator()
    
    print("=== 认知逻辑公式生成器 ===\n")
    
    # 生成所有深度的公式
    all_formulas = generator.generate_all_formulas(count_per_depth=3)
    
    # 展示结果
    for depth, formulas in all_formulas.items():
        print(f"\n--- {depth.upper()} (深度 {depth.split('_')[1]}) ---")
        print(f"生成了 {len(formulas)} 个公式：\n")
        
        for i, formula in enumerate(formulas, 1):
            print(f"{generator.display_formula(formula, i)}")
            
            # 显示详细信息
            formula_dict = generator.formula_to_dict(formula)
            print(f"  AEDNF项数: {len(formula_dict['aednf']['terms'])}")
            print(f"  AECNF子句数: {len(formula_dict['aecnf']['clauses'])}")
            
            # 显示认知文字
            if formula_dict['aednf']['terms'][0]['positive_literals']:
                print(f"  正认知文字: {len(formula_dict['aednf']['terms'][0]['positive_literals'])} 个")
            if formula_dict['aednf']['terms'][0]['negative_literals']:
                print(f"  负认知文字: {len(formula_dict['aednf']['terms'][0]['negative_literals'])} 个")
            
            print()
    
    # 保存到JSON文件
    output_data = {}
    for depth, formulas in all_formulas.items():
        output_data[depth] = [generator.formula_to_dict(f) for f in formulas]
    
    with open('generated_formulas.json', 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n=== 生成完成 ===")
    print(f"总共生成了 {sum(len(formulas) for formulas in all_formulas.values())} 个公式")
    print(f"结果已保存到 'generated_formulas.json'")

if __name__ == "__main__":
    main() 