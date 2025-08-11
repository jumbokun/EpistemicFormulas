from archiv.models import AEDNFAECNFPair, AEDNF, AECNF, AEDNFTerm, AECNFClause, KnowledgeLiteral, ObjectiveFormula, create_objective_pair
from archiv.logical_operations import land, lor, lnot, know
from archiv.obdd import V, true_node, false_node, reset_cache
import random
from typing import List, Dict, Any
import json

class ParameterizedFormulaGenerator:
    def __init__(self, 
                 agent_count: int = 3, 
                 variable_count: int = 5, 
                 max_depth: int = 3,
                 target_length: int = 15):
        """
        初始化参数化公式生成器
        
        Args:
            agent_count: 代理数量 (a1, a2, a3...)
            variable_count: 原子命题数量 (v1, v2, v3...)
            max_depth: 最大深度限制
            target_length: 目标公式长度
        """
        self.agent_count = agent_count
        self.variable_count = variable_count
        self.max_depth = max_depth
        self.target_length = target_length
        
        # 生成代理名称和变量名称
        self.agents = [f"a{i}" for i in range(1, agent_count + 1)]
        self.variables = [f"v{i}" for i in range(1, variable_count + 1)]
        
        # 存储生成的公式
        self.generated_formulas = []
        self.formula_count = 0
        
        self.reset_cache()
    
    def reset_cache(self):
        """重置OBDD缓存"""
        reset_cache()
    
    def create_atomic_formula(self, variable: str) -> AEDNFAECNFPair:
        """创建原子命题公式"""
        # 创建变量节点
        var_node = V(variable)
        print(f"创建变量 {variable}: 节点ID = {var_node.id}")
        
        # 创建AEDNF项
        aednf_term = AEDNFTerm(
            objective_part=ObjectiveFormula(obdd_node_id=var_node.id, description=variable),
            positive_literals=[],
            negative_literals=[]
        )
        
        # 创建AECNF子句
        aecnf_clause = AECNFClause(
            objective_part=ObjectiveFormula(obdd_node_id=var_node.id, description=variable),
            positive_literals=[],
            negative_literals=[]
        )
        
        # 创建AEDNF和AECNF
        aednf = AEDNF(terms=[aednf_term], depth=0)
        aecnf = AECNF(clauses=[aecnf_clause], depth=0)
        
        return AEDNFAECNFPair(aednf=aednf, aecnf=aecnf, depth=0)
    
    def generate_base_formulas(self) -> List[AEDNFAECNFPair]:
        """生成基础公式（深度0）"""
        formulas = []
        for var in self.variables:
            formula = self.create_atomic_formula(var)
            formulas.append(formula)
        return formulas
    
    def generate_depth_formulas(self, base_formulas: List[AEDNFAECNFPair], depth: int) -> List[AEDNFAECNFPair]:
        """生成指定深度的公式"""
        if depth == 0:
            return base_formulas
        
        formulas = []
        
        # 从基础公式开始，逐步构建深度公式
        current_formulas = base_formulas
        
        for current_depth in range(1, depth + 1):
            new_formulas = []
            
            # 对每个代理，创建知识公式
            for agent in self.agents:
                for formula in current_formulas:
                    if formula.depth < self.max_depth:
                        # 创建 K_agent(formula)
                        knowledge_formula = know(formula, agent)
                        new_formulas.append(knowledge_formula)
            
            # 创建否定公式
            for formula in current_formulas:
                if formula.depth < self.max_depth:
                    negated_formula = lnot(formula)
                    new_formulas.append(negated_formula)
            
            # 创建合取和析取公式
            for i in range(len(current_formulas)):
                for j in range(i + 1, len(current_formulas)):
                    if current_formulas[i].depth < self.max_depth and current_formulas[j].depth < self.max_depth:
                        # 合取
                        conj_formula = land(current_formulas[i], current_formulas[j])
                        new_formulas.append(conj_formula)
                        
                        # 析取
                        disj_formula = lor(current_formulas[i], current_formulas[j])
                        new_formulas.append(disj_formula)
            
            current_formulas = new_formulas
            formulas.extend(new_formulas)
        
        return formulas
    
    def calculate_formula_length(self, formula: AEDNFAECNFPair) -> int:
        """计算公式的复杂度长度"""
        # 计算AECNF部分的长度
        aecnf_length = 0
        for clause in formula.aecnf.clauses:
            # 客观部分
            if clause.objective_part.description not in ["⊤", "⊥"]:
                aecnf_length += 1
            
            # 认知文字
            aecnf_length += len(clause.positive_literals)
            aecnf_length += len(clause.negative_literals)
        
        # 计算AEDNF部分的长度
        aednf_length = 0
        for term in formula.aednf.terms:
            # 客观部分
            if term.objective_part.description not in ["⊤", "⊥"]:
                aednf_length += 1
            
            # 认知文字
            aednf_length += len(term.positive_literals)
            aednf_length += len(term.negative_literals)
        
        return max(aecnf_length, aednf_length)
    
    def generate_formulas_until_target_length(self) -> Dict[str, Any]:
        """从底向上生成公式，直到达到目标长度"""
        print(f"开始生成公式...")
        print(f"参数设置:")
        print(f"  代理数量: {self.agent_count} ({', '.join(self.agents)})")
        print(f"  变量数量: {self.variable_count} ({', '.join(self.variables)})")
        print(f"  最大深度: {self.max_depth}")
        print(f"  目标长度: {self.target_length}")
        print()
        
        # 生成基础公式
        base_formulas = self.generate_base_formulas()
        print(f"生成基础公式 (深度0): {len(base_formulas)} 个")
        
        # 调试信息
        from archiv.obdd import branch_cache, nodeID_2_key
        print(f"branch_cache 键: {list(branch_cache.keys())}")
        print(f"nodeID_2_key 键: {list(nodeID_2_key.keys())}")
        
        all_formulas = []
        formulas_by_depth = {}
        
        # 按深度生成公式
        for depth in range(self.max_depth + 1):
            depth_formulas = self.generate_depth_formulas(base_formulas, depth)
            
            # 过滤出符合目标长度的公式
            target_formulas = []
            for formula in depth_formulas:
                length = self.calculate_formula_length(formula)
                if length <= self.target_length:
                    target_formulas.append(formula)
            
            formulas_by_depth[f"depth_{depth}"] = target_formulas
            all_formulas.extend(target_formulas)
            
            print(f"深度 {depth}: 生成 {len(depth_formulas)} 个公式，符合长度要求的有 {len(target_formulas)} 个")
        
        # 转换为可序列化的格式
        serializable_formulas = {}
        for depth_key, formulas in formulas_by_depth.items():
            serializable_formulas[depth_key] = []
            for formula in formulas:
                formula_dict = self.formula_to_dict(formula)
                serializable_formulas[depth_key].append(formula_dict)
        
        # 保存到文件
        with open('parameterized_formulas.json', 'w', encoding='utf-8') as f:
            json.dump(serializable_formulas, f, ensure_ascii=False, indent=2)
        
        print(f"\n总共生成 {len(all_formulas)} 个符合长度要求的公式")
        print(f"结果已保存到 'parameterized_formulas.json'")
        
        return serializable_formulas
    
    def formula_to_dict(self, formula: AEDNFAECNFPair) -> Dict[str, Any]:
        """将公式转换为字典格式"""
        return {
            "aednf": {
                "terms": [
                    {
                        "objective_part": {
                            "obdd_node_id": term.objective_part.obdd_node_id,
                            "description": term.objective_part.description
                        },
                        "positive_literals": [
                            {
                                "agent": lit.agent,
                                "formula_depth": lit.formula.depth,
                                "negated": lit.negated
                            }
                            for lit in term.positive_literals
                        ],
                        "negative_literals": [
                            {
                                "agent": lit.agent,
                                "formula_depth": lit.formula.depth,
                                "negated": lit.negated
                            }
                            for lit in term.negative_literals
                        ]
                    }
                    for term in formula.aednf.terms
                ],
                "depth": formula.aednf.depth
            },
            "aecnf": {
                "clauses": [
                    {
                        "objective_part": {
                            "obdd_node_id": clause.objective_part.obdd_node_id,
                            "description": clause.objective_part.description
                        },
                        "positive_literals": [
                            {
                                "agent": lit.agent,
                                "formula_depth": lit.formula.depth,
                                "negated": lit.negated
                            }
                            for lit in clause.positive_literals
                        ],
                        "negative_literals": [
                            {
                                "agent": lit.agent,
                                "formula_depth": lit.formula.depth,
                                "negated": lit.negated
                            }
                            for lit in clause.negative_literals
                        ]
                    }
                    for clause in formula.aecnf.clauses
                ],
                "depth": formula.aecnf.depth
            },
            "depth": formula.depth
        }

def main():
    """主函数 - 演示参数化公式生成"""
    # 设置参数
    agent_count = 3      # 3个代理: a1, a2, a3
    variable_count = 4   # 4个变量: v1, v2, v3, v4
    max_depth = 2        # 最大深度2
    target_length = 8    # 目标长度8
    
    # 创建生成器
    generator = ParameterizedFormulaGenerator(
        agent_count=agent_count,
        variable_count=variable_count,
        max_depth=max_depth,
        target_length=target_length
    )
    
    # 生成公式
    formulas = generator.generate_formulas_until_target_length()
    
    print("\n生成完成！")

if __name__ == "__main__":
    main()

