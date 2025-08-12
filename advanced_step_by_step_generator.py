from archiv.models import AEDNFAECNFPair, AEDNF, AECNF, AEDNFTerm, AECNFClause, KnowledgeLiteral, ObjectiveFormula, create_objective_pair
from archiv.logical_operations import land, lor, lnot, know
from archiv.obdd import V, true_node, false_node, reset_cache, negate, AND, OR
import random
from typing import List, Dict, Any, Tuple
import json

class AdvancedStepByStepFormulaGenerator:
    def __init__(self, 
                  agent_count: int = 3, 
                  variable_count: int = 5, 
                  max_depth: int = 3,
                  target_complexity: int = 15,
                  show_detailed_fol: bool = True,
                  weight_negation: float = 0.12,
                  weight_conjunction: float = 0.44,
                  weight_disjunction: float = 0.34,
                  weight_knowledge: float = 0.10):
        """
        初始化高级逐步公式生成器
        
        Args:
            agent_count: 代理数量 (a1, a2, a3...)
            variable_count: 原子命题数量 (v1, v2, v3...)
            max_depth: 最大深度限制
            target_complexity: 目标复杂度
            show_detailed_fol: 是否显示详细的FOL格式
        """
        self.agent_count = agent_count
        self.variable_count = variable_count
        self.max_depth = max_depth
        self.target_complexity = target_complexity
        self.show_detailed_fol = show_detailed_fol
         # 操作权重（用于降低否定频率）
        self.weight_negation = weight_negation
        self.weight_conjunction = weight_conjunction
        self.weight_disjunction = weight_disjunction
        self.weight_knowledge = weight_knowledge
        
        # 生成代理和变量名称
        self.agents = [f"a{i+1}" for i in range(agent_count)]
        self.variables = [f"v{i+1}" for i in range(variable_count)]
        
        # 重置OBDD缓存
        reset_cache()
        
        # 存储生成步骤
        self.generation_steps = []
        self.step_counter = 0
        self.formula_counter = 0  # 用于生成φ_1, φ_2等占位符
 
    def _choose_operator(self, complexity: int, deg_nesting: int) -> int:
         """
         选择操作：0=否定, 1=合取, 2=析取, 3=知识。
         使用带权抽样以降低否定概率；当 deg_nesting==0 时禁用知识算子；
         当 complexity<3 时避免选择需要二元拆分的操作（保持与原逻辑一致）。
         """
         # 小复杂度时，遵循原约束
         if complexity < 3:
             if deg_nesting == 0:
                 return 0  # 只能选择否定
             # 在可用时，倾向选择知识而不是否定
             options = [0, 3]
             weights = [self.weight_negation, self.weight_knowledge if self.weight_knowledge > 0 else 0.5]
             total = sum(weights)
             if total <= 0:
                 return 3
             r = random.random() * total
             acc = 0.0
             for opt, w in zip(options, weights):
                 acc += w
                 if r <= acc:
                     return opt
             return options[-1]
 
         # complexity >= 3: 可用二元操作
         options = [0, 1, 2]
         weights = [self.weight_negation, self.weight_conjunction, self.weight_disjunction]
         if deg_nesting > 0:
             options.append(3)
             weights.append(self.weight_knowledge)
         total = sum(weights)
         # 归一化并抽样
         if total <= 0:
             # 回退到均匀随机
             return random.choice(options)
         r = random.random() * total
         acc = 0.0
         for opt, w in zip(options, weights):
             acc += w
             if r <= acc:
                 return opt
         return options[-1]
    
    def get_nested_formula_description(self, formula_depth: int, agent: str) -> str:
        """根据公式深度生成嵌套公式的描述"""
        if formula_depth == 0:
            return random.choice(self.variables)  # 随机选择一个变量
        elif formula_depth == 1:
            return f"K_{agent}({random.choice(self.variables)})"  # 深度1的认知公式
        elif formula_depth == 2:
            return f"K_{agent}(K_{agent}({random.choice(self.variables)}))"  # 深度2的认知公式
        else:
            return f"φ_{formula_depth}"  # 占位符
    
    def formula_to_detailed_fol(self, formula: AEDNFAECNFPair, is_aednf: bool = True) -> str:
        """将公式转换为详细的FOL格式字符串"""
        if is_aednf:
            structure = formula.aednf
            terms_key = "terms"
        else:
            structure = formula.aecnf
            terms_key = "clauses"
        
        if not structure.terms if is_aednf else not structure.clauses:
            return "⊥"
        
        # 对于AEDNF，我们显示第一个项；对于AECNF，我们显示第一个子句
        item = structure.terms[0] if is_aednf else structure.clauses[0]
        objective_part = item.objective_part.description if item.objective_part.description else "⊤"
        
        parts = [objective_part]
        
        # 处理正知识文字
        for lit in item.positive_literals:
            if lit.negated:
                parts.append(f"¬K_{lit.agent}({self.get_nested_formula_description(lit.depth, lit.agent)})")
            else:
                parts.append(f"K_{lit.agent}({self.get_nested_formula_description(lit.depth, lit.agent)})")
        
        # 处理负知识文字
        for lit in item.negative_literals:
            if lit.negated:
                parts.append(f"¬K_{lit.agent}({self.get_nested_formula_description(lit.depth, lit.agent)})")
            else:
                parts.append(f"K_{lit.agent}({self.get_nested_formula_description(lit.depth, lit.agent)})")
        
        return " ∧ ".join(parts) if parts else "⊤"
    
    def add_step(self, step_type: str, description: str, formula: AEDNFAECNFPair = None, 
                 sub_formulas: List[AEDNFAECNFPair] = None, complexity: int = None):
        """添加一个生成步骤"""
        self.step_counter += 1
        step = {
            "step": self.step_counter,
            "type": step_type,
            "description": description,
            "complexity": complexity,
            "formula": formula.model_dump() if formula else None,
            "sub_formulas": [f.model_dump() for f in sub_formulas] if sub_formulas else None
        }
        self.generation_steps.append(step)
        
        print(f"步骤 {self.step_counter}: {step_type}")
        print(f"  描述: {description}")
        if complexity is not None:
            print(f"  复杂度: {complexity}")
        
        if formula:
            if self.show_detailed_fol:
                aednf_str = self.formula_to_detailed_fol(formula, is_aednf=True)
                aecnf_str = self.formula_to_detailed_fol(formula, is_aednf=False)
                print(f"  AEDNF: {aednf_str}")
                print(f"  AECNF: {aecnf_str}")
            else:
                print(f"  生成公式: {self.formula_to_string(formula)}")
        
        if sub_formulas:
            print(f"  子公式:")
            for i, sub_formula in enumerate(sub_formulas):
                if self.show_detailed_fol:
                    aednf_str = self.formula_to_detailed_fol(sub_formula, is_aednf=True)
                    print(f"    {i+1}. AEDNF: {aednf_str}")
                else:
                    print(f"    {i+1}. {self.formula_to_string(sub_formula)}")
        
        print()
    
    def formula_to_string(self, formula: AEDNFAECNFPair) -> str:
        """将公式转换为可读字符串"""
        if not formula.aednf.terms:
            return "⊥"
        
        term = formula.aednf.terms[0]
        parts = []
        
        if term.objective_part.description:
            parts.append(term.objective_part.description)
        
        for lit in term.positive_literals:
            if lit.negated:
                parts.append(f"¬K_{lit.agent}(φ_{lit.depth})")
            else:
                parts.append(f"K_{lit.agent}(φ_{lit.depth})")
        
        for lit in term.negative_literals:
            if lit.negated:
                parts.append(f"¬K_{lit.agent}(φ_{lit.depth})")
            else:
                parts.append(f"K_{lit.agent}(φ_{lit.depth})")
        
        return " ∧ ".join(parts) if parts else "⊤"
    
    def create_atomic_formula(self, variable: str) -> AEDNFAECNFPair:
        """创建原子命题公式"""
        var_node = V(variable)
        
        aednf_term = AEDNFTerm(
            objective_part=ObjectiveFormula(obdd_node_id=var_node.id, description=variable),
            positive_literals=[],
            negative_literals=[]
        )
        
        aecnf_clause = AECNFClause(
            objective_part=ObjectiveFormula(obdd_node_id=var_node.id, description=variable),
            positive_literals=[],
            negative_literals=[]
        )
        
        formula = AEDNFAECNFPair(
            aednf=AEDNF(terms=[aednf_term], depth=0),
            aecnf=AECNF(clauses=[aecnf_clause], depth=0),
            depth=0
        )
        
        self.add_step("原子命题", f"创建原子命题 {variable}", formula, complexity=1)
        return formula
    
    def generate_formula_with_steps(self, complexity: int, deg_nesting: int = 1) -> AEDNFAECNFPair:
        """递归生成公式，展示每一步过程"""
        
        if complexity == 1:
            var_dice = random.randint(0, self.variable_count - 1)
            variable = self.variables[var_dice]
            return self.create_atomic_formula(variable)
        
         # 选择连接词（带权重）
        con_dice = self._choose_operator(complexity, deg_nesting)
        
        if con_dice == 0:
            # 否定
            self.add_step("选择操作", f"选择否定操作 (复杂度: {complexity})", complexity=complexity)
            sub_formula = self.generate_formula_with_steps(complexity - 1, deg_nesting)
            result = lnot(sub_formula)
            self.add_step("否定操作", f"对公式进行否定", result, [sub_formula], complexity=complexity)
            return result
            
        elif con_dice == 1:
            # 合取
            self.add_step("选择操作", f"选择合取操作 (复杂度: {complexity})", complexity=complexity)
            complex_dice = random.randint(1, complexity - 2)
            formula1 = self.generate_formula_with_steps(complex_dice, deg_nesting)
            formula2 = self.generate_formula_with_steps(complexity - complex_dice, deg_nesting)
            result = land(formula1, formula2)
            self.add_step("合取操作", f"对两个公式进行合取", result, [formula1, formula2], complexity=complexity)
            return result
            
        elif con_dice == 2:
            # 析取
            self.add_step("选择操作", f"选择析取操作 (复杂度: {complexity})", complexity=complexity)
            complex_dice = random.randint(1, complexity - 2)
            formula1 = self.generate_formula_with_steps(complex_dice, deg_nesting)
            formula2 = self.generate_formula_with_steps(complexity - complex_dice, deg_nesting)
            result = lor(formula1, formula2)
            self.add_step("析取操作", f"对两个公式进行析取", result, [formula1, formula2], complexity=complexity)
            return result
            
        else:
            # 知识算子
            self.add_step("选择操作", f"选择知识算子操作 (复杂度: {complexity})", complexity=complexity)
            agent = random.choice(self.agents)
            sub_formula = self.generate_formula_with_steps(complexity - 1, deg_nesting - 1)
            result = know(sub_formula, agent)
            self.add_step("知识算子", f"对公式应用知识算子 K_{agent}", result, [sub_formula], complexity=complexity)
            return result
    
    def generate_target_formula(self) -> Dict[str, Any]:
        """生成目标公式并返回完整过程"""
        print(f"开始生成目标公式 (复杂度: {self.target_complexity})")
        print(f"参数设置:")
        print(f"  - 代理数量: {self.agent_count}")
        print(f"  - 变量数量: {self.variable_count}")
        print(f"  - 最大深度: {self.max_depth}")
        print(f"  - 目标复杂度: {self.target_complexity}")
        print(f"  - 详细FOL显示: {self.show_detailed_fol}")
        print("=" * 60)
        
        final_formula = self.generate_formula_with_steps(self.target_complexity, self.max_depth)
        
        result = {
            "parameters": {
                "agent_count": self.agent_count,
                "variable_count": self.variable_count,
                "max_depth": self.max_depth,
                "target_complexity": self.target_complexity,
                "show_detailed_fol": self.show_detailed_fol
            },
            "agents": self.agents,
            "variables": self.variables,
            "generation_steps": self.generation_steps,
            "final_formula": final_formula.model_dump(),
            "total_steps": self.step_counter
        }
        
        print("=" * 60)
        print(f"生成完成！总共 {self.step_counter} 个步骤")
        
        if self.show_detailed_fol:
            aednf_str = self.formula_to_detailed_fol(final_formula, is_aednf=True)
            aecnf_str = self.formula_to_detailed_fol(final_formula, is_aednf=False)
            print(f"最终公式:")
            print(f"  AEDNF: {aednf_str}")
            print(f"  AECNF: {aecnf_str}")
        else:
            print(f"最终公式: {self.formula_to_string(final_formula)}")
        
        return result
    
    def save_to_file(self, filename: str = "advanced_step_by_step_generation.json"):
        """保存生成过程到文件"""
        result = self.generate_target_formula()
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"生成过程已保存到 {filename}")

def main():
    """主函数"""
    # 创建生成器
    generator = AdvancedStepByStepFormulaGenerator(
        agent_count=3,
        variable_count=5,
        max_depth=4,
        target_complexity=50,
        show_detailed_fol=True
    )
    
    # 生成公式
    generator.save_to_file()

if __name__ == "__main__":
    main()
