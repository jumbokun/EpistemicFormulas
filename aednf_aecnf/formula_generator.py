"""
深度1多代理认知公式生成器

支持指定：
- 目标长度（必须达到）
- 代理数量
- 原子命题数量  
- 生成公式数量

确保所有客观部分都由OBDD表示
"""

import random
import sys
import os
from typing import List, Set, Tuple, Optional
from dataclasses import dataclass

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from aednf_aecnf import Depth1Builder, reset_cache, display_traditional
from aednf_aecnf.models import AEDNFAECNFPair

@dataclass
class GenerationConfig:
    """生成配置"""
    target_length: int          # 目标长度
    num_agents: int            # 代理数量
    num_variables: int         # 原子命题数量
    num_formulas: int          # 生成公式数量
    max_depth: int = 1         # 最大深度（固定为1）
    
    def __post_init__(self):
        """验证配置"""
        if self.target_length < 1:
            raise ValueError("目标长度必须大于0")
        if self.num_agents < 1:
            raise ValueError("代理数量必须大于0")
        if self.num_variables < 1:
            raise ValueError("原子命题数量必须大于0")
        if self.num_formulas < 1:
            raise ValueError("生成公式数量必须大于0")
        if self.max_depth != 1:
            raise ValueError("当前只支持深度1")

class FormulaGenerator:
    """
    深度1多代理认知公式生成器
    
    生成策略：
    1. 从原子命题开始
    2. 递归应用操作符直到达到目标长度
    3. 确保所有客观部分都是OBDD表示
    """
    
    def __init__(self, config: GenerationConfig):
        """
        初始化生成器
        
        Args:
            config: 生成配置
        """
        self.config = config
        
        # 生成代理名称
        self.agents = {f'a{i+1}' for i in range(config.num_agents)}
        
        # 生成原子命题名称
        self.variables = {f'v{i+1}' for i in range(config.num_variables)}
        
        # 创建深度1构建器
        self.builder = Depth1Builder(agents=self.agents)
        
        # 缓存已生成的原子命题
        self._atom_cache = {}
        
    def _get_atom(self, var_name: str) -> AEDNFAECNFPair:
        """获取原子命题，使用缓存"""
        if var_name not in self._atom_cache:
            self._atom_cache[var_name] = self.builder.create_atom(var_name)
        return self._atom_cache[var_name]
    
    def _get_random_atom(self) -> AEDNFAECNFPair:
        """随机获取一个原子命题"""
        var_dice = random.randint(0, self.config.num_variables - 1)
        var_name = f'v{var_dice + 1}'
        return self._get_atom(var_name)
    
    def _get_random_agent(self) -> str:
        """随机获取一个代理"""
        return random.choice(list(self.agents))
    
    def _apply_operator(self, operator: str, *args) -> AEDNFAECNFPair:
        """应用操作符"""
        try:
            if operator == 'negate':
                return self.builder.lnot(args[0])
            elif operator == 'AND':
                return self.builder.land(args[0], args[1])
            elif operator == 'OR':
                return self.builder.lor(args[0], args[1])
            elif operator == 'K':
                return self.builder.lknow(args[0], args[1])
            else:
                raise ValueError(f"未知操作符: {operator}")
        except Exception as e:
            # 如果操作失败，返回一个简单的原子命题
            return self._get_random_atom()
    
    def _generate_formula_recursive(self, complexity: int, deg_nesting: int = 1, step: int = 1) -> AEDNFAECNFPair:
        """
        递归生成公式，参考gimea_formula的长度控制逻辑
        
        Args:
            complexity: 复杂度参数，控制公式的复杂度
            deg_nesting: 嵌套深度，控制K算子的使用
            step: 当前步骤编号
            
        Returns:
            生成的AEDNF/AECNF对
        """
        print(f"\n=== 步骤{step}: 复杂度={complexity}, 嵌套深度={deg_nesting} ===")
        
        # 如果复杂度为1，返回原子命题
        if complexity == 1:
            atom = self._get_random_atom()
            print(f"生成原子命题:")
            print(f"  AEDNF: {self._display_aednf(atom.aednf)}")
            print(f"  AECNF: {self._display_aecnf(atom.aecnf)}")
            return atom
        
        # 根据复杂度和嵌套深度选择操作符
        if complexity >= 3:
            if deg_nesting == 0:
                # 深度0，不使用K算子
                con_dice = random.randint(0, 2)
            else:
                # 深度1，可以使用K算子
                con_dice = random.randint(0, 3)
        else:
            if deg_nesting == 0:
                # 复杂度小于3且深度0，只使用否定
                con_dice = 0
            else:
                # 复杂度小于3且深度1，使用否定或K算子
                con_dice = random.choice([0, 3])
        
        """
        connective dice =
        0: negation
        1: conjunction
        2: disjunction
        3: K modality
        """
        if con_dice == 0:
            # 否定：复杂度减1
            print(f"选择操作符: 否定")
            sub_formula = self._generate_formula_recursive(complexity=complexity-1, deg_nesting=deg_nesting, step=step+1)
            result = self._apply_operator('negate', sub_formula)
            print(f"否定后:")
            print(f"  AEDNF: {self._display_aednf(result.aednf)}")
            print(f"  AECNF: {self._display_aecnf(result.aecnf)}")
            return result
        elif con_dice == 1:
            # 合取：复杂度分配给左右子公式
            print(f"选择操作符: 合取")
            complex_dice = random.randint(1, complexity-2)
            left_formula = self._generate_formula_recursive(complexity=complex_dice, deg_nesting=deg_nesting, step=step+1)
            right_formula = self._generate_formula_recursive(complexity=complexity-complex_dice, deg_nesting=deg_nesting, step=step+2)
            result = self._apply_operator('AND', left_formula, right_formula)
            print(f"合取后:")
            print(f"  AEDNF: {self._display_aednf(result.aednf)}")
            print(f"  AECNF: {self._display_aecnf(result.aecnf)}")
            return result
        elif con_dice == 2:
            # 析取：复杂度分配给左右子公式
            print(f"选择操作符: 析取")
            complex_dice = random.randint(1, complexity-2)
            left_formula = self._generate_formula_recursive(complexity=complex_dice, deg_nesting=deg_nesting, step=step+1)
            right_formula = self._generate_formula_recursive(complexity=complexity-complex_dice, deg_nesting=deg_nesting, step=step+2)
            result = self._apply_operator('OR', left_formula, right_formula)
            print(f"析取后:")
            print(f"  AEDNF: {self._display_aednf(result.aednf)}")
            print(f"  AECNF: {self._display_aecnf(result.aecnf)}")
            return result
        else:
            # K算子：复杂度减1，嵌套深度减1
            print(f"选择操作符: K算子")
            sub_formula = self._generate_formula_recursive(complexity=complexity-1, deg_nesting=deg_nesting-1, step=step+1)
            agent = self._get_random_agent()
            result = self._apply_operator('K', sub_formula, agent)
            print(f"K算子后:")
            print(f"  AEDNF: {self._display_aednf(result.aednf)}")
            print(f"  AECNF: {self._display_aecnf(result.aecnf)}")
            return result
    
    def _generate_depth0_formula(self, complexity: int) -> AEDNFAECNFPair:
        """
        专门生成深度0的公式（用于K算子的参数）
        
        Args:
            complexity: 复杂度参数
            
        Returns:
            生成的深度0的AEDNF/AECNF对
        """
        # 如果复杂度为1，返回原子命题
        if complexity == 1:
            return self._get_random_atom()
        
        # 深度0只能使用基本逻辑操作符
        if complexity >= 3:
            con_dice = random.randint(0, 2)
        else:
            con_dice = 0
        
        if con_dice == 0:
            # 否定：复杂度减1
            return self._apply_operator('negate', 
                self._generate_depth0_formula(complexity=complexity-1))
        elif con_dice == 1:
            # 合取：复杂度分配给左右子公式
            complex_dice = random.randint(1, complexity-2)
            return self._apply_operator('AND',
                self._generate_depth0_formula(complexity=complex_dice),
                self._generate_depth0_formula(complexity=complexity-complex_dice))
        else:
            # 析取：复杂度分配给左右子公式
            complex_dice = random.randint(1, complexity-2)
            return self._apply_operator('OR',
                self._generate_depth0_formula(complexity=complex_dice),
                self._generate_depth0_formula(complexity=complexity-complex_dice))
    
    def _calculate_formula_length(self, formula: AEDNFAECNFPair) -> int:
        """计算公式的长度（操作符和原子命题的总数）"""
        if formula.aednf.depth == 0:
            # 深度0公式，计算OBDD节点数
            from aednf_aecnf.obdd import rt_nodes_list, nodeID_2_key, branch_cache
            if formula.aednf.terms:
                node_id = formula.aednf.terms[0].objective_part.obdd_node_id
                if node_id in nodeID_2_key:
                    key = nodeID_2_key[node_id]
                    node = branch_cache[key]
                    return len(rt_nodes_list(node))
            return 1  # 原子命题
        else:
            # 深度1公式，计算客观部分 + 知识文字
            total_length = 0
            for term in formula.aednf.terms:
                # 客观部分
                if term.objective_part.obdd_node_id == 0:  # false_node
                    total_length += 0  # ⊥不计算长度
                elif term.objective_part.obdd_node_id == 1:  # true_node
                    total_length += 0  # ⊤不计算长度
                else:
                    # 客观部分是一个OBDD，计算其节点数
                    from aednf_aecnf.obdd import rt_nodes_list, nodeID_2_key, branch_cache
                    if term.objective_part.obdd_node_id in nodeID_2_key:
                        key = nodeID_2_key[term.objective_part.obdd_node_id]
                        node = branch_cache[key]
                        total_length += len(rt_nodes_list(node))
                
                # 知识文字：每个知识文字算1个长度
                total_length += len(term.positive_literals) + len(term.negative_literals)
            
            return total_length
    
    def generate_single_formula(self) -> AEDNFAECNFPair:
        """生成单个公式"""
        # 移除reset_cache()调用，避免清空OBDD缓存
        
        # 使用递归生成策略
        formula = self._generate_formula_recursive(self.config.target_length, 1)
        
        # 计算实际复杂度
        actual_complexity = self._calculate_formula_length(formula)
        print(f"  目标长度: {self.config.target_length}, 实际复杂度: {actual_complexity}")
        
        return formula
    
    def generate_formulas(self) -> List[AEDNFAECNFPair]:
        """生成指定数量的公式"""
        formulas = []
        
        for i in range(self.config.num_formulas):
            try:
                formula = self.generate_single_formula()
                formulas.append(formula)
                print(f"✓ 生成公式 {i+1}/{self.config.num_formulas}")
            except Exception as e:
                print(f"✗ 生成公式 {i+1} 失败: {e}")
                # 生成一个简单的原子命题作为备选
                reset_cache()
                backup_formula = self._get_random_atom()
                formulas.append(backup_formula)
        
        return formulas
    
    def display_formula(self, formula: AEDNFAECNFPair, index: int = None) -> str:
        """显示公式"""
        prefix = f"公式 {index}: " if index is not None else ""
        
        # 显示AEDNF
        aednf_str = self._display_aednf(formula.aednf)
        
        # 显示AECNF
        aecnf_str = self._display_aecnf(formula.aecnf)
        
        return f"{prefix}AEDNF: {aednf_str} | AECNF: {aecnf_str}"
    
    def _display_aednf(self, aednf) -> str:
        """显示AEDNF"""
        if aednf.depth == 0:
            # 深度0，显示OBDD
            from aednf_aecnf.obdd import nodeID_2_key, branch_cache
            if aednf.terms:
                node_id = aednf.terms[0].objective_part.obdd_node_id
                if node_id in nodeID_2_key:
                    key = nodeID_2_key[node_id]
                    node = branch_cache[key]
                    return display_traditional(node)
            return "v?"
        else:
            # 深度1，显示结构
            terms = []
            for term in aednf.terms:
                # 客观部分
                obj_str = self._display_objective_part(term.objective_part)
                
                # 知识文字 - 显示具体的客观公式
                pos_lits = []
                for lit in term.positive_literals:
                    # lit.formula是AEDNF类型，直接显示
                    obj_formula_str = self._display_aednf(lit.formula)
                    pos_lits.append(f"K_{lit.agent}({obj_formula_str})")
                
                neg_lits = []
                for lit in term.negative_literals:
                    # lit.formula是AEDNF类型，直接显示
                    obj_formula_str = self._display_aednf(lit.formula)
                    neg_lits.append(f"¬K_{lit.agent}({obj_formula_str})")
                
                parts = [obj_str] + pos_lits + neg_lits
                terms.append(" ∧ ".join(parts))
            
            return " ∨ ".join(terms)
    
    def _display_aecnf(self, aecnf) -> str:
        """显示AECNF"""
        if aecnf.depth == 0:
            # 深度0，显示OBDD
            from aednf_aecnf.obdd import nodeID_2_key, branch_cache
            if aecnf.clauses:
                node_id = aecnf.clauses[0].objective_part.obdd_node_id
                if node_id in nodeID_2_key:
                    key = nodeID_2_key[node_id]
                    node = branch_cache[key]
                    return display_traditional(node)
            return "v?"
        else:
            # 深度1，显示结构
            clauses = []
            for clause in aecnf.clauses:
                # 客观部分
                obj_str = self._display_objective_part(clause.objective_part)
                
                # 知识文字 - 显示具体的客观公式
                pos_lits = []
                for lit in clause.positive_literals:
                    # lit.formula是AEDNF类型，直接显示
                    obj_formula_str = self._display_aednf(lit.formula)
                    pos_lits.append(f"K_{lit.agent}({obj_formula_str})")
                
                neg_lits = []
                for lit in clause.negative_literals:
                    # lit.formula是AEDNF类型，直接显示
                    obj_formula_str = self._display_aednf(lit.formula)
                    neg_lits.append(f"¬K_{lit.agent}({obj_formula_str})")
                
                parts = [obj_str] + pos_lits + neg_lits
                clauses.append(" ∨ ".join(parts))  # 子句内部用 ∨ 连接（析取）
            
            return " ∧ ".join(clauses)  # 子句之间用 ∧ 连接（合取）
    
    def _display_objective_part(self, obj_part) -> str:
        """显示客观部分"""
        from aednf_aecnf.obdd import nodeID_2_key, branch_cache, number_2_symbol
        node_id = obj_part.obdd_node_id
        
        # 检查是否是特殊节点
        if node_id == 0:  # false_node
            return "⊥"
        elif node_id == 1:  # true_node
            return "⊤"
        elif node_id in nodeID_2_key:
            key = nodeID_2_key[node_id]
            node = branch_cache[key]
            # 如果节点有变量ID，尝试从number_2_symbol中找到变量名
            if node.var_id > 0 and node.var_id in number_2_symbol:
                return number_2_symbol[node.var_id]
            else:
                return display_traditional(node)
        else:
            # 尝试从number_2_symbol中找到变量名（使用节点ID作为键）
            if node_id in number_2_symbol:
                return number_2_symbol[node_id]
            else:
                return f"v{node_id}"  # 使用节点ID作为变量名

def test_generator():
    """测试公式生成器"""
    print("=== 重新设计的深度1多代理认知公式生成器测试 ===")
    
    # 配置：长度4，代理数量2
    config = GenerationConfig(
        target_length=4,      # 目标长度改为4
        num_agents=2,         # 代理数量改为2
        num_variables=3,      # 原子命题数量改为3
        num_formulas=3        # 生成公式数量改为3
    )
    
    # 创建生成器
    generator = FormulaGenerator(config)
    
    print("\n配置:")
    print(f"  目标长度: {config.target_length}")
    print(f"  代理数量: {config.num_agents} ({', '.join(generator.agents)})")
    print(f"  原子命题数量: {config.num_variables} ({', '.join([f'v{i+1}' for i in range(config.num_variables)])})")
    print(f"  生成公式数量: {config.num_formulas}")
    
    # 生成公式
    formulas = generator.generate_formulas()
    
    print("\n生成的公式:\n")
    
    # 显示每个公式
    for i, formula in enumerate(formulas, 1):
        formula_str = generator.display_formula(formula, i)
        print(formula_str)
        
        # 显示详细信息
        print(f"  深度: {formula.aednf.depth}")
        print(f"  AEDNF项数: {len(formula.aednf.terms)}")
        print(f"  AECNF子句数: {len(formula.aecnf.clauses)}")
        
        # 计算知识文字数量
        knowledge_count = 0
        for term in formula.aednf.terms:
            knowledge_count += len(term.positive_literals) + len(term.negative_literals)
        print(f"  知识文字数量: {knowledge_count}")
        print()
    
    print("=== 测试完成 ===")

def test_step_by_step():
    """逐步测试，直接使用gimea_formula生成OBDD公式，然后转换为AEDNF/AECNF"""
    print("=== 逐步测试AEDNF/AECNF操作 ===")
    
    # 重置缓存
    from aednf_aecnf.obdd import reset_cache
    reset_cache()
    
    print("\n配置:")
    print(f"  目标复杂度: 4")
    print(f"  代理数量: 2")
    print(f"  原子命题数量: 3")
    
    print("\n=== 使用gimea_formula生成公式 ===")
    
    # 直接使用gimea_formula生成OBDD公式
    from aednf_aecnf.obdd import gimea_formula, display_traditional
    
    formula_node = gimea_formula(num_var=3, complexity=4, deg_nesting=1)
    
    print(f"生成的OBDD公式: {display_traditional(formula_node)}")
    
    # TODO: 将OBDD公式转换为AEDNF/AECNF
    print("\n=== 转换为AEDNF/AECNF ===")
    print("TODO: 实现OBDD到AEDNF/AECNF的转换")
    
    print("\n=== 测试完成 ===")

if __name__ == "__main__":
    test_step_by_step() 