"""
AEDNF/AECNF深度0操作构建器
专门处理纯命题逻辑（OBDD）的操作，输入输出都是AEDNF/AECNF对
"""

from typing import List, Tuple
from .models import ObjectiveFormula, AEDNFTerm, AECNFClause, AEDNF, AECNF, AEDNFAECNFPair, create_objective_pair
from .obdd import Node, OBDDBuilder, AND, OR, NOT, implies, display_traditional, reset_cache

class Depth0Builder:
    """
    深度0（纯命题逻辑）的AEDNF/AECNF操作构建器
    
    所有操作都基于OBDD，确保：
    1. 输入：AEDNF/AECNF对
    2. 输出：AEDNF/AECNF对
    3. 深度：始终为0（纯命题逻辑）
    """
    
    def __init__(self):
        """初始化构建器，确保OBDD环境clean"""
        reset_cache()
        self.obdd_builder = OBDDBuilder()
    
    def _reset_cache(self):
        """重置OBDD缓存"""
        reset_cache()
        self.obdd_builder = OBDDBuilder()
    
    def create_atom(self, var_name: str) -> AEDNFAECNFPair:
        """创建原子变量"""
        # 不要每次都重置缓存，只在初始化时重置
        return create_objective_pair(var_name)
    
    def create_true(self) -> AEDNFAECNFPair:
        """创建真值⊤"""
        return create_objective_pair("⊤")
    
    def create_false(self) -> AEDNFAECNFPair:
        """创建假值⊥"""
        return create_objective_pair("⊥")
    
    def land(self, pair1: AEDNFAECNFPair, pair2: AEDNFAECNFPair) -> AEDNFAECNFPair:
        """
        逻辑合取 (AND)
        
        AEDNF: (α₁ ∨ α₂ ∨ ...) ∧ (β₁ ∨ β₂ ∨ ...) = (α₁∧β₁) ∨ (α₁∧β₂) ∨ ...
        AECNF: (α₁ ∧ α₂ ∧ ...) ∧ (β₁ ∧ β₂ ∧ ...) = α₁ ∧ α₂ ∧ ... ∧ β₁ ∧ β₂ ∧ ...
        """
        # 获取OBDD节点
        node1_aednf = self._get_obdd_node(pair1.aednf)
        node2_aednf = self._get_obdd_node(pair2.aednf)
        node1_aecnf = self._get_obdd_node(pair1.aecnf)
        node2_aecnf = self._get_obdd_node(pair2.aecnf)
        
        # 执行OBDD操作
        result_aednf_node = AND(node1_aednf, node2_aednf)
        result_aecnf_node = AND(node1_aecnf, node2_aecnf)
        
        # 转换为AEDNF/AECNF对
        return self._nodes_to_pair(result_aednf_node, result_aecnf_node)
    
    def lor(self, pair1: AEDNFAECNFPair, pair2: AEDNFAECNFPair) -> AEDNFAECNFPair:
        """
        逻辑析取 (OR)
        
        AEDNF: (α₁ ∨ α₂ ∨ ...) ∨ (β₁ ∨ β₂ ∨ ...) = α₁ ∨ α₂ ∨ ... ∨ β₁ ∨ β₂ ∨ ...
        AECNF: (α₁ ∧ α₂ ∧ ...) ∨ (β₁ ∧ β₂ ∧ ...) = (α₁∨β₁) ∧ (α₁∨β₂) ∧ ...
        """
        # 获取OBDD节点
        node1_aednf = self._get_obdd_node(pair1.aednf)
        node2_aednf = self._get_obdd_node(pair2.aednf)
        node1_aecnf = self._get_obdd_node(pair1.aecnf)
        node2_aecnf = self._get_obdd_node(pair2.aecnf)
        
        # 执行OBDD操作
        result_aednf_node = OR(node1_aednf, node2_aednf)
        result_aecnf_node = OR(node1_aecnf, node2_aecnf)
        
        # 转换为AEDNF/AECNF对
        return self._nodes_to_pair(result_aednf_node, result_aecnf_node)
    
    def lnot(self, pair: AEDNFAECNFPair) -> AEDNFAECNFPair:
        """
        逻辑否定 (NOT)
        
        根据德摩根定律：
        ¬(α₁ ∨ α₂ ∨ ...) = ¬α₁ ∧ ¬α₂ ∧ ... (AEDNF → AECNF)
        ¬(α₁ ∧ α₂ ∧ ...) = ¬α₁ ∨ ¬α₂ ∨ ... (AECNF → AEDNF)
        """
        # 获取OBDD节点
        node_aednf = self._get_obdd_node(pair.aednf)
        node_aecnf = self._get_obdd_node(pair.aecnf)
        
        # 执行OBDD操作
        result_aednf_node = NOT(node_aecnf)  # ¬AECNF → AEDNF
        result_aecnf_node = NOT(node_aednf)  # ¬AEDNF → AECNF
        
        # 转换为AEDNF/AECNF对
        return self._nodes_to_pair(result_aednf_node, result_aecnf_node)
    
    def limplies(self, pair1: AEDNFAECNFPair, pair2: AEDNFAECNFPair) -> AEDNFAECNFPair:
        """
        逻辑蕴含 (IMPLIES)
        
        φ → ψ = ¬φ ∨ ψ
        """
        # 获取OBDD节点
        node1_aednf = self._get_obdd_node(pair1.aednf)
        node2_aednf = self._get_obdd_node(pair2.aednf)
        node1_aecnf = self._get_obdd_node(pair1.aecnf)
        node2_aecnf = self._get_obdd_node(pair2.aecnf)
        
        # 执行OBDD操作：φ → ψ = ¬φ ∨ ψ
        result_aednf_node = implies(node1_aednf, node2_aednf)
        result_aecnf_node = implies(node1_aecnf, node2_aecnf)
        
        # 转换为AEDNF/AECNF对
        return self._nodes_to_pair(result_aednf_node, result_aecnf_node)
    
    def lequiv(self, pair1: AEDNFAECNFPair, pair2: AEDNFAECNFPair) -> AEDNFAECNFPair:
        """
        逻辑等价 (EQUIV)
        
        φ ↔ ψ = (φ → ψ) ∧ (ψ → φ) = (¬φ ∨ ψ) ∧ (¬ψ ∨ φ)
        """
        # 先计算蕴含
        forward = self.limplies(pair1, pair2)
        backward = self.limplies(pair2, pair1)
        
        # 再计算合取
        return self.land(forward, backward)
    
    def lnand(self, pair1: AEDNFAECNFPair, pair2: AEDNFAECNFPair) -> AEDNFAECNFPair:
        """
        逻辑与非 (NAND)
        
        φ ⊼ ψ = ¬(φ ∧ ψ)
        """
        # 先计算合取
        conjunction = self.land(pair1, pair2)
        
        # 再计算否定
        return self.lnot(conjunction)
    
    def lnor(self, pair1: AEDNFAECNFPair, pair2: AEDNFAECNFPair) -> AEDNFAECNFPair:
        """
        逻辑或非 (NOR)
        
        φ ⊽ ψ = ¬(φ ∨ ψ)
        """
        # 先计算析取
        disjunction = self.lor(pair1, pair2)
        
        # 再计算否定
        return self.lnot(disjunction)
    
    def lxor(self, pair1: AEDNFAECNFPair, pair2: AEDNFAECNFPair) -> AEDNFAECNFPair:
        """
        逻辑异或 (XOR)
        
        φ ⊕ ψ = (φ ∧ ¬ψ) ∨ (¬φ ∧ ψ)
        """
        # 计算 (φ ∧ ¬ψ)
        not_pair2 = self.lnot(pair2)
        left_part = self.land(pair1, not_pair2)
        
        # 计算 (¬φ ∧ ψ)
        not_pair1 = self.lnot(pair1)
        right_part = self.land(not_pair1, pair2)
        
        # 计算析取
        return self.lor(left_part, right_part)
    
    def _get_obdd_node(self, formula) -> Node:
        """从AEDNF/AECNF公式中提取OBDD节点"""
        if formula.depth == 0:
            # 深度0，直接获取客观部分的OBDD节点
            from .obdd import nodeID_2_key, branch_cache
            
            if hasattr(formula, 'terms'):  # AEDNF
                term = formula.terms[0]  # 深度0只有一个项
                key = nodeID_2_key[term.objective_part.obdd_node_id]
            elif hasattr(formula, 'clauses'):  # AECNF
                clause = formula.clauses[0]  # 深度0只有一个子句
                key = nodeID_2_key[clause.objective_part.obdd_node_id]
            else:
                raise ValueError("未知的公式类型")
            
            return branch_cache[key]
        else:
            raise ValueError("此构建器只处理深度0的公式")
    
    def _nodes_to_pair(self, aednf_node: Node, aecnf_node: Node) -> AEDNFAECNFPair:
        """将OBDD节点转换为AEDNF/AECNF对"""
        # 创建客观公式
        obj_aednf = ObjectiveFormula(obdd_node_id=aednf_node.id, description="")
        obj_aecnf = ObjectiveFormula(obdd_node_id=aecnf_node.id, description="")
        
        # 创建项和子句
        term = AEDNFTerm(objective_part=obj_aednf)
        clause = AECNFClause(objective_part=obj_aecnf)
        
        # 创建AEDNF和AECNF
        aednf = AEDNF(terms=[term], depth=0)
        aecnf = AECNF(clauses=[clause], depth=0)
        
        return AEDNFAECNFPair(aednf=aednf, aecnf=aecnf)
    
    def build_complex_formula(self, operations: List[Tuple[str, ...]]) -> AEDNFAECNFPair:
        """
        构建复杂公式
        
        operations: 操作列表，每个元素是 (操作符, 参数1, 参数2, ...)
        例如: [('create_atom', 'v1'), ('create_atom', 'v2'), ('land', 0, 1)]
        """
        stack = []
        
        for i, operation in enumerate(operations):
            op = operation[0]
            args = operation[1:]
            
            if op == 'create_atom':
                var_name = args[0]
                stack.append(self.create_atom(var_name))
            
            elif op == 'create_true':
                stack.append(self.create_true())
            
            elif op == 'create_false':
                stack.append(self.create_false())
            
            elif op in ['land', 'lor', 'limplies', 'lequiv', 'lnand', 'lnor', 'lxor']:
                if len(stack) < 2:
                    raise ValueError(f"操作{op}需要2个参数，但栈中只有{len(stack)}个")
                
                pair2 = stack.pop()
                pair1 = stack.pop()
                
                if op == 'land':
                    result = self.land(pair1, pair2)
                elif op == 'lor':
                    result = self.lor(pair1, pair2)
                elif op == 'limplies':
                    result = self.limplies(pair1, pair2)
                elif op == 'lequiv':
                    result = self.lequiv(pair1, pair2)
                elif op == 'lnand':
                    result = self.lnand(pair1, pair2)
                elif op == 'lnor':
                    result = self.lnor(pair1, pair2)
                elif op == 'lxor':
                    result = self.lxor(pair1, pair2)
                
                stack.append(result)
            
            elif op == 'lnot':
                if len(stack) < 1:
                    raise ValueError(f"操作{op}需要1个参数，但栈中只有{len(stack)}个")
                
                pair = stack.pop()
                result = self.lnot(pair)
                stack.append(result)
            
            else:
                raise ValueError(f"未知操作: {op}")
        
        if len(stack) != 1:
            raise ValueError(f"构建完成，但栈中有{len(stack)}个结果，应该是1个")
        
        return stack[0]

# 测试函数
def test_depth0_builder():
    """测试深度0构建器"""
    print("=== 深度0 AEDNF/AECNF构建器测试 ===\n")
    
    builder = Depth0Builder()
    
    # 测试1：基本原子操作
    print("测试1：基本原子操作")
    atom1 = builder.create_atom("v1")
    atom2 = builder.create_atom("v2")
    
    print(f"原子v1: {display_traditional(builder._get_obdd_node(atom1.aednf))}")
    print(f"原子v2: {display_traditional(builder._get_obdd_node(atom2.aednf))}")
    
    # 测试2：基本逻辑操作
    print("\n测试2：基本逻辑操作")
    
    # v1 ∧ v2
    and_result = builder.land(atom1, atom2)
    print(f"v1 ∧ v2 = {display_traditional(builder._get_obdd_node(and_result.aednf))}")
    
    # v1 ∨ v2
    or_result = builder.lor(atom1, atom2)
    print(f"v1 ∨ v2 = {display_traditional(builder._get_obdd_node(or_result.aednf))}")
    
    # ¬v1
    not_result = builder.lnot(atom1)
    print(f"¬v1 = {display_traditional(builder._get_obdd_node(not_result.aednf))}")
    
    # v1 → v2
    implies_result = builder.limplies(atom1, atom2)
    print(f"v1 → v2 = {display_traditional(builder._get_obdd_node(implies_result.aednf))}")
    
    # 测试3：复杂公式构建
    print("\n测试3：复杂公式构建")
    
    # 构建 (v1 ∧ v2) ∨ ¬v3
    operations = [
        ('create_atom', 'v1'),
        ('create_atom', 'v2'),
        ('land', 0, 1),  # v1 ∧ v2
        ('create_atom', 'v3'),
        ('lnot', 3),     # ¬v3
        ('lor', 2, 4),   # (v1 ∧ v2) ∨ ¬v3
    ]
    
    complex_result = builder.build_complex_formula(operations)
    print(f"(v1 ∧ v2) ∨ ¬v3 = {display_traditional(builder._get_obdd_node(complex_result.aednf))}")
    
    # 测试4：其他操作符
    print("\n测试4：其他操作符")
    
    # v1 ⊕ v2 (异或)
    xor_result = builder.lxor(atom1, atom2)
    print(f"v1 ⊕ v2 = {display_traditional(builder._get_obdd_node(xor_result.aednf))}")
    
    # v1 ⊼ v2 (与非)
    nand_result = builder.lnand(atom1, atom2)
    print(f"v1 ⊼ v2 = {display_traditional(builder._get_obdd_node(nand_result.aednf))}")
    
    print("\n测试完成！")

if __name__ == "__main__":
    test_depth0_builder() 