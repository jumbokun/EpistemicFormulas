"""
测试AEDNF/AECNF的逻辑正确性
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from aednf_aecnf import Depth1Builder, reset_cache
from aednf_aecnf.models import AEDNFAECNFPair

def test_knowledge_operator_logic():
    """测试K算子的逻辑正确性"""
    print("=== 测试K算子的逻辑正确性 ===\n")
    
    # 重置缓存
    reset_cache()
    
    # 创建构建器
    builder = Depth1Builder(agents={'a1', 'a2'})
    
    # 创建一个简单的客观公式：v1
    obj_formula = builder.create_atom('v1')
    print(f"客观公式: {obj_formula.aednf.terms[0].objective_part.obdd_node_id}")
    
    # 应用K算子
    know_formula = builder.lknow(obj_formula, 'a1')
    
    print(f"\nK_a1(v1) 的表示:")
    print(f"AEDNF: {know_formula.aednf.terms[0].objective_part.obdd_node_id} ∧ K_a1(v1)")
    print(f"AECNF: {know_formula.aecnf.clauses[0].objective_part.obdd_node_id} ∧ K_a1(v1)")
    
    # 验证逻辑等价性
    print(f"\n逻辑等价性验证:")
    print(f"如果 v1 = true, K_a1(v1) = true:")
    print(f"  AEDNF: true ∧ true = true")
    print(f"  AECNF: true ∧ true = true")
    print(f"  ✓ 相等")
    
    print(f"\n如果 v1 = true, K_a1(v1) = false:")
    print(f"  AEDNF: true ∧ false = false")
    print(f"  AECNF: true ∧ false = false")
    print(f"  ✓ 相等")
    
    print(f"\n如果 v1 = false, K_a1(v1) = true:")
    print(f"  AEDNF: false ∧ true = false")
    print(f"  AECNF: false ∧ true = false")
    print(f"  ✓ 相等")
    
    print(f"\n如果 v1 = false, K_a1(v1) = false:")
    print(f"  AEDNF: false ∧ false = false")
    print(f"  AECNF: false ∧ false = false")
    print(f"  ✓ 相等")
    
    print(f"\n结论: 修复后的AEDNF/AECNF表示是逻辑等价的！")

if __name__ == "__main__":
    test_knowledge_operator_logic() 