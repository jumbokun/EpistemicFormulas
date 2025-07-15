import random
import sys
import os

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入必要的函数
from ebdd2_experiment import gimea_formula, V, AND, OR, NOT, K, reset_cache, display

def test_gimea_formula_generation():
    """演示gimea_formula函数如何生成公式"""
    
    print("=== gimea_formula 函数生成公式演示 ===\n")
    
    # 重置缓存
    reset_cache()
    
    # 测试参数
    num_var = 5  # 5个命题变量 v0, v1, v2, v3, v4
    complexity_levels = [1, 2, 3, 4, 5]
    deg_nesting_options = [0, 1]
    
    print(f"可用变量: v0, v1, v2, v3, v4 (共{num_var}个)\n")
    
    for deg_nesting in deg_nesting_options:
        print(f"--- 嵌套深度 deg_nesting = {deg_nesting} ---")
        
        for complexity in complexity_levels:
            print(f"\n复杂度 complexity = {complexity}:")
            
            # 生成3个示例公式
            for i in range(3):
                formula = gimea_formula(num_var=num_var, complexity=complexity, deg_nesting=deg_nesting)
                print(f"  示例{i+1}: {display(formula)}")
            
            print()
    
    print("\n=== 生成过程详解 ===")
    print("""
gimea_formula函数的工作原理：

1. 基础情况 (complexity=1):
   - 随机选择一个命题变量 V('v'+str(var_dice))
   - 例如：V('v2') 表示命题变量 v2

2. 连接词选择规则：
   - 如果 complexity >= 3:
     * deg_nesting=0: 只能选择 0,1,2 (否定、合取、析取)
     * deg_nesting=1: 可以选择 0,1,2,3 (否定、合取、析取、知识算子)
   - 如果 complexity < 3:
     * deg_nesting=0: 只能选择 0 (否定)
     * deg_nesting=1: 可以选择 0 或 3 (否定或知识算子)

3. 连接词含义：
   - 0: 否定 (¬φ)
   - 1: 合取 (φ∧ψ)
   - 2: 析取 (φ∨ψ)  
   - 3: 知识算子 (Kφ)

4. 递归生成：
   - 否定: negate(gimea_formula(complexity-1))
   - 合取: AND(gimea_formula(complex_dice), gimea_formula(complexity-complex_dice))
   - 析取: OR(gimea_formula(complex_dice), gimea_formula(complexity-complex_dice))
   - 知识算子: K(gimea_formula(complexity-1, deg_nesting-1))
""")

def demonstrate_step_by_step():
    """逐步演示公式生成过程"""
    
    print("\n=== 逐步演示生成过程 ===")
    
    # 重置缓存
    reset_cache()
    
    print("示例：生成一个复杂度为4，嵌套深度为1的公式")
    print("参数：num_var=3, complexity=4, deg_nesting=1")
    print()
    
    # 手动模拟生成过程
    print("步骤1: 检查 complexity=4 >= 3，deg_nesting=1")
    print("       可以选择连接词: 0,1,2,3")
    print("       假设随机选择 con_dice=1 (合取)")
    print()
    
    print("步骤2: 合取需要两个子公式")
    print("       随机分配复杂度: complex_dice=2, complexity-complex_dice=2")
    print("       需要生成: AND(gimea_formula(2), gimea_formula(2))")
    print()
    
    print("步骤3: 生成第一个子公式 gimea_formula(2)")
    print("       检查 complexity=2 < 3，deg_nesting=1")
    print("       只能选择: 0 或 3")
    print("       假设选择 con_dice=0 (否定)")
    print("       生成: negate(gimea_formula(1))")
    print("       而 gimea_formula(1) = V('v1') (假设)")
    print("       所以第一个子公式 = NOT(V('v1'))")
    print()
    
    print("步骤4: 生成第二个子公式 gimea_formula(2)")
    print("       同样选择 con_dice=3 (知识算子)")
    print("       生成: K(gimea_formula(1, 0))")
    print("       而 gimea_formula(1, 0) = V('v2') (假设)")
    print("       所以第二个子公式 = K(V('v2'))")
    print()
    
    print("步骤5: 最终公式")
    print("       结果 = AND(NOT(V('v1')), K(V('v2')))")
    print("       表示: (¬v1) ∧ K(v2)")
    print()
    
    # 实际生成一个公式验证
    print("实际生成结果:")
    formula = gimea_formula(num_var=3, complexity=4, deg_nesting=1)
    print(f"生成的公式: {display(formula)}")

if __name__ == "__main__":
    test_gimea_formula_generation()
    demonstrate_step_by_step() 