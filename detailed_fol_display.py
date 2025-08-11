import json
from typing import Dict, Any

def get_nested_formula_description(formula_depth: int, agent: str) -> str:
    """根据公式深度生成嵌套公式的描述"""
    if formula_depth == 0:
        return "p"  # 基础命题变量
    elif formula_depth == 1:
        return f"K_{agent}(p)"  # 深度1的认知公式
    elif formula_depth == 2:
        return f"K_{agent}(K_{agent}(p))"  # 深度2的认知公式
    else:
        return f"φ_{formula_depth}"  # 占位符

def formula_to_fol_string(formula_data: Dict[str, Any], is_aednf: bool = False) -> str:
    """将公式数据转换为FOL格式的字符串"""
    if is_aednf:
        structure = formula_data["aednf"]
        terms_key = "terms"
    else:
        structure = formula_data["aecnf"]
        terms_key = "clauses"
    
    if not structure[terms_key]:
        return "⊥"
    
    # 对于AEDNF，我们显示第一个项；对于AECNF，我们显示第一个子句
    item = structure[terms_key][0]
    objective_part = item["objective_part"]["description"]
    
    # 构建认知文字部分
    positive_literals = []
    negative_literals = []
    
    for lit in item["positive_literals"]:
        agent = lit["agent"]
        formula_depth = lit["formula_depth"]
        # 展开嵌套公式
        sub_formula = get_nested_formula_description(formula_depth, agent)
        positive_literals.append(f"K_{agent}({sub_formula})")
    
    for lit in item["negative_literals"]:
        agent = lit["agent"]
        formula_depth = lit["formula_depth"]
        # 展开嵌套公式
        sub_formula = get_nested_formula_description(formula_depth, agent)
        negative_literals.append(f"¬K_{agent}({sub_formula})")
    
    # 组合所有部分
    parts = []
    
    # 处理客观部分
    if objective_part == "⊤":
        if not positive_literals and not negative_literals:
            return "⊤"
    elif objective_part == "⊥":
        if not positive_literals and not negative_literals:
            return "⊥"
    else:
        parts.append(objective_part)
    
    # 添加认知文字
    parts.extend(positive_literals)
    parts.extend(negative_literals)
    
    if not parts:
        return "⊤"
    
    return "(" + " ∧ ".join(parts) + ")"

def display_formulas_detailed(data: Dict[str, Any]) -> str:
    """详细显示FOL格式的公式"""
    result = "=== 认知逻辑公式 (详细FOL格式) ===\n\n"
    
    for depth_key, formulas in data.items():
        depth_num = depth_key.split('_')[1]
        result += f"深度 {depth_num}:\n"
        
        for i, formula in enumerate(formulas, 1):
            aednf_string = formula_to_fol_string(formula, is_aednf=True)
            aecnf_string = formula_to_fol_string(formula, is_aednf=False)
            result += f"  公式_{i}_深度_{depth_num}:\n"
            result += f"    AEDNF_{i}_f = {aednf_string}\n"
            result += f"    AECNF_{i}_f = {aecnf_string}\n"
        
        result += "\n"
    
    return result

def main():
    """主函数"""
    try:
        with open('generated_formulas.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print("数据加载成功")
        print(f"数据键: {list(data.keys())}")
        
        result = display_formulas_detailed(data)
        print(result)
        
    except FileNotFoundError:
        print("错误: 找不到 'generated_formulas.json' 文件")
        print("请先运行 'python formula_generator.py' 生成公式")
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 