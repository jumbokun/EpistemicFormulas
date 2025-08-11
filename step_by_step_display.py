import json
from typing import Dict, Any

def display_step_by_step_generation(filename: str = "step_by_step_generation.json"):
    """显示逐步生成过程"""
    
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print("=" * 60)
    print("逐步公式生成过程")
    print("=" * 60)
    
    # 显示参数
    params = data["parameters"]
    print(f"参数设置:")
    print(f"  - 代理数量: {params['agent_count']}")
    print(f"  - 变量数量: {params['variable_count']}")
    print(f"  - 最大深度: {params['max_depth']}")
    print(f"  - 目标复杂度: {params['target_complexity']}")
    print(f"  - 代理列表: {data['agents']}")
    print(f"  - 变量列表: {data['variables']}")
    print()
    
    # 显示生成步骤
    print("生成步骤:")
    print("-" * 60)
    
    for step in data["generation_steps"]:
        step_num = step["step"]
        step_type = step["type"]
        description = step["description"]
        complexity = step.get("complexity")
        
        print(f"步骤 {step_num}: {step_type}")
        print(f"  描述: {description}")
        if complexity:
            print(f"  复杂度: {complexity}")
        
        # 如果有子公式，显示它们
        if step.get("sub_formulas"):
            print(f"  子公式:")
            for i, sub_formula in enumerate(step["sub_formulas"]):
                if sub_formula and "aednf" in sub_formula and sub_formula["aednf"]["terms"]:
                    term = sub_formula["aednf"]["terms"][0]
                    if term["objective_part"]["description"]:
                        print(f"    {i+1}. {term['objective_part']['description']}")
                    else:
                        print(f"    {i+1}. [复杂公式]")
        
        # 如果有生成的公式，显示它
        if step.get("formula"):
            formula = step["formula"]
            if formula and "aednf" in formula and formula["aednf"]["terms"]:
                term = formula["aednf"]["terms"][0]
                if term["objective_part"]["description"]:
                    print(f"  生成公式: {term['objective_part']['description']}")
                else:
                    print(f"  生成公式: [复杂公式]")
        
        print()
    
    # 显示最终结果
    print("=" * 60)
    print("最终结果:")
    print(f"总步骤数: {data['total_steps']}")
    
    final_formula = data["final_formula"]
    if final_formula and "aednf" in final_formula and final_formula["aednf"]["terms"]:
        term = final_formula["aednf"]["terms"][0]
        if term["objective_part"]["description"]:
            print(f"最终公式: {term['objective_part']['description']}")
        else:
            print(f"最终公式: [复杂公式]")
    
    print("=" * 60)

def display_formula_details(filename: str = "step_by_step_generation.json"):
    """显示公式的详细信息"""
    
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print("\n" + "=" * 60)
    print("公式详细信息")
    print("=" * 60)
    
    final_formula = data["final_formula"]
    if final_formula:
        print("AEDNF部分:")
        if "aednf" in final_formula and final_formula["aednf"]["terms"]:
            for i, term in enumerate(final_formula["aednf"]["terms"]):
                print(f"  项 {i+1}:")
                print(f"    客观部分: {term['objective_part']['description']}")
                if term["positive_literals"]:
                    print(f"    正知识文字: {len(term['positive_literals'])} 个")
                if term["negative_literals"]:
                    print(f"    负知识文字: {len(term['negative_literals'])} 个")
        
        print("\nAECNF部分:")
        if "aecnf" in final_formula and final_formula["aecnf"]["clauses"]:
            for i, clause in enumerate(final_formula["aecnf"]["clauses"]):
                print(f"  子句 {i+1}:")
                print(f"    客观部分: {clause['objective_part']['description']}")
                if clause["positive_literals"]:
                    print(f"    正知识文字: {len(clause['positive_literals'])} 个")
                if clause["negative_literals"]:
                    print(f"    负知识文字: {len(clause['negative_literals'])} 个")
    
    print("=" * 60)

def main():
    """主函数"""
    try:
        display_step_by_step_generation()
        display_formula_details()
    except FileNotFoundError:
        print("错误: 找不到 step_by_step_generation.json 文件")
        print("请先运行 step_by_step_formula_generator.py 生成数据")
    except Exception as e:
        print(f"错误: {e}")

if __name__ == "__main__":
    main()
