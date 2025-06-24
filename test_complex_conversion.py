#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
复杂AEDNF/AECNF转换测试
测试更复杂的认知逻辑公式转换
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 直接导入模块内容，指定UTF-8编码
exec(open('AEDNF-AECNF-P.py', encoding='utf-8').read())

def create_complex_test_formulas():
    """创建复杂的测试公式"""
    formulas = []
    
    # 测试1: 简单的析取 - 应该展示DNF vs CNF的区别
    # (¬K_1(p) ∨ K_2(q)) ∧ (K_1(r) ∨ ¬K_2(s))
    formula1 = BinaryConnective(
        BinaryConnective(
            Negation(Knowledge(1, Proposition('p'))),
            Knowledge(2, Proposition('q')),
            '∨'
        ),
        BinaryConnective(
            Knowledge(1, Proposition('r')),
            Negation(Knowledge(2, Proposition('s'))),
            '∨'
        ),
        '∧'
    )
    formulas.append(("复杂合取-析取", formula1))
    
    # 测试2: 嵌套知识算子
    # K_1(K_2(p) ∨ ¬K_3(q))
    formula2 = Knowledge(1, 
        BinaryConnective(
            Knowledge(2, Proposition('p')),
            Negation(Knowledge(3, Proposition('q'))),
            '∨'
        )
    )
    formulas.append(("嵌套知识算子", formula2))
    
    # 测试3: 复杂的蕴含关系
    # (K_1(p) → K_2(q)) ∧ (¬K_2(q) → ¬K_1(p))
    formula3 = BinaryConnective(
        BinaryConnective(
            Knowledge(1, Proposition('p')),
            Knowledge(2, Proposition('q')),
            '→'
        ),
        BinaryConnective(
            Negation(Knowledge(2, Proposition('q'))),
            Negation(Knowledge(1, Proposition('p'))),
            '→'
        ),
        '∧'
    )
    formulas.append(("复杂蕴含关系", formula3))
    
    # 测试4: 多层嵌套
    # K_1(K_2(K_3(p) ∨ q) ∧ ¬K_3(r))
    formula4 = Knowledge(1,
        BinaryConnective(
            Knowledge(2,
                BinaryConnective(
                    Knowledge(3, Proposition('p')),
                    Proposition('q'),
                    '∨'
                )
            ),
            Negation(Knowledge(3, Proposition('r'))),
            '∧'
        )
    )
    formulas.append(("多层嵌套", formula4))
    
    # 测试5: 分配律测试 - 这应该显示DNF和CNF的明显区别
    # (p ∨ q) ∧ (r ∨ s) 
    # DNF: (p∧r) ∨ (p∧s) ∨ (q∧r) ∨ (q∧s)
    # CNF: (p∨q) ∧ (r∨s)
    formula5 = BinaryConnective(
        BinaryConnective(
            Proposition('p'),
            Proposition('q'),
            '∨'
        ),
        BinaryConnective(
            Proposition('r'),
            Proposition('s'),
            '∨'
        ),
        '∧'
    )
    formulas.append(("分配律测试", formula5))
    
    # 测试6: 知识算子的分配律
    # K_1(p ∨ q) vs (K_1(p) ∨ K_1(q))
    formula6a = Knowledge(1,
        BinaryConnective(
            Proposition('p'),
            Proposition('q'),
            '∨'
        )
    )
    formula6b = BinaryConnective(
        Knowledge(1, Proposition('p')),
        Knowledge(1, Proposition('q')),
        '∨'
    )
    formulas.append(("K_1(p ∨ q)", formula6a))
    formulas.append(("K_1(p) ∨ K_1(q)", formula6b))
    
    # 测试7: 复杂的否定传播
    # ¬(K_1(p) ∧ K_2(q)) ≡ (¬K_1(p) ∨ ¬K_2(q))
    formula7 = Negation(
        BinaryConnective(
            Knowledge(1, Proposition('p')),
            Knowledge(2, Proposition('q')),
            '∧'
        )
    )
    formulas.append(("复杂否定传播", formula7))
    
    # 测试8: 德摩根定律在知识算子上的应用
    # ¬(K_1(p) ∨ K_2(q)) ≡ (¬K_1(p) ∧ ¬K_2(q))
    formula8 = Negation(
        BinaryConnective(
            Knowledge(1, Proposition('p')),
            Knowledge(2, Proposition('q')),
            '∨'
        )
    )
    formulas.append(("德摩根定律", formula8))
    
    return formulas

def detailed_analysis(converter, formula, name):
    """详细分析单个公式的转换"""
    print(f"\n{'='*60}")
    print(f"测试公式: {name}")
    print(f"原公式: {formula}")
    print(f"{'='*60}")
    
    result = converter.convert_formula(formula)
    
    print(f"模态深度: {result.modal_depth}")
    print(f"转换时间: {result.conversion_time:.6f}s")
    print()
    
    # AEDNF分析
    print(f"AEDNF (析取正规形式):")
    print(f"  表示: {result.aednf}")
    print(f"  子句数量: {len(result.aednf.clauses)}")
    for i, clause in enumerate(result.aednf.clauses):
        print(f"  子句 {i+1}: {clause}")
        print(f"    - 客观部分: {clause.objective}")
        print(f"    - 正知识: {clause.positive_knowledge}")
        print(f"    - 负知识: {clause.negative_knowledge}")
    print()
    
    # AECNF分析
    print(f"AECNF (合取正规形式):")
    print(f"  表示: {result.aecnf}")
    print(f"  子句数量: {len(result.aecnf.clauses)}")
    for i, clause in enumerate(result.aecnf.clauses):
        print(f"  子句 {i+1}: {clause}")
        print(f"    - 客观部分: {clause.objective}")
        print(f"    - 负知识: {clause.negative_knowledge}")
        print(f"    - 正知识: {clause.positive_knowledge}")
    print()
    
    # 比较分析
    aednf_clauses = len(result.aednf.clauses)
    aecnf_clauses = len(result.aecnf.clauses)
    
    print(f"比较分析:")
    print(f"  AEDNF子句数: {aednf_clauses}")
    print(f"  AECNF子句数: {aecnf_clauses}")
    
    if aednf_clauses != aecnf_clauses:
        print(f"  ✓ 转换展示了DNF与CNF的结构差异")
    else:
        print(f"  - 两种形式的子句数相同")
    
    return result

def main():
    print("复杂AEDNF/AECNF转换测试")
    print("=" * 60)
    
    # 创建转换器
    agents = [1, 2, 3]
    converter = AEDNFAECNFConverter(agents)
    
    # 获取复杂测试公式
    formulas = create_complex_test_formulas()
    
    results = []
    
    # 逐一测试每个公式
    for name, formula in formulas:
        result = detailed_analysis(converter, formula, name)
        results.append((name, result))
    
    # 总结报告
    print(f"\n{'='*60}")
    print("总结报告")
    print(f"{'='*60}")
    
    print(f"总共测试了 {len(results)} 个公式")
    print()
    
    print("转换复杂度分析:")
    for name, result in results:
        aednf_clauses = len(result.aednf.clauses)
        aecnf_clauses = len(result.aecnf.clauses)
        complexity_indicator = "复杂" if abs(aednf_clauses - aecnf_clauses) > 0 else "简单"
        
        print(f"  {name:20} | 模态深度: {result.modal_depth} | "
              f"AEDNF: {aednf_clauses}子句 | AECNF: {aecnf_clauses}子句 | {complexity_indicator}")
    
    print()
    print("关键观察:")
    print("1. AEDNF (析取正规形式) 将公式表示为析取子句的析取")
    print("2. AECNF (合取正规形式) 将同一公式表示为合取子句的合取")
    print("3. 子句数量的差异反映了不同正规形式的结构特点")
    print("4. 模态深度影响转换的复杂性")

if __name__ == "__main__":
    main() 