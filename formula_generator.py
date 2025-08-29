#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
认知逻辑公式生成器

从0开始根据参数构造n个目标式子，并且记录所用时间以及最终结果
"""

import sys
import os
import time
import random
import json
import argparse
from typing import List, Dict, Optional
from datetime import datetime

# 导入主系统
from epistemic_logic_system import (
    FormulaGenerator, create_objective_pair, know, land, lor, lnot, sat_pair,
    simplify_pair, simple_deintrospective_k
)


class AdvancedFormulaGenerator:
    """高级公式生成器"""
    
    def __init__(self, 
                 max_depth: int = 3, 
                 max_agents: int = 3, 
                 max_vars: int = 10,
                 target_length: Optional[int] = None,
                 seed: Optional[int] = None):
        self.max_depth = max_depth
        self.max_agents = max_agents
        self.max_vars = max_vars
        self.target_length = target_length
        self.agents = [f"agent_{i}" for i in range(max_agents)]
        self.variables = [f"p{i}" for i in range(max_vars)]
        
        if seed is not None:
            random.seed(seed)
        
        self.generator = FormulaGenerator(max_depth, max_agents, max_vars)
        self.reset_cache()
    
    def reset_cache(self):
        """重置缓存"""
        self.generator.reset_cache()
    
    def generate_formula_with_constraints(self, 
                                        target_depth: Optional[int] = None,
                                        use_knowledge: bool = True,
                                        use_and: bool = True,
                                        use_or: bool = True,
                                        use_not: bool = True,
                                        min_depth: int = 0) -> 'AEDNFAECNFPair':
        """根据约束生成公式"""
        if target_depth is None:
            target_depth = random.randint(min_depth, self.max_depth)
        
        if target_depth == 0:
            # 生成基本变量
            var = random.choice(self.variables)
            return create_objective_pair(var)
        
        # 根据约束选择操作类型
        available_ops = []
        if use_knowledge:
            available_ops.append('knowledge')
        if use_and:
            available_ops.append('and')
        if use_or:
            available_ops.append('or')
        if use_not:
            available_ops.append('not')
        
        # 如果没有可用操作，生成变量
        if not available_ops:
            var = random.choice(self.variables)
            return create_objective_pair(var)
        
        # 正常选择操作类型
        op_type = random.choice(available_ops)
        
        # 如果指定了目标长度，尝试控制公式长度
        if self.target_length is not None:
            # 估算当前公式长度
            current_length = self._estimate_formula_length(target_depth)
            # 允许更大的误差范围，只在明显超长时才限制
            if current_length > self.target_length * 2.0:
                # 如果估算长度远超过目标长度，有30%概率生成变量
                if random.random() < 0.3:
                    var = random.choice(self.variables)
                    return create_objective_pair(var)
        
        op_type = random.choice(available_ops)
        
        if op_type == 'knowledge':
            sub_formula = self.generate_formula_with_constraints(
                target_depth - 1, use_knowledge, use_and, use_or, use_not, min_depth
            )
            agent = random.choice(self.agents)
            return know(sub_formula, agent)
        
        elif op_type == 'and':
            left = self.generate_formula_with_constraints(
                target_depth - 1, use_knowledge, use_and, use_or, use_not, min_depth
            )
            right = self.generate_formula_with_constraints(
                target_depth - 1, use_knowledge, use_and, use_or, use_not, min_depth
            )
            return land(left, right)
        
        elif op_type == 'or':
            left = self.generate_formula_with_constraints(
                target_depth - 1, use_knowledge, use_and, use_or, use_not, min_depth
            )
            right = self.generate_formula_with_constraints(
                target_depth - 1, use_knowledge, use_and, use_or, use_not, min_depth
            )
            return lor(left, right)
        
        elif op_type == 'not':
            sub_formula = self.generate_formula_with_constraints(
                target_depth - 1, use_knowledge, use_and, use_or, use_not, min_depth
            )
            return lnot(sub_formula)
    
    def generate_formulas_batch(self, 
                              count: int,
                              target_depth: Optional[int] = None,
                              use_knowledge: bool = True,
                              use_and: bool = True,
                              use_or: bool = True,
                              use_not: bool = True,
                              min_depth: int = 0) -> List['AEDNFAECNFPair']:
        """批量生成公式"""
        formulas = []
        for i in range(count):
            formula = self.generate_formula_with_constraints(
                target_depth, use_knowledge, use_and, use_or, use_not, min_depth
            )
            formulas.append(formula)
        return formulas
    
    def test_formulas_detailed(self, formulas: List['AEDNFAECNFPair']) -> List[Dict]:
        """详细测试公式"""
        results = []
        total_start_time = time.time()
        
        for i, formula in enumerate(formulas):
            start_time = time.time()
            try:
                is_sat = sat_pair(formula)
                end_time = time.time()
                
                # 分析公式结构
                structure = self.analyze_formula_structure(formula)
                formula_length = self.get_formula_length(formula)
                
                results.append({
                    'index': i,
                    'formula_str': str(formula),
                    'depth': formula.depth,
                    'length': formula_length,
                    'is_satisfiable': is_sat,
                    'time': end_time - start_time,
                    'success': True,
                    'structure': structure
                })
            except Exception as e:
                end_time = time.time()
                results.append({
                    'index': i,
                    'formula_str': str(formula),
                    'depth': formula.depth,
                    'length': self.get_formula_length(formula),
                    'is_satisfiable': None,
                    'time': end_time - start_time,
                    'success': False,
                    'error': str(e),
                    'structure': None
                })
        
        total_end_time = time.time()
        
        # 添加总体统计
        total_time = total_end_time - total_start_time
        success_count = sum(1 for r in results if r['success'])
        sat_count = sum(1 for r in results if r['success'] and r['is_satisfiable'])
        
        return {
            'results': results,
            'summary': {
                'total_formulas': len(formulas),
                'success_count': success_count,
                'sat_count': sat_count,
                'total_time': total_time,
                'average_time': total_time / len(formulas) if formulas else 0,
                'success_rate': success_count / len(formulas) if formulas else 0,
                'sat_rate': sat_count / success_count if success_count > 0 else 0
            }
        }
    
    def analyze_formula_structure(self, formula: 'AEDNFAECNFPair') -> Dict:
        """分析公式结构"""
        structure = {
            'depth': formula.depth,
            'has_knowledge': False,
            'has_and': False,
            'has_or': False,
            'has_not': False,
            'agent_count': 0,
            'variable_count': 0
        }
        
        # 分析AEDNF项
        for term in formula.aednf.terms:
            # 检查知识文字
            for lit in term.positive_literals + term.negative_literals:
                structure['has_knowledge'] = True
                structure['agent_count'] = max(structure['agent_count'], 
                                             len(set(lit.agent for lit in term.positive_literals + term.negative_literals)))
        
        # 检查逻辑操作
        if '∧' in str(formula):
            structure['has_and'] = True
        if '∨' in str(formula):
            structure['has_or'] = True
        if '¬' in str(formula):
            structure['has_not'] = True
        
        # 统计变量
        for var in self.variables:
            if var in str(formula):
                structure['variable_count'] += 1
        
        return structure
    
    def _estimate_formula_length(self, depth: int) -> int:
        """估算公式长度"""
        if depth == 0:
            return 350  # 基本变量的字符串表示长度约为350字符
        
        # 根据深度估算长度
        # 每个操作大约增加200-300个字符
        base_length = 350
        for i in range(depth):
            base_length = base_length * 1.5 + 250  # 更合理的增长
        return int(base_length)
    
    def get_formula_length(self, formula: 'AEDNFAECNFPair') -> int:
        """获取公式的实际长度"""
        formula_str = str(formula)
        # 计算非空白字符的数量
        return len([c for c in formula_str if not c.isspace()])
    
    def save_results(self, results: Dict, filename: str):
        """保存结果到文件"""
        # 添加时间戳
        results['timestamp'] = datetime.now().isoformat()
        results['generator_config'] = {
            'max_depth': self.max_depth,
            'max_agents': self.max_agents,
            'max_vars': self.max_vars,
            'target_length': self.target_length
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    
    def print_summary(self, results: Dict):
        """打印结果摘要"""
        summary = results['summary']
        
        print(f"\n{'='*60}")
        print(f"公式生成和测试结果摘要")
        print(f"{'='*60}")
        print(f"总公式数: {summary['total_formulas']}")
        print(f"成功测试: {summary['success_count']}")
        print(f"可满足公式: {summary['sat_count']}")
        print(f"总时间: {summary['total_time']:.4f}秒")
        print(f"平均时间: {summary['average_time']:.4f}秒")
        print(f"成功率: {summary['success_rate']:.2%}")
        print(f"可满足率: {summary['sat_rate']:.2%}")
        
        # 长度统计
        if any('length' in result for result in results['results']):
            lengths = [result['length'] for result in results['results'] if 'length' in result]
            if lengths:
                print(f"平均长度: {sum(lengths)/len(lengths):.1f}字符")
                print(f"最小长度: {min(lengths)}字符")
                print(f"最大长度: {max(lengths)}字符")
        
        # 按深度统计
        depth_stats = {}
        for result in results['results']:
            if result['success']:
                depth = result['depth']
                if depth not in depth_stats:
                    depth_stats[depth] = {'count': 0, 'sat_count': 0, 'total_time': 0, 'avg_length': 0}
                depth_stats[depth]['count'] += 1
                depth_stats[depth]['total_time'] += result['time']
                if 'length' in result:
                    depth_stats[depth]['avg_length'] += result['length']
                if result['is_satisfiable']:
                    depth_stats[depth]['sat_count'] += 1
        
        if depth_stats:
            print(f"\n按深度统计:")
            for depth in sorted(depth_stats.keys()):
                stats = depth_stats[depth]
                avg_time = stats['total_time'] / stats['count']
                sat_rate = stats['sat_count'] / stats['count']
                avg_length = stats['avg_length'] / stats['count'] if stats['count'] > 0 else 0
                print(f"  深度 {depth}: {stats['count']}个公式, "
                      f"可满足率 {sat_rate:.2%}, 平均时间 {avg_time:.4f}秒, 平均长度 {avg_length:.1f}字符")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='认知逻辑公式生成器')
    parser.add_argument('--count', type=int, default=100, help='生成公式数量')
    parser.add_argument('--max-depth', type=int, default=3, help='最大层数')
    parser.add_argument('--max-agents', type=int, default=3, help='代理数量')
    parser.add_argument('--max-vars', type=int, default=10, help='变量数量')
    parser.add_argument('--target-depth', type=int, help='目标深度')
    parser.add_argument('--target-length', type=int, help='目标长度（字符数）')
    parser.add_argument('--min-depth', type=int, default=0, help='最小深度')
    parser.add_argument('--no-knowledge', action='store_true', help='不使用知识算子')
    parser.add_argument('--no-and', action='store_true', help='不使用与操作')
    parser.add_argument('--no-or', action='store_true', help='不使用或操作')
    parser.add_argument('--no-not', action='store_true', help='不使用非操作')
    parser.add_argument('--seed', type=int, help='随机种子')
    parser.add_argument('--output', type=str, default='formula_results.json', help='输出文件名')
    parser.add_argument('--verbose', action='store_true', help='详细输出')
    
    args = parser.parse_args()
    
    print(f"=== 认知逻辑公式生成器 ===")
    print(f"参数配置:")
    print(f"  公式数量: {args.count}")
    print(f"  最大层数: {args.max_depth}")
    print(f"  代理数量: {args.max_agents}")
    print(f"  变量数量: {args.max_vars}")
    print(f"  目标深度: {args.target_depth}")
    print(f"  目标长度: {args.target_length}")
    print(f"  最小深度: {args.min_depth}")
    print(f"  使用知识算子: {not args.no_knowledge}")
    print(f"  使用与操作: {not args.no_and}")
    print(f"  使用或操作: {not args.no_or}")
    print(f"  使用非操作: {not args.no_not}")
    print(f"  随机种子: {args.seed}")
    print(f"  输出文件: {args.output}")
    
    # 创建生成器
    generator = AdvancedFormulaGenerator(
        max_depth=args.max_depth,
        max_agents=args.max_agents,
        max_vars=args.max_vars,
        target_length=args.target_length,
        seed=args.seed
    )
    
    # 生成公式
    print(f"\n生成 {args.count} 个公式...")
    start_time = time.time()
    formulas = generator.generate_formulas_batch(
        count=args.count,
        target_depth=args.target_depth,
        use_knowledge=not args.no_knowledge,
        use_and=not args.no_and,
        use_or=not args.no_or,
        use_not=not args.no_not,
        min_depth=args.min_depth
    )
    generation_time = time.time() - start_time
    print(f"公式生成完成，耗时: {generation_time:.4f}秒")
    
    # 测试公式
    print(f"测试公式的可满足性...")
    results = generator.test_formulas_detailed(formulas)
    
    # 打印摘要
    generator.print_summary(results)
    
    # 详细输出
    if args.verbose:
        print(f"\n详细结果:")
        for result in results['results'][:10]:  # 只显示前10个
            print(f"公式 {result['index']}:")
            print(f"  深度: {result['depth']}")
            print(f"  长度: {result.get('length', 'N/A')}字符")
            print(f"  可满足: {result['is_satisfiable']}")
            print(f"  时间: {result['time']:.4f}秒")
            print(f"  成功: {result['success']}")
            if result['structure']:
                print(f"  结构: {result['structure']}")
            if not result['success']:
                print(f"  错误: {result['error']}")
            print()
    
    # 保存结果
    generator.save_results(results, args.output)
    print(f"结果已保存到: {args.output}")


if __name__ == "__main__":
    main()
