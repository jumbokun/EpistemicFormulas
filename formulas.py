import random
import json
import sys
import argparse
from datetime import datetime
from typing import List, Set, Union
from config import EpistemicConfig, DEFAULT_CONFIG

class EpistemicFormula:
    """Base class for epistemic formulas"""
    pass

class Proposition(EpistemicFormula):
    def __init__(self, name: str):
        self.name = name
    
    def __str__(self):
        return self.name

class Negation(EpistemicFormula):
    def __init__(self, formula: EpistemicFormula):
        self.formula = formula
    
    def __str__(self):
        return f"¬{self.formula}"

class BinaryConnective(EpistemicFormula):
    def __init__(self, left: EpistemicFormula, right: EpistemicFormula, operator: str):
        self.left = left
        self.right = right
        self.operator = operator
    
    def __str__(self):
        return f"({self.left} {self.operator} {self.right})"

class Knowledge(EpistemicFormula):
    def __init__(self, agent: int, formula: EpistemicFormula):
        self.agent = agent
        self.formula = formula
    
    def __str__(self):
        return f"K_{self.agent}({self.formula})"

class EpistemicFormulaGenerator:
    def __init__(self, config: EpistemicConfig):
        self.config = config
        self.agents = list(range(1, config.num_agents + 1))
        self.propositions = [chr(ord('p') + i) for i in range(config.num_props)]
        self.binary_ops = ['∧', '∨', '→']
        
    def generate_formula(self, current_depth: int = 0, used_operators: int = 0) -> EpistemicFormula:
        """
        Recursively generate an epistemic formula
        
        Args:
            current_depth: Current nesting depth
            used_operators: Number of operators already used
        
        Returns:
            Generated epistemic formula
        """
        # Base cases - return proposition if we're at limits
        if current_depth >= self.config.max_depth or used_operators >= self.config.max_length:
            return self._generate_proposition()
        
        # Choose formula type based on weights
        formula_types = []
        weights = []
        
        # Always allow propositions
        formula_types.append('prop')
        weights.append(self.config.prop_weight)
        
        # Add other operators only if we have room
        if used_operators < self.config.max_length:
            # Negation (uses 1 operator)
            formula_types.append('neg')
            weights.append(self.config.neg_weight)
            
            # Binary connectives (uses 1 operator)
            formula_types.append('binary')
            weights.append(self.config.binary_weight)
            
            # Knowledge operators (uses 1 operator)
            if current_depth + 1 < self.config.max_depth:
                formula_types.append('knowledge')
                weights.append(self.config.knowledge_weight)
        
        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        # Choose formula type
        formula_type = random.choices(formula_types, weights=weights)[0]
        
        return self._generate_by_type(formula_type, current_depth, used_operators)
    
    def _generate_by_type(self, formula_type: str, depth: int, used_operators: int) -> EpistemicFormula:
        """Generate formula of specific type"""
        
        if formula_type == 'prop':
            return self._generate_proposition()
        
        elif formula_type == 'neg':
            # 否定使用1个操作符
            subformula = self.generate_formula(depth, used_operators + 1)
            return Negation(subformula)
        
        elif formula_type == 'binary':
            # 二元连接词使用1个操作符，剩余操作符平均分配
            remaining_operators = self.config.max_length - (used_operators + 1)
            if remaining_operators <= 0:
                return self._generate_proposition()
            
            # 平均分配剩余操作符给左右子公式
            left_operators = used_operators + 1 + (remaining_operators // 2)
            right_operators = used_operators + 1 + (remaining_operators - remaining_operators // 2)
            
            operator = random.choice(self.binary_ops)
            left = self.generate_formula(depth, left_operators)
            right = self.generate_formula(depth, right_operators)
            return BinaryConnective(left, right, operator)
        
        elif formula_type == 'knowledge':
            # 知识算子使用1个操作符
            agent = random.choice(self.agents)
            subformula = self.generate_formula(depth + 1, used_operators + 1)
            return Knowledge(agent, subformula)
        
        else:
            return self._generate_proposition()
    
    def _generate_proposition(self) -> Proposition:
        """Generate a random propositional variable"""
        return Proposition(random.choice(self.propositions))
    
    def generate_multiple(self, count: int) -> List[EpistemicFormula]:
        """Generate multiple epistemic formulas with deduplication and progress feedback"""
        formulas = []
        attempts = 0
        max_attempts = count * 50  # 增加尝试次数
        seen = set()  # 使用set进行更高效的去重
        
        print(f"Generating {count} unique formulas...")
        
        while len(formulas) < count and attempts < max_attempts:
            formula = self.generate_formula()
            formula_str = str(formula)
            
            if formula_str not in seen:
                formulas.append(formula)
                seen.add(formula_str)
            
            attempts += 1
            
            # 每100次尝试显示一次进度
            if attempts % 100 == 0:
                print(f"  [Progress] {len(formulas)}/{count} formulas generated, {attempts} attempts...")
        
        if len(formulas) < count:
            print(f"[Warning] Only generated {len(formulas)} unique formulas after {attempts} attempts.")
            print(f"[Info] Consider reducing complexity or increasing max_attempts.")
        else:
            print(f"[Success] Generated {len(formulas)} unique formulas in {attempts} attempts.")
        
        return formulas

def print_formula_stats(formula: EpistemicFormula, depth: int = 0) -> dict:
    """Calculate statistics about a formula"""
    stats = {
        'max_depth': depth,
        'total_operators': 0,
        'propositions': 0,
        'knowledge_ops': 0,
        'negations': 0,
        'binary_ops': 0
    }
    
    if isinstance(formula, Proposition):
        stats['propositions'] = 1
        stats['max_depth'] = depth  # 命题的深度就是当前深度
    
    elif isinstance(formula, Negation):
        stats['negations'] = 1
        stats['total_operators'] = 1
        sub_stats = print_formula_stats(formula.formula, depth)  # 否定不增加知识深度
        for key in stats:
            if key != 'negations' and key != 'total_operators':
                stats[key] += sub_stats[key]
            elif key == 'total_operators':
                stats[key] += sub_stats[key]
        stats['max_depth'] = sub_stats['max_depth']  # 取子公式的最大深度
    
    elif isinstance(formula, BinaryConnective):
        stats['binary_ops'] = 1
        stats['total_operators'] = 1
        left_stats = print_formula_stats(formula.left, depth)    # 二元连接词不增加知识深度
        right_stats = print_formula_stats(formula.right, depth)  # 二元连接词不增加知识深度
        for key in stats:
            if key != 'binary_ops' and key != 'total_operators':
                stats[key] += left_stats[key] + right_stats[key]
            elif key == 'total_operators':
                stats[key] += left_stats[key] + right_stats[key]
        stats['max_depth'] = max(left_stats['max_depth'], right_stats['max_depth'])  # 取左右子公式的最大深度
    
    elif isinstance(formula, Knowledge):
        stats['knowledge_ops'] = 1
        stats['total_operators'] = 1
        sub_stats = print_formula_stats(formula.formula, depth + 1)  # K算子增加知识深度
        for key in stats:
            if key != 'knowledge_ops' and key != 'total_operators':
                stats[key] += sub_stats[key]
            elif key == 'total_operators':
                stats[key] += sub_stats[key]
        stats['max_depth'] = sub_stats['max_depth']  # 取子公式的最大深度
    
    return stats

# Example usage and testing
def main():
    # Use default configuration
    config = DEFAULT_CONFIG
    
    # Generate epistemic formula generator
    generator = EpistemicFormulaGenerator(config)
    
    print("Epistemic Formula Generator")
    print("=" * 50)
    print(f"Configuration:")
    print(f"  Agents: {config.num_agents}")
    print(f"  Max Depth: {config.max_depth}")
    print(f"  Max Length: {config.max_length}")
    print(f"  Propositions: {config.num_props}")
    print()
    
    # Generate and display formulas
    print("Generated Epistemic Formulas:")
    print("-" * 30)
    
    for i in range(5):
        formula = generator.generate_formula()
        stats = print_formula_stats(formula)
        
        print(f"{i+1}. {formula}")
        print(f"   Stats: depth={stats['max_depth']}, operators={stats['total_operators']}")
        print()

def parse_arguments():
    """Parse command line arguments in format: key=value"""
    args = {}
    for arg in sys.argv[1:]:
        if '=' in arg:
            key, value = arg.split('=', 1)
            args[key] = value
    return args

def generate_formulas_with_args():
    """Generate formulas with command line arguments"""
    args = parse_arguments()
    
    # Start with default configuration
    config = DEFAULT_CONFIG
    
    # Override with command line arguments
    if 'depth' in args:
        config.max_depth = int(args['depth'])
    if 'length' in args:
        config.max_length = int(args['length'])
    if 'agents' in args:
        config.num_agents = int(args['agents'])
    if 'props' in args:
        config.num_props = int(args['props'])
    if 'number' in args:
        num_formulas = int(args['number'])
    else:
        num_formulas = 10
    
    # Generate formulas
    print(f"Starting generation with configuration:")
    print(f"  Depth: {config.max_depth}, Length: {config.max_length}")
    print(f"  Agents: {config.num_agents}, Props: {config.num_props}")
    print(f"  Weights: prop={config.prop_weight}, neg={config.neg_weight}, binary={config.binary_weight}, knowledge={config.knowledge_weight}")
    print()
    
    generator = EpistemicFormulaGenerator(config)
    formulas = generator.generate_multiple(num_formulas)
    
    # Calculate statistics for each formula
    formula_data = []
    for i, formula in enumerate(formulas):
        stats = print_formula_stats(formula)
        formula_data.append({
            'id': i + 1,
            'formula': str(formula),
            'stats': stats
        })
    
    # Create output data
    output_data = {
        'generation_time': datetime.now().isoformat(),
        'parameters': {
            'num_agents': config.num_agents,
            'max_depth': config.max_depth,
            'max_length': config.max_length,
            'num_props': config.num_props,
            'prop_weight': config.prop_weight,
            'neg_weight': config.neg_weight,
            'binary_weight': config.binary_weight,
            'knowledge_weight': config.knowledge_weight,
            'num_formulas': num_formulas
        },
        'formulas': formula_data
    }
    
    # Save to JSON file
    filename = f"epistemic_formulas_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    # Print summary
    print(f"Generated {num_formulas} epistemic formulas")
    print(f"Configuration: depth={config.max_depth}, length={config.max_length}, agents={config.num_agents}")
    print(f"Results saved to: {filename}")
    
    # Print first few formulas as preview
    print("\nPreview of generated formulas:")
    print("-" * 40)
    for i in range(min(3, len(formulas))):
        formula = formulas[i]
        stats = print_formula_stats(formula)
        print(f"{i+1}. {formula}")
        print(f"   Stats: depth={stats['max_depth']}, operators={stats['total_operators']}")
        print()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        generate_formulas_with_args()
    else:
        main()