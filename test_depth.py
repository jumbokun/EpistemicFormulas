#!/usr/bin/env python3
"""
Test script to generate formulas with high depth
"""

from formulas import EpistemicFormulaGenerator, print_formula_stats
from config import EpistemicConfig

def test_depth_generation():
    print("Testing Depth Generation")
    print("=" * 50)
    
    # Configuration optimized for depth
    config = EpistemicConfig(
        num_agents=5,
        max_depth=10,
        max_length=30,  # Moderate length to allow depth
        num_props=8,
        prop_weight=0.05,      # Very low proposition weight
        neg_weight=0.15,       # Low negation weight
        binary_weight=0.1,     # Low binary connective weight
        knowledge_weight=0.7   # High knowledge operator weight
    )
    
    generator = EpistemicFormulaGenerator(config)
    
    print(f"Config: max_depth={config.max_depth}, max_length={config.max_length}")
    print(f"Weights: prop={config.prop_weight}, neg={config.neg_weight}, binary={config.binary_weight}, knowledge={config.knowledge_weight}")
    print()
    
    max_depth_found = 0
    deep_formulas = []
    
    for i in range(20):
        formula = generator.generate_formula()
        stats = print_formula_stats(formula)
        
        print(f"Formula {i+1}:")
        print(f"  Formula: {formula}")
        print(f"  Stats: depth={stats['max_depth']}, operators={stats['total_operators']}")
        print(f"  Components: props={stats['propositions']}, negs={stats['negations']}, binary={stats['binary_ops']}, knowledge={stats['knowledge_ops']}")
        
        if stats['max_depth'] > max_depth_found:
            max_depth_found = stats['max_depth']
            deep_formulas.append((i+1, formula, stats))
        
        print()
    
    print(f"Maximum depth found: {max_depth_found}")
    print(f"Deepest formulas:")
    for i, formula, stats in deep_formulas[-3:]:  # Show last 3 deepest
        print(f"  Formula {i}: depth={stats['max_depth']}, operators={stats['total_operators']}")
        print(f"    {formula}")

if __name__ == "__main__":
    test_depth_generation() 