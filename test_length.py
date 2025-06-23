#!/usr/bin/env python3
"""
Test script to verify length limits are working correctly
"""

from formulas import EpistemicFormulaGenerator, print_formula_stats
from config import EpistemicConfig

def test_length_limits():
    print("Testing Length Limits")
    print("=" * 50)
    
    # Test with very strict length limits
    config = EpistemicConfig(
        num_agents=3,
        max_depth=3,
        max_length=5,  # Very strict length limit
        num_props=5,
        prop_weight=0.3,
        neg_weight=0.3,
        binary_weight=0.2,
        knowledge_weight=0.2
    )
    
    generator = EpistemicFormulaGenerator(config)
    
    print(f"Config: max_length={config.max_length}, max_depth={config.max_depth}")
    print()
    
    for i in range(10):
        formula = generator.generate_formula()
        stats = print_formula_stats(formula)
        
        print(f"Formula {i+1}:")
        print(f"  Formula: {formula}")
        print(f"  Stats: depth={stats['max_depth']}, operators={stats['total_operators']}")
        print(f"  Components: props={stats['propositions']}, negs={stats['negations']}, binary={stats['binary_ops']}, knowledge={stats['knowledge_ops']}")
        
        # Check if length limit is respected
        if stats['total_operators'] > config.max_length:
            print(f"  [ERROR] Operators ({stats['total_operators']}) exceed max_length ({config.max_length})!")
        else:
            print(f"  [OK] Operators ({stats['total_operators']}) within limit ({config.max_length})")
        print()

if __name__ == "__main__":
    test_length_limits() 