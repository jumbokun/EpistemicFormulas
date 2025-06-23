#!/usr/bin/env python3
"""
Test script for complex formula generation
"""

from formulas import EpistemicFormulaGenerator
from config import COMPLEX_CONFIG, VERY_COMPLEX_CONFIG, DEFAULT_CONFIG

def test_config(config, name, num_formulas=3):
    print(f"\n{'='*60}")
    print(f"Testing {name}")
    print(f"{'='*60}")
    print(f"Config: depth={config.max_depth}, length={config.max_length}")
    print(f"Weights: prop={config.prop_weight}, neg={config.neg_weight}, binary={config.binary_weight}, knowledge={config.knowledge_weight}")
    
    generator = EpistemicFormulaGenerator(config)
    formulas = generator.generate_multiple(num_formulas)
    
    print(f"\nGenerated {len(formulas)} formulas:")
    print("-" * 40)
    for i, formula in enumerate(formulas, 1):
        from formulas import print_formula_stats
        stats = print_formula_stats(formula)
        print(f"{i}. {formula}")
        print(f"   Stats: depth={stats['max_depth']}, operators={stats['total_operators']}")
        print(f"   Components: props={stats['propositions']}, negs={stats['negations']}, binary={stats['binary_ops']}, knowledge={stats['knowledge_ops']}")
        print()

if __name__ == "__main__":
    print("Epistemic Formula Generator - Configuration Test")
    print("This will test different configurations with progress feedback.")
    
    # Test different configurations with fewer formulas
    test_config(DEFAULT_CONFIG, "Default Config", 3)
    test_config(COMPLEX_CONFIG, "Complex Config", 3)
    test_config(VERY_COMPLEX_CONFIG, "Very Complex Config", 3)
    
    print("\nTest completed!") 