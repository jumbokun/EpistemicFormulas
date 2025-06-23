#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Import from the main file
with open('AEDNF-AECNF-P.py', 'r', encoding='utf-8') as f:
    exec(f.read())

def inspect_clause(clause, clause_type):
    print(f"    {clause_type} Clause:")
    print(f"      Objective: {clause.objective}")
    print(f"      Positive Knowledge: {clause.positive_knowledge}")
    print(f"      Negative Knowledge: {clause.negative_knowledge}")

def test_simple_cases():
    parser = FormulaParser()
    converter = AEDNFAECNFConverter([1, 2, 3])
    
    test_cases = [
        "¬K_1(p)",
        "K_1(¬w)",
        "(¬K_2(p) ∨ K_1(¬w))"
    ]
    
    for test_case in test_cases:
        print(f"\nTesting: {test_case}")
        try:
            formula = parser.parse_formula(test_case)
            result = converter.convert_formula(formula)
            print(f"  AEDNF: {result.aednf}")
            print(f"  AECNF: {result.aecnf}")
            
            # Inspect AEDNF clauses
            print(f"  AEDNF has {len(result.aednf.clauses)} clauses:")
            for i, clause in enumerate(result.aednf.clauses):
                inspect_clause(clause, f"AEDNF[{i}]")
                
            # Inspect AECNF clauses  
            print(f"  AECNF has {len(result.aecnf.clauses)} clauses:")
            for i, clause in enumerate(result.aecnf.clauses):
                inspect_clause(clause, f"AECNF[{i}]")
                
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_simple_cases() 