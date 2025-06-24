import time
import random
import json
from typing import List, Set, Union, Dict, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod

# Reuse the previous EpistemicFormula class definitions
class EpistemicFormula:
    """Base class for epistemic formulas"""
    pass

class Proposition(EpistemicFormula):
    def __init__(self, name: str):
        self.name = name
    
    def __str__(self):
        return self.name
    
    def __eq__(self, other):
        return isinstance(other, Proposition) and self.name == other.name
    
    def __hash__(self):
        return hash(('prop', self.name))

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

class CommonKnowledge(EpistemicFormula):
    def __init__(self, agents: Set[int], formula: EpistemicFormula):
        self.agents = agents
        self.formula = formula
    
    def __str__(self):
        agent_list = sorted(list(self.agents))
        return f"C_{{{','.join(map(str, agent_list))}}}({self.formula})"

# AEDNF/AECNF representation classes
class ObjectiveFormula(EpistemicFormula):
    """Represents an a-objective formula"""
    def __init__(self, formula: EpistemicFormula):
        self.formula = formula
    
    def __str__(self):
        return str(self.formula)
    
    def __repr__(self):
        return f"ObjectiveFormula({self.formula})"

class AEDNFClause:
    """Single clause in AEDNF: α ∧ ⋀_a(K_a φ_a ∧ ⋀_j ¬K_a ψ_{a,j})"""
    def __init__(self, 
                 objective: ObjectiveFormula = None,
                 positive_knowledge: Dict[int, ObjectiveFormula] = None,
                 negative_knowledge: Dict[int, List[ObjectiveFormula]] = None):
        self.objective = objective  # α part (can be None if no propositional part)
        self.positive_knowledge = positive_knowledge or {}  # K_a φ_a for each agent a
        self.negative_knowledge = negative_knowledge or {}  # ¬K_a ψ_{a,j} for each agent a
    
    def __str__(self):
        parts = []
        
        # Add objective part if present
        if self.objective:
            parts.append(str(self.objective))
        
        # Add positive knowledge
        for agent, pos_formula in self.positive_knowledge.items():
            parts.append(f"K_{agent}({pos_formula})")
        
        # Add negative knowledge
        for agent, neg_formulas in self.negative_knowledge.items():
            for neg_formula in neg_formulas:
                parts.append(f"¬K_{agent}({neg_formula})")
        
        if not parts:
            return "⊤"  # Empty clause represents ⊤
        return " ∧ ".join(parts)

class AECNFClause:
    """Single clause in AECNF: α ∨ ⋁_a(¬K_a φ_a ∨ ⋁_j K_a ψ_{a,j})"""
    def __init__(self,
                 objective: ObjectiveFormula = None,
                 negative_knowledge: Dict[int, ObjectiveFormula] = None,
                 positive_knowledge: Dict[int, List[ObjectiveFormula]] = None):
        self.objective = objective  # α part (can be None)
        self.negative_knowledge = negative_knowledge or {}  # ¬K_a φ_a for each agent a
        self.positive_knowledge = positive_knowledge or {}  # K_a ψ_{a,j} for each agent a
    
    def __str__(self):
        parts = []
        
        # Add objective part if present
        if self.objective:
            parts.append(str(self.objective))
        
        # Add negative knowledge
        for agent, neg_formula in self.negative_knowledge.items():
            parts.append(f"¬K_{agent}({neg_formula})")
        
        # Add positive knowledge
        for agent, pos_formulas in self.positive_knowledge.items():
            for pos_formula in pos_formulas:
                parts.append(f"K_{agent}({pos_formula})")
        
        if not parts:
            return "⊥"  # Empty clause represents ⊥
        return " ∨ ".join(parts)

class AEDNF:
    """AEDNF representation: disjunction of AEDNF clauses"""
    def __init__(self, clauses: List[AEDNFClause], modal_depth: int = 1):
        self.clauses = clauses
        self.modal_depth = modal_depth
    
    def __str__(self):
        if not self.clauses:
            return "⊥"
        return " ∨ ".join(f"({clause})" for clause in self.clauses)

class AECNF:
    """AECNF representation: conjunction of AECNF clauses"""
    def __init__(self, clauses: List[AECNFClause], modal_depth: int = 1):
        self.clauses = clauses
        self.modal_depth = modal_depth
    
    def __str__(self):
        if not self.clauses:
            return "⊤"
        return " ∧ ".join(f"({clause})" for clause in self.clauses)

@dataclass
class ConversionResult:
    """Conversion result"""
    aednf: AEDNF
    aecnf: AECNF
    conversion_time: float
    original_formula: EpistemicFormula
    modal_depth: int
    
class AEDNFAECNFConverter:
    """Converter for transforming epistemic formulas to AEDNF/AECNF pairs"""
    
    def __init__(self, agents: List[int]):
        self.agents = agents
        
    def convert_formula(self, formula: EpistemicFormula) -> ConversionResult:
        """
        Convert epistemic formula to AEDNF/AECNF pair
        """
        start_time = time.time()
        
        # 1. Compute modal depth
        modal_depth = self._compute_modal_depth(formula)
        
        # 2. Check alternating constraint
        if not self._check_alternating_constraint(formula):
            # If alternating constraint is not satisfied, preprocess first
            formula = self._preprocess_alternating(formula)
        
        # 3. Convert to AEDNF (Disjunctive Normal Form)
        aednf = self._convert_to_aednf(formula, modal_depth)
        
        # 4. Convert AEDNF to AECNF (Conjunctive Normal Form)
        # Both should represent the same original formula
        aecnf = self._convert_dnf_to_cnf(aednf, formula, modal_depth)
        
        # 5. Simplify normal forms
        aednf = self._simplify_aednf(aednf)
        aecnf = self._simplify_aecnf(aecnf)
        
        end_time = time.time()
        conversion_time = end_time - start_time
        
        return ConversionResult(
            aednf=aednf,
            aecnf=aecnf,
            conversion_time=conversion_time,
            original_formula=formula,
            modal_depth=modal_depth
        )
    
    def _compute_modal_depth(self, formula: EpistemicFormula) -> int:
        """Compute modal depth of formula"""
        if isinstance(formula, Proposition):
            return 0
        elif isinstance(formula, Negation):
            return self._compute_modal_depth(formula.formula)
        elif isinstance(formula, BinaryConnective):
            return max(
                self._compute_modal_depth(formula.left),
                self._compute_modal_depth(formula.right)
            )
        elif isinstance(formula, Knowledge):
            return 1 + self._compute_modal_depth(formula.formula)
        elif isinstance(formula, CommonKnowledge):
            return 1 + self._compute_modal_depth(formula.formula)
        else:
            return 0
    
    def _check_alternating_constraint(self, formula: EpistemicFormula) -> bool:
        """Check if alternating constraint is satisfied"""
        return self._check_alternating_for_agent(formula, None)
    
    def _check_alternating_for_agent(self, formula: EpistemicFormula, last_agent: int) -> bool:
        """Recursively check alternating constraint"""
        if isinstance(formula, Proposition):
            return True
        elif isinstance(formula, Negation):
            return self._check_alternating_for_agent(formula.formula, last_agent)
        elif isinstance(formula, BinaryConnective):
            return (self._check_alternating_for_agent(formula.left, last_agent) and
                    self._check_alternating_for_agent(formula.right, last_agent))
        elif isinstance(formula, Knowledge):
            # Check if alternating constraint is violated
            if last_agent == formula.agent:
                return False  # K_a K_a φ is not allowed
            return self._check_alternating_for_agent(formula.formula, formula.agent)
        elif isinstance(formula, CommonKnowledge):
            # Simplified handling: treat common knowledge as single agent
            return self._check_alternating_for_agent(formula.formula, -1)
        else:
            return True
    
    def _preprocess_alternating(self, formula: EpistemicFormula) -> EpistemicFormula:
        """Preprocess to satisfy alternating constraint (simplified implementation)"""
        # More complex preprocessing logic can be implemented here
        # For now, return original formula
        return formula
    
    def _convert_to_aednf(self, formula: EpistemicFormula, modal_depth: int) -> AEDNF:
        """Convert to AEDNF"""
        if modal_depth == 0:
            # Pure propositional formula
            objective = ObjectiveFormula(formula)
            clause = AEDNFClause(objective, {}, {})
            return AEDNF([clause], 0)
        
        # Recursively build AEDNF
        clauses = self._build_aednf_clauses(formula)
        return AEDNF(clauses, modal_depth)
    
    def _build_aednf_clauses(self, formula: EpistemicFormula) -> List[AEDNFClause]:
        """Build AEDNF clauses"""
        if isinstance(formula, Proposition):
            objective = ObjectiveFormula(formula)
            return [AEDNFClause(objective, {}, {})]
        
        elif isinstance(formula, Negation):
            inner_formula = formula.formula
            if isinstance(inner_formula, Knowledge):
                # ¬K_a φ -> create clause with negative knowledge
                agent = inner_formula.agent
                inner_obj = ObjectiveFormula(inner_formula.formula)
                neg_knowledge = {agent: [inner_obj]}
                return [AEDNFClause(None, {}, neg_knowledge)]
            else:
                # ¬φ where φ is not knowledge -> objective formula
                objective = ObjectiveFormula(formula)
                return [AEDNFClause(objective, {}, {})]
        
        elif isinstance(formula, BinaryConnective):
            if formula.operator == '∧':
                # (φ₁ ∨ φ₂ ∨ ...) ∧ (ψ₁ ∨ ψ₂ ∨ ...) = ∨ᵢⱼ (φᵢ ∧ ψⱼ)
                left_clauses = self._build_aednf_clauses(formula.left)
                right_clauses = self._build_aednf_clauses(formula.right)
                result = []
                for left_clause in left_clauses:
                    for right_clause in right_clauses:
                        merged_clause = self._merge_aednf_clauses(left_clause, right_clause)
                        result.append(merged_clause)
                return result
            
            elif formula.operator == '∨':
                # φ ∨ ψ = directly combine clauses
                left_clauses = self._build_aednf_clauses(formula.left)
                right_clauses = self._build_aednf_clauses(formula.right)
                return left_clauses + right_clauses
            
            else:  # →, ↔ etc.
                # Convert to basic connectives
                if formula.operator == '→':
                    equiv = BinaryConnective(
                        Negation(formula.left), formula.right, '∨'
                    )
                elif formula.operator == '↔':
                    left_to_right = BinaryConnective(
                        Negation(formula.left), formula.right, '∨'
                    )
                    right_to_left = BinaryConnective(
                        Negation(formula.right), formula.left, '∨'
                    )
                    equiv = BinaryConnective(left_to_right, right_to_left, '∧')
                else:
                    equiv = formula
                
                return self._build_aednf_clauses(equiv)
        
        elif isinstance(formula, Knowledge):
            # K_a φ -> create clause with positive knowledge
            agent = formula.agent
            inner_formula = formula.formula
            
            if self._is_objective_for_agent(inner_formula, agent):
                # φ is a-objective, add to positive knowledge
                pos_knowledge = {agent: ObjectiveFormula(inner_formula)}
                return [AEDNFClause(None, pos_knowledge, {})]
            else:
                # φ is not a-objective, treat as objective formula for now
                # This is a simplification - in a full implementation we'd need
                # to apply more complex transformations
                objective = ObjectiveFormula(formula)
                return [AEDNFClause(objective, {}, {})]
        
        else:
            # Simplified handling for other cases
            objective = ObjectiveFormula(formula)
            return [AEDNFClause(objective, {}, {})]
    
    def _merge_aednf_clauses(self, clause1: AEDNFClause, clause2: AEDNFClause) -> AEDNFClause:
        """Merge two AEDNF clauses"""
        # Merge objective parts (conjunction)
        merged_objective = None
        if clause1.objective and clause2.objective:
            merged_objective = ObjectiveFormula(
                BinaryConnective(clause1.objective.formula, clause2.objective.formula, '∧')
            )
        elif clause1.objective:
            merged_objective = clause1.objective
        elif clause2.objective:
            merged_objective = clause2.objective
        # If both are None, merged_objective remains None
        
        # Merge positive knowledge
        merged_pos = clause1.positive_knowledge.copy()
        for agent, formula in clause2.positive_knowledge.items():
            if agent in merged_pos:
                # K_a φ₁ ∧ K_a φ₂ = K_a (φ₁ ∧ φ₂)
                combined = ObjectiveFormula(
                    BinaryConnective(merged_pos[agent].formula, formula.formula, '∧')
                )
                merged_pos[agent] = combined
            else:
                merged_pos[agent] = formula
        
        # Merge negative knowledge
        merged_neg = clause1.negative_knowledge.copy()
        for agent, formulas in clause2.negative_knowledge.items():
            if agent in merged_neg:
                merged_neg[agent].extend(formulas)
            else:
                merged_neg[agent] = formulas[:]
        
        return AEDNFClause(merged_objective, merged_pos, merged_neg)
    
    def _is_objective_for_agent(self, formula: EpistemicFormula, agent: int) -> bool:
        """Check if formula is objective for agent a"""
        if isinstance(formula, Proposition):
            return True
        elif isinstance(formula, Negation):
            return self._is_objective_for_agent(formula.formula, agent)
        elif isinstance(formula, BinaryConnective):
            return (self._is_objective_for_agent(formula.left, agent) and
                    self._is_objective_for_agent(formula.right, agent))
        elif isinstance(formula, Knowledge):
            return formula.agent != agent
        elif isinstance(formula, CommonKnowledge):
            return agent not in formula.agents
        else:
            return True
    
    def _convert_dnf_to_cnf(self, aednf: AEDNF, original_formula: EpistemicFormula, modal_depth: int) -> AECNF:
        """Convert AEDNF (DNF) to AECNF (CNF) - both represent the same formula"""
        
        # For simple cases, directly convert the original formula to CNF
        if modal_depth == 0:
            # Pure propositional formula - convert to CNF
            cnf_clauses = self._build_aecnf_clauses(original_formula)
            return AECNF(cnf_clauses, 0)
        
        # For epistemic formulas, we need to build CNF representation
        cnf_clauses = self._build_aecnf_clauses(original_formula)
        return AECNF(cnf_clauses, modal_depth)
    
    def _build_aecnf_clauses(self, formula: EpistemicFormula) -> List[AECNFClause]:
        """Build AECNF clauses (CNF representation)"""
        if isinstance(formula, Proposition):
            objective = ObjectiveFormula(formula)
            return [AECNFClause(objective, {}, {})]
        
        elif isinstance(formula, Negation):
            inner_formula = formula.formula
            if isinstance(inner_formula, Knowledge):
                # ¬K_a φ -> create clause with negative knowledge
                agent = inner_formula.agent
                inner_obj = ObjectiveFormula(inner_formula.formula)
                neg_knowledge = {agent: inner_obj}
                return [AECNFClause(None, neg_knowledge, {})]
            else:
                # ¬φ where φ is not knowledge -> objective formula
                objective = ObjectiveFormula(formula)
                return [AECNFClause(objective, {}, {})]
        
        elif isinstance(formula, BinaryConnective):
            if formula.operator == '∧':
                # φ ∧ ψ -> combine all clauses from both sides
                left_clauses = self._build_aecnf_clauses(formula.left)
                right_clauses = self._build_aecnf_clauses(formula.right)
                return left_clauses + right_clauses
            
            elif formula.operator == '∨':
                # φ ∨ ψ -> create cross product of clauses (distribution)
                left_clauses = self._build_aecnf_clauses(formula.left)
                right_clauses = self._build_aecnf_clauses(formula.right)
                
                if len(left_clauses) == 1 and len(right_clauses) == 1:
                    # Simple case: merge into one clause
                    merged_clause = self._merge_aecnf_clauses(left_clauses[0], right_clauses[0])
                    return [merged_clause]
                else:
                    # Complex case: for now, create a single clause with disjunction
                    # This is a simplification - full CNF conversion is more complex
                    objective = ObjectiveFormula(formula)
                    return [AECNFClause(objective, {}, {})]
            
            else:  # →, ↔ etc.
                # Convert to basic connectives
                if formula.operator == '→':
                    equiv = BinaryConnective(
                        Negation(formula.left), formula.right, '∨'
                    )
                elif formula.operator == '↔':
                    left_to_right = BinaryConnective(
                        Negation(formula.left), formula.right, '∨'
                    )
                    right_to_left = BinaryConnective(
                        Negation(formula.right), formula.left, '∨'
                    )
                    equiv = BinaryConnective(left_to_right, right_to_left, '∧')
                else:
                    equiv = formula
                
                return self._build_aecnf_clauses(equiv)
        
        elif isinstance(formula, Knowledge):
            # K_a φ -> create clause with positive knowledge
            agent = formula.agent
            inner_formula = formula.formula
            
            if self._is_objective_for_agent(inner_formula, agent):
                # φ is a-objective, add to positive knowledge
                pos_knowledge = {agent: [ObjectiveFormula(inner_formula)]}
                return [AECNFClause(None, {}, pos_knowledge)]
            else:
                # φ is not a-objective, treat as objective formula for now
                objective = ObjectiveFormula(formula)
                return [AECNFClause(objective, {}, {})]
        
        else:
            # Simplified handling for other cases
            objective = ObjectiveFormula(formula)
            return [AECNFClause(objective, {}, {})]
    
    def _merge_aecnf_clauses(self, clause1: AECNFClause, clause2: AECNFClause) -> AECNFClause:
        """Merge two AECNF clauses (disjunction)"""
        # Merge objective parts (disjunction)
        merged_objective = None
        if clause1.objective and clause2.objective:
            merged_objective = ObjectiveFormula(
                BinaryConnective(clause1.objective.formula, clause2.objective.formula, '∨')
            )
        elif clause1.objective:
            merged_objective = clause1.objective  
        elif clause2.objective:
            merged_objective = clause2.objective
        
        # Merge negative knowledge (disjunction)
        merged_neg = clause1.negative_knowledge.copy()
        for agent, formula in clause2.negative_knowledge.items():
            if agent in merged_neg:
                # ¬K_a φ₁ ∨ ¬K_a φ₂ = ¬K_a (φ₁ ∧ φ₂) (by modal logic)
                combined = ObjectiveFormula(
                    BinaryConnective(merged_neg[agent].formula, formula.formula, '∧')
                )
                merged_neg[agent] = combined
            else:
                merged_neg[agent] = formula
        
        # Merge positive knowledge (disjunction)
        merged_pos = clause1.positive_knowledge.copy()
        for agent, formulas in clause2.positive_knowledge.items():
            if agent in merged_pos:
                merged_pos[agent].extend(formulas)
            else:
                merged_pos[agent] = formulas[:]
        
        return AECNFClause(merged_objective, merged_neg, merged_pos)
    
    def _simplify_aednf(self, aednf: AEDNF) -> AEDNF:
        """Simplify AEDNF - remove unsatisfiable terms"""
        simplified_clauses = []
        for clause in aednf.clauses:
            if self._is_satisfiable_aednf_clause(clause):
                simplified_clauses.append(clause)
        return AEDNF(simplified_clauses, aednf.modal_depth)
    
    def _simplify_aecnf(self, aecnf: AECNF) -> AECNF:
        """Simplify AECNF - remove tautological clauses"""
        simplified_clauses = []
        for clause in aecnf.clauses:
            if not self._is_tautological_aecnf_clause(clause):
                simplified_clauses.append(clause)
        return AECNF(simplified_clauses, aecnf.modal_depth)
    
    def _is_satisfiable_aednf_clause(self, clause: AEDNFClause) -> bool:
        """Check if AEDNF clause is satisfiable (simplified version)"""
        # Simplified: assume most clauses are satisfiable
        return True
    
    def _is_tautological_aecnf_clause(self, clause: AECNFClause) -> bool:
        """Check if AECNF clause is tautology (simplified version)"""
        # Simplified: assume few tautological clauses
        return False

# Formula parser for JSON input
class FormulaParser:
    """Parser for converting string formulas to EpistemicFormula objects"""
    
    def __init__(self):
        self.propositions = set()
    
    def parse_formula(self, formula_str: str) -> EpistemicFormula:
        """Parse string formula to EpistemicFormula object"""
        # Remove extra whitespace
        formula_str = formula_str.strip()
        
        # Handle simple propositions first
        if self._is_simple_proposition(formula_str):
            self.propositions.add(formula_str)
            return Proposition(formula_str)
        
        # Handle negation
        if formula_str.startswith('¬'):
            inner_formula = self.parse_formula(formula_str[1:])
            return Negation(inner_formula)
        
        # Handle knowledge operators
        if formula_str.startswith('K_'):
            return self._parse_knowledge(formula_str)
        
        # Handle parentheses
        if formula_str.startswith('(') and formula_str.endswith(')'):
            return self._parse_binary_connective(formula_str[1:-1])
        
        # Try to find binary operators at top level
        main_op_pos = self._find_main_operator(formula_str)
        if main_op_pos != -1:
            return self._parse_binary_connective(formula_str)
        
        # If we get here, treat as proposition
        self.propositions.add(formula_str)
        return Proposition(formula_str)
    
    def _is_simple_proposition(self, formula_str: str) -> bool:
        """Check if string represents a simple proposition"""
        return (len(formula_str) == 1 and formula_str.isalpha()) or \
               (formula_str.isalnum() and len(formula_str) <= 3 and 
                not any(op in formula_str for op in ['K_', '¬', '(', ')', '∧', '∨', '→', '↔']))
    
    def _parse_knowledge(self, formula_str: str) -> Knowledge:
        """Parse knowledge operator K_a(φ)"""
        # Find the agent number
        start = formula_str.find('_') + 1
        end = formula_str.find('(', start)
        agent_str = formula_str[start:end]
        agent = int(agent_str)
        
        # Find the inner formula
        inner_start = end + 1
        inner_end = self._find_matching_paren(formula_str, inner_start - 1)
        inner_formula_str = formula_str[inner_start:inner_end]
        
        inner_formula = self.parse_formula(inner_formula_str)
        return Knowledge(agent, inner_formula)
    
    def _parse_binary_connective(self, formula_str: str) -> EpistemicFormula:
        """Parse binary connective (φ op ψ)"""
        # Find the main operator
        op_pos = self._find_main_operator(formula_str)
        if op_pos == -1:
            # No operator found, try removing outer parentheses
            if formula_str.startswith('(') and formula_str.endswith(')'):
                return self.parse_formula(formula_str[1:-1])
            return self.parse_formula(formula_str)
        
        operator = formula_str[op_pos]
        left_str = formula_str[:op_pos].strip()
        right_str = formula_str[op_pos+1:].strip()
        
        left_formula = self.parse_formula(left_str)
        right_formula = self.parse_formula(right_str)
        
        return BinaryConnective(left_formula, right_formula, operator)
    
    def _find_matching_paren(self, formula_str: str, start: int) -> int:
        """Find matching closing parenthesis"""
        if start >= len(formula_str) or formula_str[start] != '(':
            return len(formula_str)
            
        count = 0
        for i in range(start, len(formula_str)):
            if formula_str[i] == '(':
                count += 1
            elif formula_str[i] == ')':
                count -= 1
                if count == 0:
                    return i
        return len(formula_str) - 1
    
    def _find_main_operator(self, formula_str: str) -> int:
        """Find the main binary operator"""
        operators = ['∨', '∧', '→', '↔']
        paren_count = 0
        
        # Look for operators from right to left to handle precedence
        for i in range(len(formula_str) - 1, -1, -1):
            char = formula_str[i]
            if char == ')':
                paren_count += 1
            elif char == '(':
                paren_count -= 1
            elif char in operators and paren_count == 0:
                return i
        
        return -1

# Conversion benchmark and statistics class
class ConversionBenchmark:
    """Conversion performance benchmark"""
    
    def __init__(self, converter: AEDNFAECNFConverter):
        self.converter = converter
        self.results: List[ConversionResult] = []
    
    def benchmark_formula_set(self, formulas: List[EpistemicFormula]) -> Dict:
        """Benchmark a set of formulas"""
        print(f"Starting conversion of {len(formulas)} formulas...")
        
        total_time = 0
        modal_depth_stats = {}
        
        for i, formula in enumerate(formulas):
            print(f"Converting formula {i+1}/{len(formulas)}: {formula}")
            
            result = self.converter.convert_formula(formula)
            self.results.append(result)
            
            total_time += result.conversion_time
            
            # Statistics for modal depth
            depth = result.modal_depth
            if depth not in modal_depth_stats:
                modal_depth_stats[depth] = []
            modal_depth_stats[depth].append(result.conversion_time)
            
            print(f"  -> AEDNF: {result.aednf}")
            print(f"  -> AECNF: {result.aecnf}")
            print(f"  -> Conversion time: {result.conversion_time:.6f}s")
            print(f"  -> Modal Depth: {result.modal_depth}")
            print()
        
        # Calculate statistics
        avg_time = total_time / len(formulas) if formulas else 0
        max_time = max((r.conversion_time for r in self.results), default=0)
        min_time = min((r.conversion_time for r in self.results), default=0)
        
        stats = {
            'total_formulas': len(formulas),
            'total_time': total_time,
            'avg_time': avg_time,
            'max_time': max_time,
            'min_time': min_time,
            'modal_depth_stats': modal_depth_stats
        }
        
        self.print_statistics(stats)
        return stats
    
    def print_statistics(self, stats: Dict):
        """Print statistics"""
        print("=" * 60)
        print("Conversion Statistics")
        print("=" * 60)
        print(f"Total formulas: {stats['total_formulas']}")
        print(f"Total conversion time: {stats['total_time']:.6f}s")
        print(f"Average conversion time: {stats['avg_time']:.6f}s")
        print(f"Maximum conversion time: {stats['max_time']:.6f}s")
        print(f"Minimum conversion time: {stats['min_time']:.6f}s")
        print()
        
        print("Statistics by Modal Depth:")
        for depth, times in stats['modal_depth_stats'].items():
            avg_time_for_depth = sum(times) / len(times)
            print(f"  Depth {depth}: {len(times)} formulas, average time {avg_time_for_depth:.6f}s")
        print()
    
    def save_results_to_json(self, filename: str):
        """Save conversion results to JSON file"""
        results_data = {
            "conversion_time": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "parameters": {
                "num_agents": len(self.converter.agents),
                "agents": self.converter.agents
            },
            "statistics": {
                "total_formulas": len(self.results),
                "total_time": sum(r.conversion_time for r in self.results),
                "avg_time": sum(r.conversion_time for r in self.results) / len(self.results) if self.results else 0,
                "max_time": max((r.conversion_time for r in self.results), default=0),
                "min_time": min((r.conversion_time for r in self.results), default=0)
            },
            "results": []
        }
        
        for i, result in enumerate(self.results):
            result_data = {
                "id": i + 1,
                "original_formula": str(result.original_formula),
                "aednf": str(result.aednf),
                "aecnf": str(result.aecnf),
                "modal_depth": result.modal_depth,
                "conversion_time": result.conversion_time,
                "aednf_clauses": len(result.aednf.clauses),
                "aecnf_clauses": len(result.aecnf.clauses)
            }
            results_data["results"].append(result_data)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        
        print(f"Results saved to {filename}")

# Test case generator (reuse previous generator)
def generate_test_formulas(num_formulas: int = 10) -> List[EpistemicFormula]:
    """Generate test epistemic formulas"""
    formulas = []
    
    for i in range(num_formulas):
        if i % 4 == 0:
            # Simple proposition
            formula = Proposition(f'p{i}')
        elif i % 4 == 1:
            # Single layer knowledge
            formula = Knowledge(1, Proposition(f'p{i}'))
        elif i % 4 == 2:
            # Binary connective
            formula = BinaryConnective(
                Knowledge(1, Proposition(f'p{i}')),
                Knowledge(2, Proposition(f'q{i}')),
                '∧'
            )
        else:
            # Nested knowledge (may violate alternating constraint)
            formula = Knowledge(1, Knowledge(2, Proposition(f'p{i}')))
        
        formulas.append(formula)
    
    return formulas

def find_latest_epistemic_json() -> str:
    """Find the latest epistemic formulas JSON file"""
    import glob
    import os
    
    pattern = "epistemic_formulas_*.json"
    json_files = glob.glob(pattern)
    
    if not json_files:
        return None
    
    latest_file = max(json_files, key=os.path.getmtime)
    return latest_file

def load_formulas_from_json(filename: str = None) -> List[EpistemicFormula]:
    """Load formulas from JSON file"""
    if filename is None:
        filename = find_latest_epistemic_json()
        if filename is None:
            raise FileNotFoundError("No epistemic formulas JSON file found")
        print(f"Auto-detected JSON file: {filename}")
    
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    parser = FormulaParser()
    formulas = []
    
    for item in data['formulas']:
        formula_str = item['formula']
        formula = parser.parse_formula(formula_str)
        formulas.append(formula)
    
    return formulas

# Main program
def main():
    import sys
    
    print("AEDNF/AECNF Converter Test")
    print("=" * 50)
    
    # Create converter
    agents = [1, 2, 3]
    converter = AEDNFAECNFConverter(agents)
    
    # 检查命令行参数
    input_file = None
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        print(f"Using input file from command line: {input_file}")
    
    # Load formulas from JSON file
    try:
        formulas = load_formulas_from_json(input_file)  # 如果input_file为None，会自动检测最新文件
        print(f"Loaded {len(formulas)} formulas from JSON file")
    except FileNotFoundError as e:
        print(f"JSON file not found: {e}")
        print("Using generated test formulas instead")
        formulas = generate_test_formulas(8)
    
    # Run benchmark on all formulas
    print(f"\nRunning conversion on all {len(formulas)} formulas...")
    benchmark = ConversionBenchmark(converter)
    stats = benchmark.benchmark_formula_set(formulas)
    
    # Save results to JSON
    output_filename = f"aednf_aecnf_results_{time.strftime('%Y%m%d_%H%M%S')}.json"
    benchmark.save_results_to_json(output_filename)
    
    # Show some detailed conversion results
    print("Detailed conversion result examples:")
    print("-" * 40)
    for i, result in enumerate(benchmark.results[:3]):  # Only show first 3
        print(f"Formula {i+1}: {result.original_formula}")
        print(f"AEDNF ({len(result.aednf.clauses)} clauses): {result.aednf}")
        print(f"AECNF ({len(result.aecnf.clauses)} clauses): {result.aecnf}")
        print(f"Modal Depth: {result.modal_depth}")
        print(f"Conversion time: {result.conversion_time:.6f}s")
        print("-" * 40)

if __name__ == "__main__":
    main()