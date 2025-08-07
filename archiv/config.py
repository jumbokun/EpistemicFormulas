from dataclasses import dataclass

@dataclass
class EpistemicConfig:
    """Configuration for epistemic formula generation"""
    num_agents: int = 3          # Size of agent set
    max_depth: int = 10           # Maximum nesting depth (K operators only)
    max_length: int = 100         # Maximum formula length (number of operators)
    num_props: int = 10           # Number of propositional variables
    
    # Probability weights for different formula types
    prop_weight: float = 0.1      # Propositional variables (reduced)
    neg_weight: float = 0.2       # Negation (reduced)
    binary_weight: float = 0.2    # Binary connectives (∧, ∨, →) (reduced)
    knowledge_weight: float = 0.5 # Knowledge operators K_i (increased significantly)

# Default configuration for different scenarios
DEFAULT_CONFIG = EpistemicConfig()

# Configuration for testing
TEST_CONFIG = EpistemicConfig(
    num_agents=2,
    max_depth=2,
    max_length=8,
    num_props=3,
    prop_weight=0.4,
    neg_weight=0.2,
    binary_weight=0.2,
    knowledge_weight=0.2
)

# Configuration for complex formulas
COMPLEX_CONFIG = EpistemicConfig(
    num_agents=5,
    max_depth=5,
    max_length=20,
    num_props=8,
    prop_weight=0.1,      # Very low proposition weight
    neg_weight=0.25,      # High negation weight
    binary_weight=0.4,    # High binary connective weight
    knowledge_weight=0.25 # High knowledge operator weight
)

# Configuration for very complex formulas
VERY_COMPLEX_CONFIG = EpistemicConfig(
    num_agents=7,
    max_depth=8,
    max_length=50,
    num_props=12,
    prop_weight=0.05,     # Very low proposition weight
    neg_weight=0.2,       # Moderate negation weight
    binary_weight=0.45,   # Very high binary connective weight
    knowledge_weight=0.3  # High knowledge operator weight
) 