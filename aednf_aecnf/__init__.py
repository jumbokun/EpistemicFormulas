"""
AEDNF/AECNF模块

提供交替认知析取范式(AEDNF)和交替认知合取范式(AECNF)的实现
"""

# OBDD相关
from .obdd import (
    Node, OBDDBuilder, 
    AND, OR, NOT, implies,
    display, display_traditional,
    reset_cache, V
)

# 模型相关
from .models import (
    Agent, Variable,
    Formula, ObjectiveFormula, KnowledgeLiteral,
    AEDNFTerm, AECNFClause,
    AEDNF, AECNF, AEDNFAECNFPair,
    create_objective_pair
)

# 构建器相关
from .builders import Depth0Builder
from .builders_depth1 import Depth1Builder

__all__ = [
    # OBDD相关
    'Node', 'OBDDBuilder', 'AND', 'OR', 'NOT', 'implies',
    'display', 'display_traditional', 'reset_cache', 'V',
    
    # 模型相关
    'Agent', 'Variable', 'Formula', 'ObjectiveFormula', 'KnowledgeLiteral',
    'AEDNFTerm', 'AECNFClause', 'AEDNF', 'AECNF', 'AEDNFAECNFPair',
    'create_objective_pair',
    
    # 构建器相关
    'Depth0Builder', 'Depth1Builder',
]