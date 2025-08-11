from typing import List, Optional, Set
from pydantic import BaseModel, Field, field_validator
from abc import ABC, abstractmethod
from enum import Enum

# 类型别名
Agent = str
Variable = str

class FormulaType(Enum):
    AEDNF = "aednf"
    AECNF = "aecnf"
    OBJECTIVE = "objective"

class Formula(BaseModel, ABC):
    @abstractmethod
    def is_objective_for_agent(self, agent: Agent) -> bool:
        """检查公式是否对指定代理是客观的"""
        pass
    
    @abstractmethod
    def get_depth(self) -> int:
        """获取公式的模态深度"""
        pass

class ObjectiveFormula(Formula):
    """
    客观公式（AECNF_0/AEDNF_0）
    实际上是OBDD节点的包装
    """
    obdd_node_id: int = Field(..., description="OBDD节点的ID")
    description: Optional[str] = Field(None, description="公式的文本描述，用于调试")
    depth: int = Field(0)
    
    def is_objective_for_agent(self, agent: Agent) -> bool:
        """客观公式对所有代理都是客观的"""
        return True
    
    def get_depth(self) -> int:
        """客观公式的深度为0"""
        return 0

class KnowledgeLiteral(BaseModel):
    """认知文字：K_a(φ) 或 ¬K_a(φ)"""
    agent: Agent
    formula: 'AEDNFAECNFPair'
    negated: bool = False
    depth: int = Field(..., ge=0)

    def __init__(self, **data):
        super().__init__(**data)
        # 验证交替约束
        # if not self.formula.is_objective_for_agent(self.agent):
        #     raise ValueError(f"交替约束违反：公式必须对代理人{self.agent}是客观的")

class AEDNFTerm(BaseModel):
    """
    AEDNF中的一个项（term）
    形式：α ∧ ⋀_{a∈A} (K_a φ_a ∧ ⋀_j ¬K_a ψ_{a,j})
    """
    objective_part: ObjectiveFormula
    positive_literals: List[KnowledgeLiteral] = Field(default_factory=list)
    negative_literals: List[KnowledgeLiteral] = Field(default_factory=list)
    
    @field_validator('positive_literals', 'negative_literals')
    @classmethod
    def validate_literals(cls, literals, info):
        field_name = info.field_name
        for lit in literals:
            if field_name == 'positive_literals' and lit.negated:
                raise ValueError("正文字列表中不应有否定的认知文字")
            if field_name == 'negative_literals' and not lit.negated:
                raise ValueError("负文字列表中不应有肯定的认知文字")
        return literals

class AECNFClause(BaseModel):
    """
    AECNF中的一个子句（clause）
    形式：α ∨ ⋁_{a∈A} (¬K_a φ_a ∨ ⋁_j K_a ψ_{a,j})
    """
    objective_part: ObjectiveFormula
    positive_literals: List[KnowledgeLiteral] = Field(default_factory=list)
    negative_literals: List[KnowledgeLiteral] = Field(default_factory=list)

class AEDNF(Formula):
    """
    交替认知析取范式
    形式：⋁_i (αᵢ ∧ ⋀_{a∈A} (K_a φ_a ∧ ⋀_j ¬K_a ψ_{a,j}))
    """
    terms: List[AEDNFTerm] = Field(..., min_length=1)
    depth: int = Field(..., ge=0)
    
    def is_objective_for_agent(self, agent: Agent) -> bool:
        if self.depth == 0:
            return True
        # 检查是否所有项都不以该代理人的认知算子开头
        for term in self.terms:
            # 如果有任何关于该代理人的认知文字，就不是客观的
            for lit in term.positive_literals + term.negative_literals:
                if lit.agent == agent:
                    return False
        return True
    
    def get_depth(self) -> int:
        return self.depth
    
    @field_validator('terms')
    @classmethod
    def validate_terms(cls, terms, info):
        # 获取当前实例的深度值
        data = info.data if hasattr(info, 'data') else {}
        depth = data.get('depth', 0)
        
        if depth > 0:  # 只有深度大于0时才验证子公式深度
            for term in terms:
                # 验证所有子公式的深度
                for lit in term.positive_literals + term.negative_literals:
                    if lit.formula.get_depth() >= depth:
                        raise ValueError(f"子公式深度{lit.formula.get_depth()}不能大于等于当前深度{depth}")
        return terms

class AECNF(Formula):
    """
    交替认知合取范式
    形式：⋀_i (αᵢ ∨ ⋁_{a∈A} (¬K_a φ_a ∨ ⋁_j K_a ψ_{a,j}))
    """
    clauses: List[AECNFClause] = Field(..., min_length=1)
    depth: int = Field(..., ge=0)
    
    def is_objective_for_agent(self, agent: Agent) -> bool:
        if self.depth == 0:
            return True
        # 检查是否所有子句都不以该代理人的认知算子开头
        for clause in self.clauses:
            # 如果有任何关于该代理人的认知文字，就不是客观的
            for lit in clause.positive_literals + clause.negative_literals:
                if lit.agent == agent:
                    return False
        return True
    
    def get_depth(self) -> int:
        return self.depth
    
    # @field_validator('clauses')
    # @classmethod
    # def validate_clauses(cls, clauses, info):
    #     # 获取当前实例的深度值
    #     data = info.data if hasattr(info, 'data') else {}
    #     depth = data.get('depth', 0)
        
    #     if depth > 0:  # 只有深度大于0时才验证子公式深度
    #         for clause in clauses:
    #             # 验证所有子公式的深度
    #             for lit in clause.positive_literals + clause.negative_literals:
    #                 if lit.formula.get_depth() >= depth:
    #                     raise ValueError(f"子公式深度{lit.formula.get_depth()}不能大于等于当前深度{depth}")
    #     return clauses

class AEDNFAECNFPair(BaseModel):
    """AEDNF和AECNF的对"""
    aednf: AEDNF
    aecnf: AECNF
    depth: int = Field(..., ge=0)

def create_objective_pair(content: str) -> AEDNFAECNFPair:
    """
    创建客观公式的AEDNF/AECNF对
    
    Args:
        content: 公式内容，如 "v1", "⊤", "⊥" 等
    """
    from .obdd import V, true_node, false_node, symbol_2_number
    
    # 根据内容创建OBDD节点
    if content == "⊤":
        node = true_node
    elif content == "⊥":
        node = false_node
    else:
        node = V(content)
    
    # 创建客观公式
    obj_formula = ObjectiveFormula(obdd_node_id=node.id, description=content)
    
    # 创建项和子句
    term = AEDNFTerm(objective_part=obj_formula)
    clause = AECNFClause(objective_part=obj_formula)
    
    # 创建AEDNF和AECNF
    aednf = AEDNF(terms=[term], depth=0)
    aecnf = AECNF(clauses=[clause], depth=0)
    
    return AEDNFAECNFPair(aednf=aednf, aecnf=aecnf, depth=0)

# def validate_alternating_constraint(formula: Formula, agent: Agent) -> bool:
#     """验证交替约束"""
#     return formula.is_objective_for_agent(agent)

# def validate_aednf_structure(aednf: AEDNF) -> bool:
#     """验证AEDNF结构"""
#     if aednf.depth == 0:
#         return len(aednf.terms) == 1 and len(aednf.terms[0].positive_literals) == 0 and len(aednf.terms[0].negative_literals) == 0
#     else:
#         # 深度1+的验证逻辑
#         return True

# def validate_aecnf_structure(aecnf: AECNF) -> bool:
#     """验证AECNF结构"""
#     if aecnf.depth == 0:
#         return len(aecnf.clauses) == 1 and len(aecnf.clauses[0].positive_literals) == 0 and len(aecnf.clauses[0].negative_literals) == 0
#     else:
#         # 深度1+的验证逻辑
#         return True
