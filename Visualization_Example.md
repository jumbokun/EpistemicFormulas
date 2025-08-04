# AEDNF/AECNF 可视化示例

## 目标格式

我们的可视化系统将生成如下格式的输出，解决了递归生成过程中pair形式和最终展示形式的问题：

## 示例 1：双智能体 MA-EDNF

```
=== MA-EDNF 公式结构 ===

主公式: (v1? true: false) ∧ K(1)?(true: false) ∧ ¬K(2)?(true: false)

命题部分 α:
  α_AEDNF (DNF): v1 ∨ (v2 ∧ v3)
  α_AECNF (CNF): (v1 ∨ v2) ∧ (v1 ∨ v3)

正知识项:
  K_1(φ_1,1) = v2 ∨ ¬v4
  K_2(φ_2,1) = v3 ∧ v5

负知识项:
  ¬K_1(ψ_1,1) = v4 ∨ v5
  ¬K_2(ψ_2,1) = ¬v2 ∧ v1
```

## 示例 2：对应的 MA-ECNF

```
=== MA-ECNF 公式结构 ===

主公式: (v1? true: false) ∨ ¬K(1)?(true: false) ∨ K(2)?(true: false)

命题部分 α:
  α_AEDNF (DNF): v1 ∨ (v2 ∧ v3)
  α_AECNF (CNF): (v1 ∨ v2) ∧ (v1 ∨ v3)

正知识项:
  K_1(φ_1,1) = v2 ∨ ¬v4
  K_2(φ_2,1) = v3 ∧ v5

负知识项:
  ¬K_1(ψ_1,1) = v4 ∨ v5
  ¬K_2(ψ_2,1) = ¬v2 ∧ v1
```

## 核心解决方案

### 1. 递归生成时的Pair维护

在递归生成过程中：
- ✅ 所有子公式 α、φ_a、ψ_{a,j} 都作为 AEDNF₀/AECNF₀ 的 pair 生成
- ✅ 保持交替约束：知识算子内的公式是 a-objective 的
- ✅ 每一步操作后都返回 AEDNFAECNFPair 对象

### 2. 最终展示时的结构化

在可视化展示时：
- 🎯 **主公式**：显示完整的 MA-EDNF/MA-ECNF 结构
- 🎯 **命题部分α**：单独展示其 AEDNF/AECNF pair（即 DNF/CNF）
- 🎯 **知识项分解**：按智能体分类显示正负知识项
- 🎯 **简化形式**：去掉空项，只显示实际存在的部分

### 3. 实际使用流程

```python
# Step 1: 生成公式对（内部保持pair形式）
agents = ["1", "2"]
pair = generate_aednf_aecnf_pair(
    num_agents=2, 
    num_var=5, 
    complexity=12, 
    target_depth=1
)

# Step 2: 结构化提取和可视化
structures = generate_and_display_ma_formula(
    agents, num_var=5, complexity=12, depth=1
)

# Step 3: 格式化输出
print(structures["MA-EDNF"])
print(structures["MA-ECNF"])
```

## 核心数据结构

### MAFormulaStructure
```python
@dataclass
class MAFormulaStructure:
    alpha_aednf: str          # α的AEDNF部分（propositional DNF）
    alpha_aecnf: str          # α的AECNF部分（propositional CNF）
    positive_knowledge: Dict  # agent -> [K_a(φ) formulas]
    negative_knowledge: Dict  # agent -> [¬K_a(ψ) formulas]
    formula_type: str         # "MA-EDNF" or "MA-ECNF"
    original_formula: str     # 完整的原始公式显示
```

## 优势总结

1. **理论正确性**: 严格遵循AEDNF/AECNF定义，所有子公式都是pair
2. **实用可视化**: 最终展示符合论文格式，易于理解和验证
3. **结构清晰**: 分别展示命题部分和知识部分
4. **对比友好**: MA-EDNF和MA-ECNF可以并列比较
5. **扩展性强**: 容易扩展到更高深度

## 表示符合论文格式

我们的输出完全符合您图中的格式：

```
α ∧ ⋀_{a∈A}(K_a φ_a ∧ ⋀_{j∈J_a} ¬K_a ψ_{a,j})
```

其中：
- α 单独展示为 AEDNF/AECNF pair
- 各个 K_a φ_a 和 ¬K_a ψ_{a,j} 项分类列出
- 去掉了空的项（如某个智能体没有正知识或负知识）

这样既保证了生成过程的理论正确性，又提供了清晰的最终展示形式！