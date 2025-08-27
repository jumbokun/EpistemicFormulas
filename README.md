# 认知逻辑系统 (Epistemic Logic System)

一个完整的认知逻辑系统实现，支持K45认知逻辑的公式生成、SAT检查和反内省操作。

## 功能特性

### 🧠 认知逻辑支持
- **K45认知逻辑**：支持代理的不一致知识状态
- **多层嵌套**：支持任意深度的知识算子嵌套
- **多代理系统**：支持多个代理的知识交互

### 🔧 核心功能
- **公式生成**：从0开始随机生成认知逻辑公式
- **SAT检查**：检查认知逻辑公式的可满足性
- **反内省**：实现认知逻辑的反内省操作
- **OBDD支持**：基于有序二元决策图的底层实现

### 📊 性能监控
- **时间统计**：记录每个公式的SAT检查时间
- **成功率统计**：监控公式生成和测试的成功率
- **可满足性分析**：统计可满足公式的比例

## 系统架构

### 数据模型
- `ObjectiveFormula`：客观公式
- `KnowledgeLiteral`：知识文字
- `AEDNFTerm`：AEDNF项
- `AEDNF`：交替认知析取范式
- `AECNFClause`：AECNF子句
- `AECNF`：交替认知合取范式
- `AEDNFAECNFPair`：AEDNF-AECNF对

### 核心算法
- **SAT检查算法**：基于Γ/Δ框架的认知一致性检查
- **公式生成算法**：递归随机生成认知逻辑公式
- **OBDD算法**：有序二元决策图的构建和操作

## 快速开始

### 安装依赖
```bash
pip install pydantic
```

### 基本使用
```python
from epistemic_logic_system import FormulaGenerator, create_objective_pair, know, land

# 创建公式生成器
generator = FormulaGenerator(max_depth=3, max_agents=3, max_vars=5)

# 生成公式
formulas = generator.generate_formulas(10)

# 测试公式
results = generator.test_formulas(formulas)

# 查看结果
for result in results:
    print(f"公式 {result['index']}: 可满足={result['is_satisfiable']}, 时间={result['time']:.4f}秒")
```

### 手动构造公式
```python
# 创建基本变量
p = create_objective_pair("p")
q = create_objective_pair("q")

# 构造知识公式
k_a_p = know(p, "agent_0")
k_b_q = know(q, "agent_1")

# 构造复杂公式
complex_formula = land(k_a_p, k_b_q)

# 检查可满足性
is_sat = sat_pair(complex_formula)
print(f"公式可满足: {is_sat}")
```

## 算法说明

### SAT检查算法
系统实现了基于Γ/Δ框架的认知一致性检查：

1. **客观层检查**：使用OBDD检查客观公式的可满足性
2. **认知层检查**：使用`sat_K(Γ, Δ)`检查认知一致性
3. **递归处理**：递归处理嵌套的知识算子

### 公式生成算法
采用递归随机生成策略：

1. **深度控制**：限制公式的最大嵌套深度
2. **操作选择**：随机选择逻辑操作（∧, ∨, ¬, K）
3. **变量分配**：随机分配命题变量和代理

### K45语义
在K45认知逻辑中：
- 代理可以有不一致的知识状态
- `K_a(false)` 被认为是可满足的
- 支持多层嵌套的知识算子

## 性能特点

### 时间复杂度
- **公式生成**：O(d^n)，其中d是最大深度，n是操作数
- **SAT检查**：O(2^k)，其中k是变量数
- **认知检查**：O(a * 2^v)，其中a是代理数，v是变量数

### 空间复杂度
- **OBDD存储**：O(n)，其中n是节点数
- **公式表示**：O(d * a)，其中d是深度，a是代理数

## 测试结果

系统经过全面测试，验证了以下场景：

### ✅ 单层认知逻辑
- 基本客观公式的SAT检查
- 单层知识算子的处理
- 简单逻辑操作的正确性

### ✅ 多层嵌套认知逻辑
- 2层嵌套知识算子的处理
- 混合层级公式的正确性
- 复杂嵌套场景的验证

### ✅ K45语义正确性
- `K_a(false)` 的可满足性
- 代理不一致知识状态的处理
- 嵌套矛盾的正确处理

## 文件结构

```
EpistemicFormulas/
├── epistemic_logic_system.py    # 主系统文件（整合所有功能）
├── README.md                    # 项目说明文档
└── requirements.txt             # 依赖文件
```

## 依赖要求

- Python 3.7+
- pydantic >= 1.8.0
- typing (内置)
- dataclasses (内置)

## 贡献指南

欢迎提交Issue和Pull Request来改进系统：

1. Fork项目
2. 创建功能分支
3. 提交更改
4. 推送到分支
5. 创建Pull Request

## 许可证

本项目采用MIT许可证。

## 联系方式

如有问题或建议，请通过以下方式联系：
- 提交GitHub Issue
- 发送邮件至项目维护者

---

**注意**：本系统实现了K45认知逻辑的语义，其中代理可以有不一致的知识状态。这是认知逻辑的一个重要特性，反映了现实世界中代理可能具有不完整或不一致的知识。
