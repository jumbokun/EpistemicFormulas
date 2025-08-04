from typing import Dict, Optional, List, Set
import re

# 全局常量
false_id = 0
true_id = 1

# 全局变量
node_counter = 2
branch_cache: Dict[str, 'Node'] = {}
conjoin_cache: Dict[str, 'Node'] = {} 
negate_cache: Dict[int, 'Node'] = {}
symbol_2_number: Dict[str, int] = {}
number_2_symbol: Dict[int, str] = {}
nodeID_2_key: Dict[int, str] = {}
K_flat_cache: Dict[int, 'Node'] = {}
kp_sat_cache: Dict[str, 'Node'] = {}
conjoin_sat_cache: Dict[str, bool] = {}
good_order = 0
bad_order = 0
kp_sat_cache_usage = 0
conjoin_sat_cahce_usage = 0

# 全局节点变量（将在Node类定义后初始化）
false_node: Optional['Node'] = None
true_node: Optional['Node'] = None

class Node:
    """OBDD节点"""
    _instances = []
    
    def __init__(self, id: int, var_id: Optional[int], when1: Optional['Node'], when0: Optional['Node'], is_enode: bool = False):
        self.id = id
        self.var_id = var_id
        self.when0 = when0
        self.when1 = when1
        self.is_enode = is_enode
        Node._instances.append(self)
    
    @classmethod
    def delete_all_instances(cls):
        cls._instances.clear()

# 初始化基本节点
false_node = Node(id=0, var_id=None, when0=None, when1=None)
true_node = Node(id=1, var_id=None, when0=None, when1=None)
branch_cache["false"] = false_node
branch_cache["true"] = true_node
nodeID_2_key[false_id] = "false"
nodeID_2_key[true_id] = "true"

def reset_cache():
    """重置所有缓存和全局变量"""
    global node_counter, branch_cache, conjoin_cache, negate_cache
    global symbol_2_number, number_2_symbol, nodeID_2_key
    global false_node, true_node, K_flat_cache, good_order, bad_order
    global kp_sat_cache, kp_sat_cache_usage, conjoin_sat_cache, conjoin_sat_cahce_usage

    good_order = 0
    bad_order = 0
    kp_sat_cache_usage = 0
    conjoin_sat_cahce_usage = 0
    node_counter = 2
    branch_cache = {}
    conjoin_cache = {} 
    negate_cache = {}
    symbol_2_number = {}
    number_2_symbol = {}
    nodeID_2_key = {}   
    K_flat_cache = {}
    kp_sat_cache = {}
    conjoin_sat_cache = {}
    
    Node.delete_all_instances()
    false_node = Node(id=0, var_id=None, when0=None, when1=None)
    true_node = Node(id=1, var_id=None, when0=None, when1=None)
    branch_cache["false"] = false_node
    branch_cache["true"] = true_node
    nodeID_2_key[false_id] = "false"
    nodeID_2_key[true_id] = "true"

def get_key(head_id: int, when0: Node, when1: Node) -> str:
    """生成节点的唯一键"""
    if head_id < 0:
        key = "K(" + str(-head_id) + ")?" + str(when1.id) + ":" + str(when0.id)
    else:
        key = str(head_id) + "?" + str(when1.id) + ":" + str(when0.id)
    return key

def mk_branch(var_id: int, when0: Node, when1: Node) -> Node:
    """创建分支节点（命题变量节点）"""
    if when0.id == when1.id:
        return when0
    
    key = get_key(head_id=var_id, when0=when0, when1=when1)
    if key not in branch_cache:
        counter = len(branch_cache)
        node = Node(id=counter, var_id=var_id, when1=when1, when0=when0, is_enode=False)
        branch_cache[key] = node
        nodeID_2_key[counter] = key
        return node
    else:
        return branch_cache[key]

def mk_know(node_id: int, when0: Node, when1: Node) -> Node:
    """创建认知节点（K算子节点）"""
    knode = branch_cache[nodeID_2_key[node_id]]
    assert knode.var_id >= 0
    
    key = get_key(head_id=-node_id, when0=when0, when1=when1)
    
    if key not in branch_cache:
        counter = len(branch_cache)
        enode = Node(id=counter, var_id=-node_id, when1=when1, when0=when0, is_enode=True)
        branch_cache[key] = enode
        nodeID_2_key[counter] = key
        return enode
    else:
        return branch_cache[key]

def negate(node: Node) -> Node:
    """否定操作"""
    if node.id == true_id:
        return false_node
    if node.id == false_id:
        return true_node
    if node.id not in negate_cache:
        rlt = mk_branch(var_id=node.var_id, when0=negate(node.when0), when1=negate(node.when1))
        negate_cache[node.id] = rlt
        return rlt
    else:
        return negate_cache[node.id]

def conjoin(lhs: Node, rhs: Node) -> Node:
    """合取操作（AND）"""
    if lhs.id == rhs.id:
        return lhs
    if lhs.id == false_id or rhs.id == false_id:
        return branch_cache["false"]
    if lhs.id == true_id:
        return rhs
    if rhs.id == true_id:
        return lhs
    if lhs.var_id > rhs.var_id:
        tmp = lhs
        lhs = rhs
        rhs = tmp
    
    key = str(lhs.id) + ":" + str(rhs.id)
    if key not in conjoin_cache:
        # 比较变量
        if lhs.var_id > 0:
            if lhs.var_id == rhs.var_id:
                rlt = mk_branch(var_id=lhs.var_id, 
                              when0=conjoin(lhs=lhs.when0, rhs=rhs.when0), 
                              when1=conjoin(lhs=lhs.when1, rhs=rhs.when1))
            else:  # lhs.var_id < rhs.var_id
                rlt = mk_branch(var_id=lhs.var_id, 
                              when0=conjoin(lhs=lhs.when0, rhs=rhs), 
                              when1=conjoin(lhs=lhs.when1, rhs=rhs))
        else:
            if lhs.var_id == rhs.var_id:
                rlt = mk_know(node_id=-lhs.var_id, 
                            when0=conjoin(lhs=lhs.when0, rhs=rhs.when0), 
                            when1=conjoin(lhs=lhs.when1, rhs=rhs.when1))
            else:  # lhs.var_id < rhs.var_id
                rlt = mk_know(node_id=-lhs.var_id, 
                            when0=conjoin(lhs=lhs.when0, rhs=rhs), 
                            when1=conjoin(lhs=lhs.when1, rhs=rhs))
        conjoin_cache[key] = rlt
    
    return conjoin_cache[key]

def disjoin(lhs: Node, rhs: Node) -> Node:
    """析取操作（OR）"""
    if lhs.id == true_id or rhs.id == true_id:
        return true_node
    if lhs.id == false_id:
        return rhs
    if rhs.id == false_id:
        return lhs
    if rhs.id == lhs.id:
        return rhs
    return negate(conjoin(lhs=negate(lhs), rhs=negate(rhs)))

def implies(premise: Node, conclusion: Node) -> Node:
    """蕴含操作（→）"""
    if premise.id == true_id:
        return conclusion
    if premise.id == false_id:
        return true_node
    if conclusion.id == true_id:
        return true_node
    if conclusion.id == false_id:
        return negate(premise)
    if premise.id == conclusion.id:
        return true_node
    return disjoin(negate(premise), conclusion)

def display(x: Node) -> str:
    """显示节点为字符串"""
    if x.id == true_id: 
        return "true"
    if x.id == false_id:
        return "false"
    if x.var_id > 0:
        rlt = "(" + str(number_2_symbol[x.var_id]) + "? " + display(x.when1) + ": " + display(x.when0) + ")"
    else:
        rlt = "(K" + str(display(branch_cache[nodeID_2_key[-x.var_id]])) + "? " + display(x.when1) + ": " + display(x.when0) + ")"
    return rlt

def display_traditional(x: Node) -> str:
    """以传统逻辑符号显示公式"""
    if x.id == true_id: 
        return "⊤"
    if x.id == false_id:
        return "⊥"
    if x.var_id > 0:
        var_name = str(number_2_symbol[x.var_id])
        when1_str = display_traditional(x.when1)
        when0_str = display_traditional(x.when0)
        
        # 简化常见情况
        if when1_str == "⊤" and when0_str == "⊥":
            return var_name
        elif when1_str == "⊥" and when0_str == "⊤":
            return f"¬{var_name}"
        else:
            return f"({var_name} ∧ {when1_str}) ∨ (¬{var_name} ∧ {when0_str})"
    else:
        # 这是认知算子
        inner_formula = display_traditional(branch_cache[nodeID_2_key[-x.var_id]])
        when1_str = display_traditional(x.when1)
        when0_str = display_traditional(x.when0)
        
        if when1_str == "⊤" and when0_str == "⊥":
            return f"K({inner_formula})"
        elif when1_str == "⊥" and when0_str == "⊤":
            return f"¬K({inner_formula})"
        else:
            return f"(K({inner_formula}) ∧ {when1_str}) ∨ (¬K({inner_formula}) ∧ {when0_str})"

def V(x: str) -> Node:
    """创建变量节点"""
    if x in symbol_2_number:
        return mk_branch(var_id=symbol_2_number[x], when0=false_node, when1=true_node)
    index = len(symbol_2_number) + 1
    number_2_symbol[index] = x
    symbol_2_number[x] = index
    node = mk_branch(var_id=index, when0=false_node, when1=true_node)
    return node

# 便利函数
def NOT(node: Node) -> Node:
    """否定"""
    return negate(node)

def OR(node1: Node, node2: Node) -> Node:
    """析取"""
    return disjoin(node1, node2)

def AND(node1: Node, node2: Node) -> Node:
    """合取"""
    return conjoin(node1, node2)

def rt_nodes_list(node:Node):
    if node.id == true_id or node.id == false_id:
        return [node.id]
    else:
        return list(set(rt_nodes_list(node=node.when0)+rt_nodes_list(node=node.when1)+[node.id]))

def rt_edges_list(node:Node):
    if node.id == true_id or node.id == false_id:
        return []
    else:
        edge1 = "{}|1|{}".format(str(node.id), str(node.when1.id))
        edge0 = "{}|0|{}".format(str(node.id), str(node.when0.id))
        return list(set([edge0]+[edge1] + rt_edges_list(node=node.when0) + rt_edges_list(node=node.when1)))

def rt_evar_list(node:Node):
    if node.id == true_id or node.id == false_id:
        return []
    if node.var_id >= 0:
        return []
    return list(set(rt_evar_list(node=node.when0)+rt_evar_list(node=node.when1)+[node.var_id]))

class OBDDBuilder:
    """
    将文本表示的客观公式编译为OBDD的构建器
    """
    
    def __init__(self):
        """初始化构建器，确保OBDD环境clean"""
        reset_cache()
        self.variables: Set[str] = set()
    
    def parse_and_build(self, formula_text: str) -> Node:
        """
        解析文本公式并构建OBDD
        
        支持的语法：
        - 变量: v1, v2, p, q 等
        - 否定: ¬v1, ~v1, NOT v1
        - 合取: v1 ∧ v2, v1 & v2, v1 AND v2
        - 析取: v1 ∨ v2, v1 | v2, v1 OR v2
        - 括号: (v1 ∨ v2) ∧ v3
        - 蕴含: v1 → v2, v1 -> v2, v1 IMPLIES v2
        """
        # 预处理：标准化符号
        normalized = self._normalize_formula(formula_text)
        
        # 提取所有变量
        variables = self._extract_variables(normalized)
        self.variables.update(variables)
        
        # 解析并构建OBDD
        obdd_node = self._parse_expression(normalized)
        
        return obdd_node
    
    def _normalize_formula(self, text: str) -> str:
        """标准化公式文本"""
        # 移除空格
        text = re.sub(r'\s+', '', text)
        
        # 标准化逻辑符号
        replacements = [
            ('¬', '~'),
            ('∧', '&'),
            ('∨', '|'),
            ('→', '>'),
            ('->','>'),
            ('NOT', '~'),
            ('AND', '&'),
            ('OR', '|'),
            ('IMPLIES', '>'),
        ]
        
        for old, new in replacements:
            text = text.replace(old, new)
        
        return text
    
    def _extract_variables(self, text: str) -> Set[str]:
        """提取公式中的所有变量"""
        # 匹配变量模式：字母开头，可能跟数字
        pattern = r'\b[a-zA-Z][a-zA-Z0-9]*\b'
        variables = set(re.findall(pattern, text))
        
        # 过滤掉逻辑关键词
        keywords = {'true', 'false', 'TRUE', 'FALSE'}
        return variables - keywords
    
    def _parse_expression(self, text: str) -> Node:
        """解析表达式为OBDD节点"""
        return self._parse_implication(text)
    
    def _parse_implication(self, text: str) -> Node:
        """解析蕴含（最低优先级）"""
        parts = self._split_by_operator(text, '>')
        if len(parts) == 1:
            return self._parse_disjunction(parts[0])
        
        # 右结合：A > B > C = A > (B > C)
        premise = self._parse_disjunction(parts[0])
        conclusion = self._parse_implication('>'.join(parts[1:]))
        return implies(premise, conclusion)
    
    def _parse_disjunction(self, text: str) -> Node:
        """解析析取"""
        parts = self._split_by_operator(text, '|')
        if len(parts) == 1:
            return self._parse_conjunction(parts[0])
        
        result = self._parse_conjunction(parts[0])
        for part in parts[1:]:
            result = OR(result, self._parse_conjunction(part))
        return result
    
    def _parse_conjunction(self, text: str) -> Node:
        """解析合取"""
        parts = self._split_by_operator(text, '&')
        if len(parts) == 1:
            return self._parse_negation(parts[0])
        
        result = self._parse_negation(parts[0])
        for part in parts[1:]:
            result = AND(result, self._parse_negation(part))
        return result
    
    def _parse_negation(self, text: str) -> Node:
        """解析否定"""
        if text.startswith('~'):
            return NOT(self._parse_atom(text[1:]))
        return self._parse_atom(text)
    
    def _parse_atom(self, text: str) -> Node:
        """解析原子公式"""
        # 移除外层括号
        while text.startswith('(') and text.endswith(')'):
            if self._matching_paren(text) == len(text) - 1:
                text = text[1:-1]
            else:
                break
        
        # 特殊常量
        if text.lower() in ['true', '⊤']:
            return true_node
        if text.lower() in ['false', '⊥']:
            return false_node
        
        # 如果包含操作符，递归解析
        if any(op in text for op in ['&', '|', '>', '~']):
            return self._parse_expression(text)
        
        # 原子变量
        return V(text)
    
    def _split_by_operator(self, text: str, op: str) -> List[str]:
        """按操作符分割，考虑括号嵌套"""
        parts = []
        current = ""
        paren_depth = 0
        i = 0
        
        while i < len(text):
            char = text[i]
            
            if char == '(':
                paren_depth += 1
                current += char
            elif char == ')':
                paren_depth -= 1
                current += char
            elif char == op and paren_depth == 0:
                parts.append(current)
                current = ""
            else:
                current += char
            
            i += 1
        
        if current:
            parts.append(current)
        
        return parts if len(parts) > 1 else [text]
    
    def _matching_paren(self, text: str) -> int:
        """找到匹配的右括号位置"""
        if not text.startswith('('):
            return -1
        
        depth = 0
        for i, char in enumerate(text):
            if char == '(':
                depth += 1
            elif char == ')':
                depth -= 1
                if depth == 0:
                    return i
        return -1

# 测试函数
def test_obdd():
    """测试OBDD功能"""
    print("=== OBDD测试 ===")
    
    builder = OBDDBuilder()
    
    test_cases = [
        "v1",
        "¬v1", 
        "v1 ∧ v2",
        "v1 ∨ v2",
        "¬v1 ∨ v2",
        "(v1 ∧ v2) ∨ v3",
        "v1 → v2",
    ]
    
    for formula_text in test_cases:
        print(f"\n测试: {formula_text}")
        try:
            builder = OBDDBuilder()  # 重置避免变量冲突
            obdd_node = builder.parse_and_build(formula_text)
            print(f"  ✓ OBDD: {display(obdd_node)}")
            print(f"  ✓ 传统: {display_traditional(obdd_node)}")
        except Exception as e:
            print(f"  ✗ 失败: {e}")

if __name__ == "__main__":
    test_obdd()