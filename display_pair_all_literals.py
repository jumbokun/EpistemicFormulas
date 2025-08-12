import json
from pathlib import Path
from typing import Any, Dict, List, Optional

# 可调：递归展开的最大层数（避免输出爆炸）
MAX_EXPAND_DEPTH = 3


def load_final_formula(path: str) -> Optional[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        print(f"找不到 {path}，请先运行生成器。")
        return None
    data = json.loads(p.read_text(encoding='utf-8'))
    return data.get('final_formula')


def format_objective(obj: Optional[Dict[str, Any]]) -> str:
    if not obj:
        return '⊥'
    desc = obj.get('description')
    return desc if desc else '⊤'


def maybe_simplify_knowledge_like(pair: Dict[str, Any], depth: int) -> Optional[str]:
    """如果 pair 形如 (⊤ ∧ 单一知识文字)，则将其简化为 K/¬K 的直观形式。否则返回 None。"""
    aednf = pair.get('aednf', {})
    terms = aednf.get('terms', [])
    if len(terms) != 1:
        return None
    t = terms[0]
    obj_desc = format_objective(t.get('objective_part'))
    pos = t.get('positive_literals', [])
    neg = t.get('negative_literals', [])
    # 仅在客观部分为 ⊤ 且只有一个知识文字时进行简化
    if obj_desc == '⊤' and (len(pos) + len(neg) == 1):
        if pos:
            lit = pos[0]
            inner_pair = lit.get('formula')
            inner = fol_of_pair(inner_pair, depth + 1) if isinstance(inner_pair, dict) else 'φ'
            return f"K_{lit.get('agent')}({inner})"
        if neg:
            lit = neg[0]
            inner_pair = lit.get('formula')
            inner = fol_of_pair(inner_pair, depth + 1) if isinstance(inner_pair, dict) else 'φ'
            return f"¬K_{lit.get('agent')}({inner})"
    return None


def fol_of_pair(pair: Dict[str, Any], depth: int = 0) -> str:
    """将一个 pair 的 AEDNF 第一项转为简短FOL串，递归受限展开。"""
    if depth > MAX_EXPAND_DEPTH:
        return f"φ_{pair.get('depth', '?')}"
    # 尝试简化为直观的 K/¬K 形式
    simplified = maybe_simplify_knowledge_like(pair, depth)
    if simplified is not None:
        return simplified

    aednf = pair.get('aednf', {})
    terms = aednf.get('terms', [])
    if not terms:
        return '⊥'
    t = terms[0]
    parts: List[str] = []
    parts.append(format_objective(t.get('objective_part')))
    for lit in t.get('positive_literals', []):
        inner = '…'
        inner_pair = lit.get('formula')
        if isinstance(inner_pair, dict):
            inner = fol_of_pair(inner_pair, depth + 1)
        parts.append(f"K_{lit.get('agent')}(" + inner + ")")
    for lit in t.get('negative_literals', []):
        inner = '…'
        inner_pair = lit.get('formula')
        if isinstance(inner_pair, dict):
            inner = fol_of_pair(inner_pair, depth + 1)
        parts.append(f"¬K_{lit.get('agent')}(" + inner + ")")
    return " ∧ ".join(parts)


def print_aednf(aednf: Dict[str, Any]):
    terms = aednf.get('terms', [])
    print(f"AEDNF: 共 {len(terms)} 个 term，depth={aednf.get('depth')}")
    for idx, t in enumerate(terms):
        print(f"- term[{idx}]:")
        print(f"  objective: {format_objective(t.get('objective_part'))}")
        pos = t.get('positive_literals', [])
        neg = t.get('negative_literals', [])
        print(f"  +literals ({len(pos)}):")
        for i, lit in enumerate(pos):
            inner_pair = lit.get('formula')
            inner_str = fol_of_pair(inner_pair) if isinstance(inner_pair, dict) else 'φ'
            print(f"    +[{i}] K_{lit.get('agent')}(" + inner_str + f")  [depth={lit.get('depth')}, neg={lit.get('negated')}]")
        print(f"  -literals ({len(neg)}):")
        for i, lit in enumerate(neg):
            inner_pair = lit.get('formula')
            inner_str = fol_of_pair(inner_pair) if isinstance(inner_pair, dict) else 'φ'
            print(f"    -[{i}] ¬K_{lit.get('agent')}(" + inner_str + f")  [depth={lit.get('depth')}, neg={lit.get('negated')}]")


def print_aecnf(aecnf: Dict[str, Any]):
    clauses = aecnf.get('clauses', [])
    print(f"AECNF: 共 {len(clauses)} 个 clause，depth={aecnf.get('depth')}")
    for idx, c in enumerate(clauses):
        print(f"- clause[{idx}]:")
        print(f"  objective: {format_objective(c.get('objective_part'))}")
        pos = c.get('positive_literals', [])
        neg = c.get('negative_literals', [])
        print(f"  +literals ({len(pos)}):")
        for i, lit in enumerate(pos):
            inner_pair = lit.get('formula')
            inner_str = fol_of_pair(inner_pair) if isinstance(inner_pair, dict) else 'φ'
            print(f"    +[{i}] K_{lit.get('agent')}(" + inner_str + f")  [depth={lit.get('depth')}, neg={lit.get('negated')}]")
        print(f"  -literals ({len(neg)}):")
        for i, lit in enumerate(neg):
            inner_pair = lit.get('formula')
            inner_str = fol_of_pair(inner_pair) if isinstance(inner_pair, dict) else 'φ'
            print(f"    -[{i}] ¬K_{lit.get('agent')}(" + inner_str + f")  [depth={lit.get('depth')}, neg={lit.get('negated')}]")


def main(json_path: str = 'ultra_formula_generation.json'):
    pair = load_final_formula(json_path)
    if not pair:
        return
    depth = pair.get('depth')
    print(f"—— 最终公式（AEDNFAECNFPair）——  depth={depth}")
    print()
    aednf = pair.get('aednf', {})
    aecnf = pair.get('aecnf', {})
    print_aednf(aednf)
    print()
    print_aecnf(aecnf)


if __name__ == '__main__':
    main()
