import json
from pathlib import Path
from typing import Any, Dict, List, Optional

MAX_EXPAND_DEPTH = 2


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


def fol_of_pair(pair: Dict[str, Any], depth: int = 0) -> str:
    if pair is None:
        return 'φ'
    if depth > MAX_EXPAND_DEPTH:
        return f"φ_{pair.get('depth', '?')}"
    aednf = pair.get('aednf', {})
    terms = aednf.get('terms', [])
    if not terms:
        return '⊥'
    t = terms[0]
    parts: List[str] = []
    parts.append(format_objective(t.get('objective_part')))
    for lit in t.get('positive_literals', []):
        inner = fol_of_pair(lit.get('formula'), depth + 1) if isinstance(lit.get('formula'), dict) else 'φ'
        parts.append(f"K_{lit.get('agent')}({inner})")
    for lit in t.get('negative_literals', []):
        inner = fol_of_pair(lit.get('formula'), depth + 1) if isinstance(lit.get('formula'), dict) else 'φ'
        parts.append(f"¬K_{lit.get('agent')}({inner})")
    return " ∧ ".join(parts)


def classify_literals(literals: List[Dict[str, Any]], agent: str):
    subj: List[str] = []
    obj: List[str] = []
    for lit in literals:
        inner = fol_of_pair(lit.get('formula')) if isinstance(lit.get('formula'), dict) else 'φ'
        s = ("¬" if lit.get('negated') else "") + f"K_{lit.get('agent')}({inner})"
        if lit.get('agent') == agent:
            subj.append(s)
        else:
            obj.append(s)
    return obj, subj


def print_aednf(aednf: Dict[str, Any], agent: str):
    terms = aednf.get('terms', [])
    print(f"AEDNF: 共 {len(terms)} 个 term，depth={aednf.get('depth')}")
    for idx, t in enumerate(terms):
        print(f"- term[{idx}]:")
        print(f"  objective (总为a-objective): {format_objective(t.get('objective_part'))}")
        pos = t.get('positive_literals', [])
        neg = t.get('negative_literals', [])
        all_lits = pos + neg
        obj, subj = classify_literals(all_lits, agent)
        print(f"  a-objective literals ({len(obj)}):")
        for i, s in enumerate(obj):
            print(f"    [{i}] {s}")
        print(f"  a-subjective literals ({len(subj)}):")
        for i, s in enumerate(subj):
            print(f"    [{i}] {s}")


def print_aecnf(aecnf: Dict[str, Any], agent: str):
    clauses = aecnf.get('clauses', [])
    print(f"AECNF: 共 {len(clauses)} 个 clause，depth={aecnf.get('depth')}")
    for idx, c in enumerate(clauses):
        print(f"- clause[{idx}]:")
        print(f"  objective (总为a-objective): {format_objective(c.get('objective_part'))}")
        pos = c.get('positive_literals', [])
        neg = c.get('negative_literals', [])
        all_lits = pos + neg
        obj, subj = classify_literals(all_lits, agent)
        print(f"  a-objective literals ({len(obj)}):")
        for i, s in enumerate(obj):
            print(f"    [{i}] {s}")
        print(f"  a-subjective literals ({len(subj)}):")
        for i, s in enumerate(subj):
            print(f"    [{i}] {s}")


def main(agent: str = 'a1', json_path: str = 'ultra_formula_generation.json'):
    pair = load_final_formula(json_path)
    if not pair:
        return
    print(f"—— 以代理 {agent} 为参照的 a-objective / a-subjective 划分 ——")
    print()
    aednf = pair.get('aednf', {})
    aecnf = pair.get('aecnf', {})
    print_aednf(aednf, agent)
    print()
    print_aecnf(aecnf, agent)


if __name__ == '__main__':
    import sys
    agent = sys.argv[1] if len(sys.argv) > 1 else 'a1'
    path = sys.argv[2] if len(sys.argv) > 2 else 'ultra_formula_generation.json'
    main(agent, path)
