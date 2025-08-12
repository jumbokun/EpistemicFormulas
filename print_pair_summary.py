import json
from pathlib import Path

def main():
    path = Path('ultra_formula_generation.json')
    if not path.exists():
        print('找不到 ultra_formula_generation.json，请先运行生成器。')
        return
    data = json.loads(path.read_text(encoding='utf-8'))
    F = data.get('final_formula', {})
    aednf = F.get('aednf', {})
    aecnf = F.get('aecnf', {})
    depth = F.get('depth')

    terms = aednf.get('terms', [])
    clauses = aecnf.get('clauses', [])

    print('—— 最终公式结构（AEDNFAECNFPair）——')
    print(f'- depth: {depth}')
    print(f'- AEDNF: terms={len(terms)}, depth={aednf.get("depth")}')
    print(f'- AECNF: clauses={len(clauses)}, depth={aecnf.get("depth")}')

    if terms:
        t = terms[0]
        obj = t.get('objective_part', {})
        print('\n[AEDNF 第一个term]')
        print(f'- objective: {obj.get("description")}')
        print(f'- +lits: {len(t.get("positive_literals", []))}, -lits: {len(t.get("negative_literals", []))}')
        for i, lit in enumerate(t.get('positive_literals', [])[:3]):
            print(f'  +[{i}] K_{lit.get("agent")}(depth={lit.get("depth")}, neg={lit.get("negated")})')
        for i, lit in enumerate(t.get('negative_literals', [])[:3]):
            print(f'  -[{i}] K_{lit.get("agent")}(depth={lit.get("depth")}, neg={lit.get("negated")})')

    if clauses:
        c = clauses[0]
        obj = c.get('objective_part', {})
        print('\n[AECNF 第一个clause]')
        print(f'- objective: {obj.get("description")}')
        print(f'- +lits: {len(c.get("positive_literals", []))}, -lits: {len(c.get("negative_literals", []))}')
        for i, lit in enumerate(c.get('positive_literals', [])[:3]):
            print(f'  +[{i}] K_{lit.get("agent")}(depth={lit.get("depth")}, neg={lit.get("negated")})')
        for i, lit in enumerate(c.get('negative_literals', [])[:3]):
            print(f'  -[{i}] K_{lit.get("agent")}(depth={lit.get("depth")}, neg={lit.get("negated")})')

if __name__ == '__main__':
    main()
