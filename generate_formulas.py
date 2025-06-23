#!/usr/bin/env python3
"""
Epistemic Formula Generator Script

Usage:
    python generate_formulas.py [number=100] [depth=5] [length=15] [agents=3] [props=4]

Examples:
    python generate_formulas.py number=50 depth=3
    python generate_formulas.py number=200 depth=10 length=20
    python generate_formulas.py
"""

from formulas import generate_formulas_with_args

if __name__ == "__main__":
    generate_formulas_with_args() 