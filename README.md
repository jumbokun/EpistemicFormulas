# Epistemic Formula Generator

This project generates epistemic logic formulas with support for nested K operators (knowledge operators).

## File Structure

- `config.py` - Configuration file containing default parameters and preset configurations
- `formulas.py` - Core generation logic
- `generate_formulas.py` - Command line script
- `README.md` - Documentation

## Usage

### 1. Basic Usage

```bash
# Generate 10 formulas with default configuration
python generate_formulas.py

# Generate 100 formulas with maximum depth of 5
python generate_formulas.py number=100 depth=5

# Generate 50 formulas with maximum depth of 3 and maximum length of 15
python generate_formulas.py number=50 depth=3 length=15
```

### 2. Command Line Parameters

- `number` - Number of formulas to generate (default: 10)
- `depth` - Maximum nesting depth (K operator nesting levels, default: 10)
- `length` - Maximum formula length (number of operators, default: 100)
- `agents` - Number of agents (default: 3)
- `props` - Number of propositional variables (default: 10)

### 3. Modifying Default Configuration

Edit the `config.py` file to modify default settings:

```python
# Modify default configuration
DEFAULT_CONFIG = EpistemicConfig(
    num_agents=4,          # Number of agents
    max_depth=5,           # Maximum nesting depth
    max_length=12,         # Maximum formula length
    num_props=5,           # Number of propositional variables
    prop_weight=0.3,       # Propositional variable weight
    neg_weight=0.2,        # Negation weight
    binary_weight=0.3,     # Binary connective weight
    knowledge_weight=0.2   # Knowledge operator weight
)
```

### 4. Avoiding Duplicates and Generating Complex Formulas

The generator includes deduplication to avoid repeated formulas. To generate more complex formulas:

1. **Reduce proposition weight** (e.g., 0.05-0.15)
2. **Increase binary connective weight** (e.g., 0.35-0.45)
3. **Increase knowledge operator weight** (e.g., 0.25-0.35)
4. **Use predefined complex configurations**:
   ```python
   from config import COMPLEX_CONFIG, VERY_COMPLEX_CONFIG
   ```

**Note**: The `length` parameter refers to the number of operators, not character count. A formula with 10 operators can be quite complex.

### 5. Generating Deep Nested Formulas

To generate formulas with high K-operator nesting depth:

1. **Increase knowledge_weight significantly** (e.g., 0.5-0.7)
2. **Reduce other weights** to favor K operators
3. **Use moderate max_length** to allow depth without hitting length limits
4. **Increase num_agents** for more variety

Example configuration for deep formulas:
```python
config = EpistemicConfig(
    max_depth=10,
    max_length=30,
    knowledge_weight=0.7,  # High K operator probability
    prop_weight=0.05,      # Low proposition probability
    binary_weight=0.1,     # Low binary connective probability
    neg_weight=0.15        # Low negation probability
)
```

## Output Format

The program generates a JSON file containing:

1. **Generation time**
2. **Run parameters**
3. **Formula list**, each formula containing:
   - Formula string
   - Statistics (depth, operator count, etc.)

## Formula Types

Generated formulas include the following elements:

- **Propositional variables**: p, q, r, s, ...
- **Negation**: ¬φ
- **Binary connectives**: ∧ (and), ∨ (or), → (implies)
- **Knowledge operators**: K_i(φ) means agent i knows φ

## Depth Calculation

Depth only counts K operator nesting levels:
- `K_1(p)` depth = 1
- `K_1(K_2(p))` depth = 2
- `K_1(K_2(K_3(p)))` depth = 3

Negation and binary connectives do not increase depth.

## Example Output

```json
{
  "generation_time": "2024-01-15T10:30:00",
  "parameters": {
    "num_agents": 3,
    "max_depth": 3,
    "max_length": 10,
    "num_props": 4,
    "num_formulas": 5
  },
  "formulas": [
    {
      "id": 1,
      "formula": "K_1(K_2(p))",
      "stats": {
        "max_depth": 2,
        "total_operators": 2,
        "knowledge_ops": 2
      }
    }
  ]
}
``` 