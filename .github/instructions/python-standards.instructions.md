---
applyTo: "**/*.py"
---

# Python Standards

## Type Hints
- All public functions and methods MUST have type hints
- Use `Optional[X]` for nullable parameters
- Use `list[X]`, `dict[K, V]` (lowercase, Python 3.9+)
- Return types always specified

## Docstrings (Google Style)
```python
def function_name(param: str, count: int = 0) -> bool:
    """One-line summary.

    Extended description if needed.

    Args:
        param: Description without repeating the type.
        count: What this controls. Defaults to 0.

    Returns:
        True if the operation succeeded.

    Raises:
        ValueError: If param is empty.
    """
```

## Import Order
```python
# 1. Standard library
import json
from pathlib import Path
from typing import Optional

# 2. Third-party
import requests

# 3. Local
from myproject.module import MyClass
```

## Error Handling
- Never use bare `except:` — always catch specific exceptions
- Raise `ValueError` for invalid inputs, `TypeError` for wrong types
- Include context in error messages: `f"Invalid score {score}: must be 0.0-1.0"`

## Patterns
- Strategy pattern for pluggable components
- Factory functions for backend / configuration selection
- Prefer `pathlib.Path` over `os.path`
- No `os.system()` or `subprocess` without explicit justification

## Claim Comments
Mark architectural decisions with traceable IDs:
```python
# CLAIM-XX: "Statement" [scope: module] [confidence: 0.9] [falsifies: "condition"]
```
