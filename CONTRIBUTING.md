# Contributing to StatefulREPL

First off, thank you for considering contributing to StatefulREPL! 🎉

The Loom protocol is built on the principle of **presence-first integration** — every contribution should increase coherence without losing what already works.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Workflow](#development-workflow)
- [Pull Request Guidelines](#pull-request-guidelines)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Documentation](#documentation)
- [Community](#community)

## 📜 Code of Conduct

This project adheres to a Code of Conduct that all contributors are expected to follow. Please read [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) before contributing.

## 🚀 How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check existing issues. When creating a bug report, include:

- Clear, descriptive title
- Exact steps to reproduce
- Expected vs. actual behavior
- Code samples and error messages
- Environment details (OS, Python version, dependencies)

Use the [bug report template](.github/ISSUE_TEMPLATE/bug_report.md).

### Suggesting Features

Feature suggestions are welcome! Please:

- Check if the feature already exists or is planned ([agents.md](agents.md) § Implementation Phases)
- Explain the use case and why it's valuable
- Provide example API design if possible
- Consider backward compatibility

Use the [feature request template](.github/ISSUE_TEMPLATE/feature_request.md).

### Improving Documentation

Documentation improvements are always welcome:

- Fix typos, grammar, or unclear explanations
- Add missing API documentation
- Create tutorials or examples
- Improve docstrings and type hints

Use the [documentation issue template](.github/ISSUE_TEMPLATE/documentation.md).

### Code Contributions

Areas where we especially welcome contributions:

- **Phase 3 Multi-Agent**: Saga transactions, HALO planning, agent roles
- **Quality Dimensions**: New quality measurement strategies
- **Model Adapters**: Support for additional LLM providers
- **Performance**: Optimization, benchmarking, profiling
- **Testing**: Increase coverage, add edge cases
- **Examples**: Real-world use cases and tutorials

## 🛠️ Development Workflow

### 1. Set Up Development Environment

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/StatefulREPL.git
cd StatefulREPL

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in development mode with all extras
pip install -e ".[dev,server,all]"

# Install pre-commit hooks (recommended)
pip install pre-commit
pre-commit install
```

### 2. Create a Branch

```bash
# Update main
git checkout main
git pull origin main

# Create a feature branch
git checkout -b feature/your-feature-name
# Or for bug fixes:
git checkout -b fix/issue-description
```

### 3. Make Changes

Follow the [Coding Standards](#coding-standards) and ensure:

- Code is clean, readable, and well-documented
- Type hints are used throughout
- Tests are added for new functionality
- Existing tests still pass

### 4. Run Tests and Checks

```bash
# Format code
black src/ tests/

# Lint
ruff check src/ tests/ --fix

# Type check
mypy src/stateful_repl --ignore-missing-imports

# Run all tests
pytest tests/ -v

# Run tests with coverage
pytest tests/ --cov=src/stateful_repl --cov-report=html
```

### 5. Commit Your Changes

Use conventional commit messages:

```bash
git add .
git commit -m "feat: add hierarchical task planner"
# Or:
git commit -m "fix: resolve event store race condition"
git commit -m "docs: update API reference for quality module"
git commit -m "test: add tests for hallucination detection"
```

**Commit message format**:
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation
- `test:` - Tests
- `refactor:` - Code refactoring
- `perf:` - Performance improvement
- `chore:` - Maintenance tasks

### 6. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

Then create a PR on GitHub using the [PR template](.github/PULL_REQUEST_TEMPLATE.md).

## ✅ Pull Request Guidelines

### Before Submitting

- [ ] Code follows project style (black, ruff)
- [ ] Type hints added
- [ ] Tests added and passing
- [ ] Documentation updated
- [ ] CHANGELOG.md updated (if user-facing change)
- [ ] Commit messages follow conventional format
- [ ] Branch is up to date with main

### PR Review Process

1. **Automated Checks**: CI must pass (tests, linting, type checking)
2. **Code Review**: At least one maintainer approval required
3. **Testing**: Reviewers may test your changes locally
4. **Iteration**: Address review feedback promptly
5. **Merge**: Maintainer will merge when ready

### PR Best Practices

- **Keep PRs focused**: One feature/fix per PR
- **Small commits**: Easier to review and revert
- **Clear descriptions**: Explain what, why, and how
- **Respond to feedback**: Discussion is part of the process
- **Be patient**: Reviews take time, but we value quality

## 🎨 Coding Standards

### Python Style

- **PEP 8** compliance (enforced by black and ruff)
- **Type hints**: All functions and methods
- **Docstrings**: Google style for all public APIs
- **Line length**: 100 characters (black default)

### Code Organization

```python
"""Module docstring explaining purpose.

Example:
    Basic usage example here.
"""

from __future__ import annotations  # For forward references

import standard_library
from typing import Type, List, Optional

import third_party_package

from stateful_repl import internal_module


class ExampleClass:
    """Brief class description.
    
    Longer explanation if needed.
    
    Args:
        param1: Description
        param2: Description
    
    Attributes:
        attr1: Description
        attr2: Description
    """
    
    def __init__(self, param1: str, param2: int) -> None:
        """Initialize ExampleClass."""
        self.attr1 = param1
        self.attr2 = param2
    
    def public_method(self, arg: str) -> bool:
        """Public method with docstring.
        
        Args:
            arg: Input parameter
        
        Returns:
            True if successful, False otherwise
        
        Raises:
            ValueError: If arg is empty
        """
        if not arg:
            raise ValueError("arg cannot be empty")
        return True
    
    def _private_method(self) -> None:
        """Private method (still documented)."""
        pass
```

### Loom Protocol Conventions

When contributing to Loom-related code:

1. **Bounded Claims**: Claims must include falsification criteria
2. **Smallest Vertical Slice**: Implement the minimum that works end-to-end
3. **Strategy Pattern**: Use pluggable components for extensibility
4. **Event Sourcing**: Emit events, don't mutate silently
5. **Surgical Edits**: No global rewrites unless creating new files

## 🧪 Testing

### Test Organization

```
tests/
├── test_loom_state.py      # Core state engine tests
├── test_quality.py          # Quality dimension tests
├── test_hallucination.py    # Hallucination detection tests
├── test_events.py           # Event sourcing tests
├── test_sandbox.py          # Sandbox security tests
└── test_server.py           # API endpoint tests
```

### Writing Tests

```python
import pytest
from stateful_repl import LoomREPL


class TestLoomState:
    """Test suite for LoomREPL state management."""
    
    def test_update_l1_goal(self):
        """Test updating L1 goal field."""
        loom = LoomREPL()
        loom.update_l1("goal", "Test goal")
        
        state = loom.read_state("L1")
        assert state["goal"] == "Test goal"
    
    def test_invalid_layer_raises_error(self):
        """Test that invalid layer name raises ValueError."""
        loom = LoomREPL()
        
        with pytest.raises(ValueError, match="Invalid layer"):
            loom.read_state("L99")
```

### Test Requirements

- **Unit tests**: Test individual functions/methods
- **Integration tests**: Test component interactions
- **Edge cases**: Boundary conditions, empty inputs, invalid data
- **Error cases**: Exceptions and error handling
- **Coverage**: Aim for >80% coverage on new code

### Running Tests

```bash
# All tests
pytest tests/ -v

# Specific test file
pytest tests/test_quality.py -v

# Specific test
pytest tests/test_quality.py::TestQualityEvaluator::test_compute_vector -v

# With coverage
pytest tests/ --cov=src/stateful_repl --cov-report=html

# Fast tests only (skip slow integration tests)
pytest tests/ -m "not slow"

# Watch mode (requires pytest-watch)
ptw tests/
```

## 📚 Documentation

### Docstring Style (Google Format)

```python
def function_name(param1: str, param2: int = 10) -> bool:
    """Brief one-line description.
    
    More detailed explanation if needed. Can span multiple
    paragraphs and include examples.
    
    Args:
        param1: Description of first parameter
        param2: Description with default value (default: 10)
    
    Returns:
        Description of return value
    
    Raises:
        ValueError: When param1 is empty
        TypeError: When param2 is negative
    
    Example:
        >>> function_name("test", 20)
        True
    """
    pass
```

### Documentation Files

- **README.md**: High-level overview, quick start
- **agents.md**: Architecture, roles, implementation phases
- **docs/api.md**: Complete API reference
- **docs/prompt_integration.md**: Integration guides
- **CHANGELOG.md**: Version history and changes

## 👥 Community

### Getting Help

- **GitHub Discussions**: For questions and general discussion
- **GitHub Issues**: For bugs and feature requests
- **Code Reviews**: Learning opportunity, not gatekeeping

### Communication Style

Follow The Loom principles in all interactions:

- **Presence-first**: Read and understand context before responding
- **Bounded claims**: State assumptions and limitations
- **Hold paradoxes**: Multiple perspectives can coexist
- **Verify**: Back up claims with evidence or tests

### Recognition

Contributors are:
- Listed in commit history
- Mentioned in release notes (for significant contributions)
- Welcomed to join as maintainers (for sustained contributions)

## 🎯 Current Priorities

Check [agents.md](agents.md) § Working Document for current focus areas.

**Phase 3 priorities:**
1. Saga transaction management
2. HALO hierarchical planning
3. Agent role implementations
4. Inter-agent communication

## ❓ Questions?

- Open a [GitHub Discussion](https://github.com/PrinceJonaa/StatefulREPL/discussions)
- Review [agents.md](agents.md) for architecture details
- Check existing issues and PRs

---

**Thank you for contributing to StatefulREPL!** 🙏

By participating, you help build a coherent, stateful foundation for AI systems that can think, remember, and evolve.
