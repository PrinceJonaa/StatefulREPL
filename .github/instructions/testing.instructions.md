---
applyTo: "**/test_*.py"
---

# Testing Standards

## Framework
- Use `pytest` as the test runner
- Group tests with descriptive class names: `class TestModuleBehavior:`
- One assertion per concept; multiple asserts OK if testing one logical outcome

## Naming
```python
def test_<what>_<condition>_<expected>():
    """Tests that <what> does <expected> when <condition>."""
```

Examples:
```python
def test_score_negative_input_raises_value_error():
def test_parse_empty_string_returns_none():
def test_cache_expired_entry_triggers_refresh():
```

## Structure (Arrange-Act-Assert)
```python
def test_example():
    # Arrange
    data = create_test_data()

    # Act
    result = process(data)

    # Assert
    assert result.status == "complete"
```

## Fuzzy Assertions
For probabilistic or numeric outputs, use ranges:
```python
assert 0.3 < score < 0.9   # NOT: assert score == 0.67
assert len(results) >= 1    # NOT: assert len(results) == 42
```

## Fixtures
- Use `@pytest.fixture` for reusable test setup
- Use `tmp_path` for file system tests
- Scope fixtures appropriately: `session` for expensive, `function` (default) for isolated

## Test Organization
- Mirror the source structure: `src/mymodule/foo.py` → `tests/test_foo.py`
- Mark slow tests: `@pytest.mark.slow`
- Use `parametrize` for testing multiple inputs:
```python
@pytest.mark.parametrize("input,expected", [
    ("hello", 5),
    ("", 0),
    ("  ", 2),
])
def test_length(input, expected):
    assert len(input) == expected
```

## Common Patterns
```python
# Testing exceptions
with pytest.raises(ValueError, match="must be positive"):
    calculate(-1)

# Testing approximate equality
assert result == pytest.approx(3.14, abs=0.01)

# Temporary directories
def test_file_ops(tmp_path):
    file = tmp_path / "test.txt"
    file.write_text("data")
    assert file.read_text() == "data"
```

## Mocking
- Use `unittest.mock.patch` for external dependencies
- Mock at the boundary, not deep internals
- Prefer dependency injection over patching when possible
