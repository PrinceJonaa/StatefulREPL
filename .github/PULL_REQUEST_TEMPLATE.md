## Description

<!-- Provide a clear and concise description of your changes -->

Fixes #<!-- issue number -->

## Type of Change

<!-- Check all that apply -->

- [ ] 🐛 Bug fix (non-breaking change which fixes an issue)
- [ ] ✨ New feature (non-breaking change which adds functionality)
- [ ] 💥 Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] 📝 Documentation update
- [ ] 🎨 Code style/refactoring (no functional changes)
- [ ] ⚡ Performance improvement
- [ ] ✅ Test coverage improvement
- [ ] 🔧 Infrastructure/tooling change

## Loom Protocol Compliance

<!-- The Loom requires presence, verification, and bounded claims -->

### Claims Made

<!-- List key claims with scope, confidence, and falsification criteria -->

```
CLAIM-001: "<statement>" [scope: local|module|system] [confidence: 0.0-1.0]
  falsifies: "<what would prove this wrong>"

CLAIM-002: ...
```

### Contracts Defined/Modified

<!-- If you're changing interfaces, document preconditions/postconditions/invariants -->

```
CONTRACT-001: <function/class name>
  PRE: <what must be true before>
  POST: <what will be true after>
  INV: <what must remain true throughout>
```

### Oracle (Verification)

<!-- How can we verify this works? What tests prove correctness? -->

- [ ] Test command: `python -m pytest tests/test_<module>.py -v`
- [ ] Expected coverage: >X% for new code
- [ ] Manual verification steps: ...

## Quality Vector Impact

<!-- How does this PR affect the 7D quality dimensions? -->

| Dimension | Impact | Notes |
|-----------|--------|-------|
| IC: Internal Consistency | ✅ Improved / ➖ No change / ⚠️ Degraded | <!-- brief explanation --> |
| EC: External Correspondence | ✅ / ➖ / ⚠️ | |
| TS: Temporal Stability | ✅ / ➖ / ⚠️ | |
| CV: Causal Validity | ✅ / ➖ / ⚠️ | |
| RI: Relational Integrity | ✅ / ➖ / ⚠️ | |
| MA: Multiscale Alignment | ✅ / ➖ / ⚠️ | |
| PA: Predictive Accuracy | ✅ / ➖ / ⚠️ | |

## Testing Checklist

<!-- Confirm you've done these steps -->

- [ ] All existing tests pass (`pytest tests/ -v`)
- [ ] Added tests for new functionality (if applicable)
- [ ] Test coverage maintained or improved
- [ ] Manual testing performed (describe below if needed)
- [ ] Edge cases considered and tested

### Manual Testing Notes

<!-- Describe any manual testing you performed -->

```bash
# Commands run and their output
```

## Documentation Requirements

<!-- Check all that apply -->

- [ ] Updated docstrings for modified functions/classes
- [ ] Updated `agents.md` if architecture changed
- [ ] Updated README.md if user-facing changes
- [ ] Added code comments for complex logic
- [ ] Updated CLI help text (if applicable)
- [ ] Added example/demo if new feature

## Breaking Changes

<!-- If this PR introduces breaking changes, describe them here -->

### What breaks?

<!-- Be specific about what existing code will no longer work -->

### Migration Path

<!-- How should users update their code? -->

```python
# Before
old_api()

# After
new_api()
```

### Deprecation Notice

<!-- If deprecating gradually, what's the timeline? -->

- Deprecated in: v0.X.0
- Removal planned for: v0.Y.0

## Performance Impact

<!-- Has this change affected performance? -->

- [ ] No performance impact
- [ ] Performance improved (describe below)
- [ ] Performance degraded (justify and describe mitigation)

<!-- If applicable, include benchmark results -->

## Checklist

<!-- Final checks before merging -->

- [ ] My code follows the project's style guidelines (runs `ruff format`)
- [ ] I have performed a self-review of my code
- [ ] I have commented complex or non-obvious code
- [ ] My changes generate no new warnings
- [ ] I have added tests that prove my fix/feature works
- [ ] New and existing tests pass locally
- [ ] Any dependent changes have been merged and published
- [ ] I have updated the documentation accordingly
- [ ] I have read and followed the Loom protocol (agents.md)

## Additional Context

<!-- Add any other context, screenshots, or information about the PR -->

---

<!-- 
For Reviewers:
1. Verify claims are bounded and testable
2. Check oracle (test) coverage
3. Validate contracts (PRE/POST/INV) if interfaces changed
4. Confirm quality vector assessment matches code
5. Ensure no loops/rushes/echo chambers in implementation
-->
