---
applyTo: "**/*.md"
---

# Documentation Standards

## Markdown Conventions
- Use ATX-style headers (`#`, `##`, `###`)
- One blank line before and after headers
- Use fenced code blocks with language identifiers: ` ```python `, ` ```yaml `, ` ```bash `
- Use `>` blockquotes for important callouts
- Tables must have aligned columns

## Document Structure
```markdown
# Title

> One-line purpose statement.

## Section

Content...

### Subsection

Details...
```

## .loom/ File Conventions
- Every `.loom/` file has a `# Title` and a `## Format` section explaining the entry structure
- Use consistent ID prefixes:
  - `ART-XXX` for artifacts
  - `CLAIM-XXX` for claims
  - `CONTRACT-XXX` for contracts
  - `ORACLE-XXX` for test oracles
  - `PARADOX-XXX` for paradoxes
  - `SCAR-XXX` for trace wisdom entries
- IDs are sequential within their file
- Entries are append-only in normal operation

## Cross-References
- Link to `.loom/` entries by ID: "See CLAIM-005", "Per CONTRACT-003"
- Link to files with relative paths: `[module.py](src/module.py)`
- Reference agents with `@agent-name` notation

## Status Markers
- ✅ Complete / Verified
- 🟡 In Progress / Partial
- 🔴 Not Started / Blocked
- ⚠️ Warning / Needs Attention
- ❌ Failed / Rejected

## Changelog Entries
When updating living documents, add dated entries:
```markdown
### YYYY-MM-DD — Change Description
- What changed and why
- Impact on other documents
```
