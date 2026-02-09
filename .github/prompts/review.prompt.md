---
agent: agent
tools: ['agent', 'read', 'search', 'execute', 'todo']
description: "Review code changes using the 7D quality vector"
---

# Code Review with Quality Vector

Review the current changes using the 7D quality framework.

## Steps

1. **Gather the changes:**
   - Read the modified files
   - Understand the intent of each change

2. **Score each quality dimension (0.0–1.0):**

   | Dimension | Score | Notes |
   |---|---|---|
   | Internal Consistency | | Does it contradict itself or existing code? |
   | External Correspondence | | Do facts/APIs/references check out? |
   | Temporal Stability | | Would this hold up if requirements shift slightly? |
   | Causal Validity | | Are the cause-effect assumptions sound? |
   | Relational Integrity | | Are entity relationships preserved? |
   | Multiscale Alignment | | Do details match the big picture? |
   | Predictive Accuracy | | Can the claims be tested? |

3. **Check for anti-patterns:**
   - Global rewrite when surgical edit would suffice?
   - Missing error handling?
   - Untested code paths?
   - Hardcoded values that should be configurable?
   - Missing type hints or documentation?

4. **Produce a Review Report:**

```
REVIEW
======
Quality Vector: [IC, EC, TS, CV, RI, MA, PA]
Overall: <PASS / NEEDS_WORK / REJECT>

Strengths:
  - <what's good>

Concerns:
  - <dimension>: <specific issue>

Recommendations:
  1. <actionable fix>
  2. <actionable fix>

Claims to record:
  CLAIM-XX: "<claim>" [scope: module] [confidence: X.X] [falsifies: "..."]
```

## Rules
- Be specific — "line X has issue Y", not "code could be better"
- Flag dimensions below 0.5 as blocking concerns
- Link findings to existing claims/contracts in .loom/ when applicable
