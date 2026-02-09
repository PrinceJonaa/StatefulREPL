# Skill: Hallucination Check

## Purpose
Detect fabricated, unsupported, or misleading content in AI outputs using a multi-method ensemble approach.

## The 4 Detection Methods

### 1. Structural Analysis
Check the output's internal structure for hallucination signals:
- **Vague hedging**: Excessive use of "might", "could", "possibly" without commitment
- **Specificity without source**: Very specific claims (numbers, dates, names) with no citation
- **Contradictory statements**: Asserting X and not-X in the same output
- **Confidence inflation**: "Definitely", "always", "never" on uncertain topics

### 2. Source Alignment
Compare output against available sources:
- Does every factual claim trace to a provided source?
- Are quotes accurate (not paraphrased as if quoted)?
- Are source relationships preserved (not mixing up which source said what)?

### 3. Consistency Probing
Test stability across reformulations:
- Ask the same question in different ways
- Check if the answer changes substantively
- High variance = low confidence = possible hallucination

### 4. Predictive Testing
Generate testable predictions from claims:
- "If X is true, then Y should also be true"
- Check whether Y holds
- Failed predictions indicate unreliable claims

## Severity Levels

| Level | Description | Action |
|---|---|---|
| **None** | No hallucination signals detected | Proceed normally |
| **Low** | Minor hedging or vague claims | Note but accept |
| **Medium** | Specific unsupported claims | Verify before using |
| **High** | Contradictions or fabricated details | Reject and regenerate |
| **Critical** | Confident falsehoods | Flag immediately, escalate |

## How to Use

### Quick Scan
1. Check for specificity without source (Method 1)
2. Verify 2-3 key facts against known sources (Method 2)
3. If either fails, escalate to full check

### Full Check
1. Run all 4 methods
2. Score overall confidence: 0.0 (certain hallucination) to 1.0 (no hallucination)
3. Document findings in `oracle-matrix.md`

## Integration with .loom/
- Record findings as ORACLE entries with method details
- Update `claim-ledger.md` confidence scores based on results
- Add fabricated claims to `paradox-queue.md` for investigation

## When to Invoke
- When AI-generated content will be used in production
- When integrating outputs from multiple AI sources
- When an output "feels too specific" or "too confident"
- Before publishing documentation or making architectural decisions
