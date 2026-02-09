# Skill: Quality Check (7D Quality Vector)

## Purpose
Evaluate any output across 7 orthogonal quality dimensions to get a holistic assessment of reliability and correctness.

## The 7 Dimensions

| # | Dimension | What It Measures | Key Question |
|---|---|---|---|
| 1 | Internal Consistency (IC) | Agreement within the output | "Does this contradict itself?" |
| 2 | External Correspondence (EC) | Alignment with known facts / sources | "Is this factually accurate?" |
| 3 | Temporal Stability (TS) | Consistency across rephrased queries | "Would I get the same answer if asked differently?" |
| 4 | Causal Validity (CV) | Logical cause-effect reasoning | "Do the causal claims hold up?" |
| 5 | Relational Integrity (RI) | Entity relationships preserved | "Are the connections between things correct?" |
| 6 | Multiscale Alignment (MA) | Micro-macro consistency | "Do the details match the big picture?" |
| 7 | Predictive Accuracy (PA) | Ability to make verifiable predictions | "Can I test this claim?" |

## How to Use

### Quick Check (3 dimensions)
For routine outputs, check IC + EC + CV:
1. **IC**: Scan for self-contradictions
2. **EC**: Verify key facts against sources
3. **CV**: Check that "because X, therefore Y" claims are sound

### Full Check (all 7)
For critical outputs (architecture decisions, public docs, production code):
1. Score each dimension 0.0–1.0
2. Flag any dimension below 0.5 as a concern
3. Report the quality vector: `[IC, EC, TS, CV, RI, MA, PA]`

### Scoring Guidelines
- **0.0–0.3**: Major issues — output unreliable on this dimension
- **0.3–0.6**: Concerns — needs review or additional sources
- **0.6–0.8**: Acceptable — minor issues may exist
- **0.8–1.0**: Strong — high confidence on this dimension

## Integration with .loom/
- Record quality assessments in `oracle-matrix.md` as ORACLE entries
- Link quality findings to claims in `claim-ledger.md`
- Flag quality failures as potential entries in `paradox-queue.md`

## When to Invoke
- After any Builder produces an artifact
- Before merging or finalizing documentation
- When reviewing code changes
- When integrating multiple sources (Distiller work)
