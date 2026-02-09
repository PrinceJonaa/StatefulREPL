---
agent: agent
tools: ['agent', 'read', 'search', 'todo']
description: "Generate a structured handoff packet for passing work between agents"
---

# Agent Handoff Packet

Generate a structured handoff packet for transferring work context between agents.

## When to Use
- Coordinator routing work to Builder/Verifier/Distiller
- Builder passing completed work to Verifier for validation
- Any agent that needs to pass context to another agent

## Handoff Format

Produce a packet in this exact format:

```
HANDOFF PACKET
==============
From: <source agent>
To: <target agent>
Timestamp: <ISO 8601>

Goal: <one sentence — what the receiving agent must accomplish>

Inputs:
  - <artifact ID or file path>
  - <artifact ID or file path>

Claims touched:
  - <CLAIM-ID>: <brief description>

Contracts touched:
  - <CONTRACT-ID>: <brief description>

Oracles:
  - <test ID or description> — run: <command or check>

Open paradoxes:
  - <PARADOX-ID or "none">

Context:
  <2-3 sentences of essential context the receiver needs>

Next action:
  <concrete first step for the receiving agent>

Success criteria:
  <how the receiver knows they're done>
```

## Steps

1. **Identify the goal** — What does the receiver need to do?
2. **Gather inputs** — What artifacts, files, or context are needed?
3. **Map claims/contracts** — What has been claimed or contracted about this work?
4. **Define success** — How will completion be verified?
5. **Write the packet** — Fill in the format above completely

## Rules
- Every field must be filled (use "none" if truly empty)
- The Goal must be a single actionable sentence
- Next Action must be concrete enough to start immediately
- Include enough context that the receiver doesn't need to ask questions
