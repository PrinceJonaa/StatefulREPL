# StatefulREPL

**A stateful AI memory and orchestration system powered by The Loom.**

StatefulREPL gives any AI persistent, layered memory (working pad → session log → wisdom base) with built-in quality measurement, loop detection, and multi-agent coordination.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the demo
python examples/demo_workflow.py

# Check generated state
cat loom_state.md
```

## Project Structure

```
StatefulREPL/
├── agents.md                  # The Loom role prompt + all agent definitions
├── README.md                  # This file
├── requirements.txt           # Python dependencies
├── src/
│   └── stateful_repl/
│       ├── __init__.py
│       ├── loom_state.py      # Core LoomREPL — L1/L2/L3 state engine
│       ├── sandbox.py         # Restricted execution environment
│       ├── quality.py         # Quality vector computation (Phase 2)
│       ├── events.py          # Event sourcing store (Phase 2)
│       └── orchestrator.py    # Multi-agent saga coordination (Phase 3)
├── examples/
│   ├── demo_workflow.py       # Basic O-P-W-T-R cycle demo
│   └── demo_multi_session.py  # Cross-session persistence demo
├── tests/
│   ├── __init__.py
│   ├── test_loom_state.py     # Core state tests
│   └── test_sandbox.py        # Sandbox security tests
├── loom_archives/             # Archived L2 session logs (auto-created)
└── docs/
    └── prompt_integration.md  # How to use The Loom with any AI
```

## Architecture

See [agents.md](agents.md) for the full role definitions and architecture mapping.

**The Loom** (interface) sits on top of **StateREPL** (runtime):

```
The Loom Role Prompt ←→ LoomREPL Python Engine ←→ loom_state.md (persistence)
```

### Memory Layers

| Layer | Name | Purpose | Persistence |
|---|---|---|---|
| L1 | Working Pad | Current turn: goal, constraints, artifacts | In-memory + file |
| L2 | Session Log | Conversation arc: decisions, shifts | Append-only log |
| L3 | Wisdom Base | Cross-session: rules, concepts, scars/boons | Permanent store |

### Core Commands

```python
from stateful_repl import LoomREPL

loom = LoomREPL()

# Read
loom.read_state("L1")            # Current working context
loom.read_state("L3")            # Accumulated wisdom

# Write
loom.update_l1("goal", "integrate 3 docs")
loom.append("L2", "Profiled doc: 5 entities, 12 claims")
loom.append("L3", {"scar": "...", "boon": "...", "newrule": "...", "glyphstamp": "..."})

# Consolidate
loom.consolidate_l1_to_l2()      # Compress working pad → session log
loom.consolidate_l2_to_l3()      # Extract patterns → wisdom base

# Validate
loom.validate_state()             # Check internal consistency
```

## Current Phase

**Phase 1: MVP** — File-based persistence, L1/L2/L3 memory, event logging, sandbox execution.

See [agents.md](agents.md) § Implementation Phases for the full roadmap.

## License

MIT
