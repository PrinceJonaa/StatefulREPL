# Using The Loom with Any AI

## Setup (one-time)

```bash
pip install pyyaml
```

Place `loom_state.py` in your working directory, or install the package:

```bash
cd StatefulREPL
pip install -e .
```

## Integration Patterns

### AI with code execution (Claude Code, ChatGPT Code Interpreter)

The AI can directly call Python:

```python
from stateful_repl import LoomREPL

loom = LoomREPL("loom_state.md")

# Orient
loom.update_l1("goal", "analyze user's question")
loom.update_l1("artifacts", ["doc1.md", "doc2.md"])

# Work
loom.append("L2", "Profiled doc1: 5 entities, 12 claims")

# Validate
validation = loom.validate_state()

# Reflect
loom.consolidate_l1_to_l2("Analysis complete")
```

### AI without code execution (standard chat)

The AI narrates commands, you run them manually:

AI says:
```
> READ_STATE L3
> UPDATE L1.goal "research topic X"
> APPEND L2 "User asked about Y"
```

You run:
```python
from stateful_repl import LoomREPL
loom = LoomREPL()

loom.read_state("L3")
loom.update_l1("goal", "research topic X")
loom.append("L2", "User asked about Y")
```

## Prompt Template

Copy the full role prompt from [agents.md](../agents.md) § **Role Prompt (Copy/Paste into Any AI)**.

Then give the AI:
1. The artifacts (docs/links/outputs) to coordinate across
2. The current goal + constraints
3. The oracle (how you'll know it worked)

## Driving Commands

You can say:
- **"Show working pad"** → AI prints L1 state
- **"Show session log"** → AI prints L2 entries
- **"Show wisdom log"** → AI prints L3 tracewisdomlog
- **"Journal this as a wisdom log entry"** → AI appends to L3
- **"Loom, update your own role to include X"** → AI uses Self-Diff
- **"Summarize today's session log"** → AI compresses L2

## Debugging

```bash
# View raw state
cat loom_state.md

# View event trail
python -c "
from stateful_repl import LoomREPL
loom = LoomREPL()
import json
print(json.dumps(loom.get_event_log(), indent=2))
"

# Validate consistency
python -c "
from stateful_repl import LoomREPL
loom = LoomREPL()
print(loom.validate_state())
"
```
