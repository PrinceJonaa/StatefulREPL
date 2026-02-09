# StatefulREPL

[![Tests](https://github.com/PrinceJonaa/StatefulREPL/actions/workflows/test.yml/badge.svg)](https://github.com/PrinceJonaa/StatefulREPL/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/PrinceJonaa/StatefulREPL/branch/main/graph/badge.svg)](https://codecov.io/gh/PrinceJonaa/StatefulREPL)
[![PyPI version](https://badge.fury.io/py/stateful-repl.svg)](https://badge.fury.io/py/stateful-repl)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**A stateful AI memory and orchestration system powered by The Loom.**

StatefulREPL gives any AI persistent, layered memory (working pad → session log → wisdom base) with built-in quality measurement, loop detection, and multi-agent coordination.

## ✨ Features

- **🧠 Persistent Memory**: L1 (working), L2 (session), L3 (wisdom) — memory that survives across sessions
- **📊 7D Quality Vector**: Measure AI output quality across 7 dimensions (consistency, accuracy, stability, etc.)
- **🔍 Hallucination Detection**: 4-method ensemble for reliable truth verification
- **🔐 Sandboxed Execution**: Safe AI code execution with whitelisted imports and resource limits
- **📝 Event Sourcing**: Complete audit trail with JSON or SQLite backends
- **🌐 REST API**: FastAPI server with real-time SSE streaming
- **🤖 Multi-Model Support**: OpenAI, Anthropic, Local/Ollama adapters
- **🔄 Multi-Agent Orchestration**: Message bus, saga transactions, HALO planner, 4 agent roles, dynamic task routing

## 🚀 Quick Start

### Installation

```bash
# Basic installation
pip install stateful-repl

# With server support
pip install stateful-repl[server]

# With OpenAI support
pip install stateful-repl[openai]

# Full installation
pip install stateful-repl[all]
```

### Development Installation

```bash
git clone https://github.com/PrinceJonaa/StatefulREPL.git
cd StatefulREPL
pip install -e ".[dev,server,all]"
```

### Basic Usage

```python
from stateful_repl import LoomREPL

# Initialize with persistent state file
loom = LoomREPL(state_file="my_state.md")

# Set a goal
loom.update_l1("goal", "Analyze three research papers")

# Add working context
loom.update_l1("artifacts", ["paper1.pdf", "paper2.pdf", "paper3.pdf"])

# Log session progress
loom.append("L2", "Profiled paper1: 5 entities, 12 claims")

# Save wisdom for future sessions
loom.append("L3", {
    "scar": "Failed to extract references from scanned PDFs",
    "boon": "OCR preprocessing improved extraction 10x",
    "newrule": "Always check if PDF is text-native before parsing",
    "glyphstamp": "pdf-extraction"
})

# Validate state consistency
checks = loom.validate_state()
print(checks)
```

### Command-Line Interface

```bash
# Initialize a new state file
loom-repl init

# Set a goal
loom-repl goal "Build recommendation system"

# Check quality metrics
loom-repl quality

# View current state
loom-repl status

# Start web server
loom-repl serve --port 8000
```

### Run Examples

```bash
# Basic O-P-W-T-R cycle demo
python examples/demo_workflow.py

# Multi-session persistence demo
python examples/demo_multi_session.py
```

## 📚 Documentation

- **[Architecture Overview](agents.md)** - The Loom protocol & system design
- **[API Reference](docs/api.md)** - Complete API documentation
- **[Integration Guide](docs/prompt_integration.md)** - Using The Loom with any AI
- **[Contributing Guide](CONTRIBUTING.md)** - How to contribute
- **[Changelog](CHANGELOG.md)** - Version history

## 🏗️ Architecture

**The Loom** (interface) sits on top of **StateREPL** (runtime):

```
┌─────────────────────────────────────────────────────────┐
│  The Loom Role Prompt (any AI)                          │
│  - O-P-W-T-R cycles                                     │
│  - Stillness Gate pre-sensing                           │
│  - Anti-loop protection                                 │
└─────────────────────────────────────────────────────────┘
                        ↕
┌─────────────────────────────────────────────────────────┐
│  LoomREPL Engine (Python)                               │
│  - L1/L2/L3 memory management                           │
│  - Quality measurement                                  │
│  - Event sourcing                                       │
└─────────────────────────────────────────────────────────┘
                        ↕
┌─────────────────────────────────────────────────────────┐
│  Persistence Layer                                      │
│  - loom_state.md (human-readable)                       │
│  - SQLite event store (machine-readable)                │
└─────────────────────────────────────────────────────────┘
```

### Memory Layers

| Layer | Name | Purpose | Lifecycle |
|---|---|---|---|
| **L1** | Working Pad | Current turn: goal, constraints, artifacts | Ephemeral (cleared per session) |
| **L2** | Session Log | Conversation arc: decisions, shifts | Session-scoped (archived) |
| **L3** | Wisdom Base | Cross-session: rules, concepts, scars/boons | Permanent (accumulates) |

### Quality Dimensions (7D Vector)

1. **Internal Consistency** - Agreement across rephrased queries
2. **External Correspondence** - Accuracy vs. ground truth
3. **Temporal Stability** - Consistency over time
4. **Causal Validity** - Logical reasoning correctness
5. **Relational Integrity** - Entity relationships preserved
6. **Multiscale Alignment** - Micro/macro consistency
7. **Predictive Accuracy** - Forward prediction capability

## 🔬 Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=src/stateful_repl --cov-report=html

# Run specific test suite
python -m pytest tests/test_loom_state.py -v
```

**Current test coverage**: 179 tests passing

## 📈 Project Status

### Phase 1: MVP ✅ **COMPLETE**
- [x] Core LoomREPL engine
- [x] L1/L2/L3 memory persistence
- [x] Markdown state files
- [x] Basic CLI
- [x] Sandbox execution
- [x] Demo scripts

### Phase 2: Core Features ✅ **COMPLETE**
- [x] 7D quality vector system
- [x] Hallucination detection ensemble
- [x] Event sourcing (JSON + SQLite)
- [x] Web API with SSE streaming
- [x] Model abstraction layer
- [x] Extended CLI
- [x] Comprehensive test suite

### Phase 3: Multi-Agent ✅ **COMPLETE**
- [x] Async message bus (InProcessBus with topic-based pub/sub)
- [x] Saga transaction management (async with retry & compensation)
- [x] HALO hierarchical planning (DAG decomposition & topological scheduling)
- [x] Dynamic task routing (capability-match, round-robin, least-busy)
- [x] Agent roles (CoordinatorLoom, BuilderLoom, VerifierLoom, DistillerLoom)
- [x] 68 new Phase 3 tests

### Phase 4: Advanced 📋 **PLANNED**
- [ ] Context compression (LLMLingua)
- [ ] Predictive prefetching
- [ ] Automated calibration learning
- [ ] Time-travel debugging UI

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
# Clone and install
git clone https://github.com/PrinceJonaa/StatefulREPL.git
cd StatefulREPL
pip install -e ".[dev,server,all]"

# Run tests
python -m pytest tests/ -v

# Format code
black src/ tests/

# Lint
ruff check src/ tests/
```

## 📜 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- Inspired by research in stateful AI systems and event sourcing patterns
- Built with LangGraph, FastAPI, and modern Python tooling
- The Loom protocol draws from presence-first programming and relational thinking

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/PrinceJonaa/StatefulREPL/issues)
- **Discussions**: [GitHub Discussions](https://github.com/PrinceJonaa/StatefulREPL/discussions)
- **Security**: See [SECURITY.md](SECURITY.md) for reporting vulnerabilities

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=PrinceJonaa/StatefulREPL&type=Date)](https://star-history.com/#PrinceJonaa/StatefulREPL&Date)

---

**Made with 🧠 by the StatefulREPL community**

