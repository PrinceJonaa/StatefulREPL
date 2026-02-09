#!/usr/bin/env python3
"""
loom — CLI for interacting with the LoomREPL state engine.

Usage:
    loom init                           Create a fresh loom_state.md
    loom status                         Show current state summary
    loom read <L1|L2|L3|ALL>            Print a memory layer
    loom goal <text>                    Set L1 goal
    loom artifact <name>                Add an artifact to L1
    loom constraint <text>              Add a constraint to L1
    loom question <text>                Add an open question to L1
    loom log <message>                  Append to L2 session log
    loom rule <id> <when> <then> <why>  Add a rule to L3
    loom scar <scar> <boon> <glyph>     Add a wisdom entry to L3
    loom consolidate <L1|L2>            Consolidate L1→L2 or L2→L3
    loom validate                       Run state consistency checks
    loom events                         Show the event log
    loom events --save [path]           Save event log to JSON
    loom clear                          Clear L1 working pad
"""

import argparse
import json
import sys
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from stateful_repl.loom_state import LoomREPL


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="loom",
        description="CLI for the LoomREPL stateful memory engine.",
    )
    p.add_argument(
        "-f", "--file",
        default="loom_state.md",
        help="Path to state file (default: loom_state.md)",
    )
    sub = p.add_subparsers(dest="command")

    # init
    sub.add_parser("init", help="Create a fresh state file")

    # status
    sub.add_parser("status", help="Show state summary")

    # read
    r = sub.add_parser("read", help="Read a memory layer")
    r.add_argument("layer", choices=["L1", "L2", "L3", "ALL", "METADATA"])

    # goal
    g = sub.add_parser("goal", help="Set L1 goal")
    g.add_argument("text", nargs="+")

    # artifact
    a = sub.add_parser("artifact", help="Add artifact to L1")
    a.add_argument("name", nargs="+")

    # constraint
    c = sub.add_parser("constraint", help="Add constraint to L1")
    c.add_argument("text", nargs="+")

    # question
    q = sub.add_parser("question", help="Add open question to L1")
    q.add_argument("text", nargs="+")

    # log
    lg = sub.add_parser("log", help="Append to L2 session log")
    lg.add_argument("message", nargs="+")

    # rule
    rl = sub.add_parser("rule", help="Add rule to L3")
    rl.add_argument("id", help="Rule identifier")
    rl.add_argument("when", help="Trigger condition")
    rl.add_argument("then", help="Action to take")
    rl.add_argument("why", help="Justification")

    # scar
    sc = sub.add_parser("scar", help="Add wisdom entry to L3")
    sc.add_argument("scar_text", help="What failed or hurt")
    sc.add_argument("boon", help="What coherence increased")
    sc.add_argument("glyph", help="Symbolic name")
    sc.add_argument("--newrule", default="", help="Rule to adopt")
    sc.add_argument("--feedback", default="", help="User feedback")

    # consolidate
    co = sub.add_parser("consolidate", help="Consolidate layers")
    co.add_argument("source", choices=["L1", "L2"])
    co.add_argument("-s", "--summary", default=None, help="Custom summary (L1→L2)")

    # validate
    sub.add_parser("validate", help="Run consistency checks")

    # events
    ev = sub.add_parser("events", help="Show or save event log")
    ev.add_argument("--save", nargs="?", const="auto", default=None, help="Save to file")

    # clear
    sub.add_parser("clear", help="Clear L1 working pad")

    # quality (Phase 2)
    sub.add_parser("quality", help="Compute and display 7D quality vector")

    # serve (Phase 2)
    sv = sub.add_parser("serve", help="Start the web API server")
    sv.add_argument("--host", default="127.0.0.1", help="Bind host")
    sv.add_argument("--port", type=int, default=8000, help="Bind port")
    sv.add_argument("--reload", action="store_true", help="Auto-reload on changes")

    # agents (Phase 3)
    ag = sub.add_parser("agents", help="Show Phase 3 agent system status")
    ag.add_argument("--plan", default=None, help="Create a plan for a goal")

    return p


def main(argv=None):
    parser = make_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return

    loom = LoomREPL(args.file)

    if args.command == "init":
        print(f"Initialized {args.file}")

    elif args.command == "status":
        l1 = loom.read_state("L1")
        l2_count = len(loom.state["L2"])
        l3 = loom.read_state("L3")
        meta = loom.read_state("METADATA")
        print(f"Session:      {meta['session_id']}")
        print(f"Last modified: {meta['last_modified']}")
        print(f"L1 Goal:      {l1['goal'] or '(none)'}")
        print(f"L1 Artifacts: {len(l1['artifacts'])}")
        print(f"L2 Entries:   {l2_count}")
        print(f"L3 Rules:     {len(l3['rules'])}")
        print(f"L3 Concepts:  {len(l3['concepts'])}")
        print(f"L3 Scars:     {len(l3['tracewisdomlog'])}")

    elif args.command == "read":
        loom.print_state(args.layer)

    elif args.command == "goal":
        text = " ".join(args.text)
        loom.update_l1("goal", text)
        print(f"L1 goal set: {text}")

    elif args.command == "artifact":
        name = " ".join(args.name)
        current = loom.read_state("L1")["artifacts"]
        current.append(name)
        loom.update_l1("artifacts", current)
        print(f"L1 artifact added: {name}")

    elif args.command == "constraint":
        text = " ".join(args.text)
        current = loom.read_state("L1")["constraints"]
        current.append(text)
        loom.update_l1("constraints", current)
        print(f"L1 constraint added: {text}")

    elif args.command == "question":
        text = " ".join(args.text)
        current = loom.read_state("L1")["open_questions"]
        current.append(text)
        loom.update_l1("open_questions", current)
        print(f"L1 question added: {text}")

    elif args.command == "log":
        msg = " ".join(args.message)
        loom.append("L2", msg)
        print(f"L2 logged: {msg}")

    elif args.command == "rule":
        loom.append("L3", {
            "rule_id": args.id,
            "when": args.when,
            "then": args.then,
            "why": args.why,
        })
        print(f"L3 rule added: {args.id}")

    elif args.command == "scar":
        entry = {
            "scar": args.scar_text,
            "boon": args.boon,
            "glyphstamp": args.glyph,
        }
        if args.newrule:
            entry["newrule"] = args.newrule
        if args.feedback:
            entry["userfeedback"] = args.feedback
        loom.append("L3", entry)
        print(f"L3 wisdom entry added: {args.glyph}")

    elif args.command == "consolidate":
        if args.source == "L1":
            entry = loom.consolidate_l1_to_l2(summary=args.summary)
            print(f"Consolidated L1→L2: {entry['summary']}")
        else:
            result = loom.consolidate_l2_to_l3()
            print(f"Consolidated L2→L3: {result}")

    elif args.command == "validate":
        result = loom.validate_state()
        print(f"Status: {result['status']}")
        for check in result["checks_passed"]:
            print(f"  ✓ {check}")
        for issue in result["issues"]:
            print(f"  ⚠ {issue}")

    elif args.command == "events":
        events = loom.get_event_log()
        if args.save:
            path = None if args.save == "auto" else args.save
            saved = loom.save_event_log(path)
            print(f"Event log saved to {saved}")
        else:
            print(json.dumps(events, indent=2, default=str))

    elif args.command == "clear":
        loom.clear_l1()
        print("L1 cleared.")

    elif args.command == "quality":
        from stateful_repl.quality import QualityEvaluator
        state = loom.read_state("ALL")
        evaluator = QualityEvaluator()
        vector = evaluator.evaluate(state)
        print(vector.summary())

    elif args.command == "serve":
        try:
            import uvicorn
        except ImportError:
            print("Install server dependencies: pip install stateful-repl[server]",
                  file=sys.stderr)
            sys.exit(1)
        import os
        os.environ["LOOM_STATE_FILE"] = args.file
        uvicorn.run(
            "stateful_repl.server:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
        )

    elif args.command == "agents":
        from stateful_repl.planner import TaskPlanner, TaskNode
        from stateful_repl.router import TaskRouter
        from stateful_repl.agents import AgentCapability
        if args.plan:
            planner = TaskPlanner()
            plan = planner.create_plan(args.plan)
            plan.add_task(TaskNode(id="task-1", name=args.plan, role="builder"))
            tiers = planner.schedule(plan)
            print(f"Plan: {plan.goal}")
            print(f"Tasks: {plan.task_count}")
            for i, tier in enumerate(tiers):
                print(f"  Tier {i}: {', '.join(tier)}")
        else:
            print("StatefulREPL Phase 3 — Multi-Agent Orchestration")
            print(f"  Version: 0.3.0")
            print(f"  Modules: message_bus, orchestrator, planner, agents, router")
            print(f"  Agent roles: coordinator, builder, verifier, distiller")
            print(f"\nUse --plan '<goal>' to create a task plan.")


if __name__ == "__main__":
    main()
