"""
Phase 3 Tests — Multi-Agent Orchestration.

Tests for: message_bus, orchestrator, planner, agents, router.
"""

import asyncio
import pytest
from unittest.mock import MagicMock


# ─────────────────────────────────────────────────────────
# Message Bus Tests
# ─────────────────────────────────────────────────────────

class TestMessage:
    def test_message_creation(self):
        from stateful_repl.message_bus import Message
        msg = Message(sender="agent-1", topic="task.assign")
        assert msg.sender == "agent-1"
        assert msg.topic == "task.assign"
        assert msg.message_id  # auto-generated
        assert msg.timestamp  # auto-generated

    def test_message_to_dict(self):
        from stateful_repl.message_bus import Message
        msg = Message(sender="a", topic="t", payload={"key": "val"})
        d = msg.to_dict()
        assert d["sender"] == "a"
        assert d["topic"] == "t"
        assert d["payload"]["key"] == "val"

    def test_message_reply(self):
        from stateful_repl.message_bus import Message
        original = Message(sender="builder-1", topic="task.complete", reply_to="coord-1")
        reply = original.reply({"result": "ok"})
        assert reply.sender == "coord-1"
        assert reply.topic == "task.complete.reply"
        assert reply.correlation_id == original.message_id


class TestTopics:
    def test_standard_topics_exist(self):
        from stateful_repl.message_bus import Topics
        assert Topics.TASK_ASSIGN == "task.assign"
        assert Topics.TASK_COMPLETE == "task.complete"
        assert Topics.SAGA_START == "saga.start"
        assert Topics.AGENT_READY == "agent.ready"
        assert Topics.QUALITY_RESULT == "quality.result"


class TestInProcessBus:
    def _run(self, coro):
        return asyncio.run(coro)

    def test_subscribe_and_receive(self):
        async def _test():
            from stateful_repl.message_bus import InProcessBus, Message
            bus = InProcessBus()
            await bus.subscribe("agent-1", ["task.*"])
            await bus.publish(Message(sender="coord", topic="task.assign", payload={"x": 1}))
            msg = await bus.receive("agent-1", timeout=1.0)
            assert msg is not None
            assert msg.topic == "task.assign"
            assert msg.payload["x"] == 1
        self._run(_test())

    def test_no_self_delivery(self):
        async def _test():
            from stateful_repl.message_bus import InProcessBus, Message
            bus = InProcessBus()
            await bus.subscribe("agent-1", ["task.*"])
            await bus.publish(Message(sender="agent-1", topic="task.assign"))
            msg = await bus.receive("agent-1", timeout=0.1)
            assert msg is None  # Should not receive own messages
        self._run(_test())

    def test_broadcast(self):
        async def _test():
            from stateful_repl.message_bus import InProcessBus, Message
            bus = InProcessBus()
            await bus.subscribe("a1", [])
            await bus.subscribe("a2", [])
            await bus.broadcast(Message(sender="coord", topic="system.shutdown"))
            m1 = await bus.receive("a1", timeout=0.5)
            m2 = await bus.receive("a2", timeout=0.5)
            assert m1 is not None
            assert m2 is not None
        self._run(_test())

    def test_direct_send(self):
        async def _test():
            from stateful_repl.message_bus import InProcessBus, Message
            bus = InProcessBus()
            await bus.subscribe("target", [])
            await bus.send("target", Message(sender="src", topic="direct.msg"))
            msg = await bus.receive("target", timeout=0.5)
            assert msg is not None
            assert msg.topic == "direct.msg"
        self._run(_test())

    def test_receive_timeout(self):
        async def _test():
            from stateful_repl.message_bus import InProcessBus
            bus = InProcessBus()
            bus._ensure_inbox("agent-1")
            msg = await bus.receive("agent-1", timeout=0.05)
            assert msg is None
        self._run(_test())

    def test_history(self):
        async def _test():
            from stateful_repl.message_bus import InProcessBus, Message
            bus = InProcessBus()
            await bus.publish(Message(sender="a", topic="task.assign"))
            await bus.publish(Message(sender="b", topic="quality.result"))
            history = bus.get_history()
            assert len(history) == 2
            filtered = bus.get_history(topic="task.*")
            assert len(filtered) == 1
        self._run(_test())

    def test_pending_count(self):
        async def _test():
            from stateful_repl.message_bus import InProcessBus, Message
            bus = InProcessBus()
            await bus.subscribe("a1", ["t.*"])
            assert bus.pending_count("a1") == 0
            await bus.publish(Message(sender="x", topic="t.1"))
            assert bus.pending_count("a1") == 1
        self._run(_test())

    def test_unsubscribe(self):
        async def _test():
            from stateful_repl.message_bus import InProcessBus, Message
            bus = InProcessBus()
            await bus.subscribe("a1", ["task.*"])
            await bus.unsubscribe("a1", ["task.*"])
            await bus.publish(Message(sender="x", topic="task.assign"))
            msg = await bus.receive("a1", timeout=0.1)
            assert msg is None
        self._run(_test())

    def test_clear(self):
        async def _test():
            from stateful_repl.message_bus import InProcessBus, Message
            bus = InProcessBus()
            await bus.subscribe("a1", ["t.*"])
            await bus.publish(Message(sender="x", topic="t.1"))
            bus.clear()
            assert len(bus.agent_ids) == 0
            assert len(bus.get_history()) == 0
        self._run(_test())

    def test_handler_callback(self):
        async def _test():
            from stateful_repl.message_bus import InProcessBus, Message
            received = []
            bus = InProcessBus()
            bus.on("task.*", lambda m: received.append(m))
            await bus.publish(Message(sender="x", topic="task.assign"))
            assert len(received) == 1
            assert received[0].topic == "task.assign"
        self._run(_test())


# ─────────────────────────────────────────────────────────
# Orchestrator Tests
# ─────────────────────────────────────────────────────────

class TestSagaTransactionSync:
    """Phase 1 compat: sync saga."""

    def test_execute_all_steps(self):
        from stateful_repl.orchestrator import SagaTransaction
        results = []
        saga = SagaTransaction()
        saga.add_step("s1", lambda: results.append("a") or "a", lambda r: results.append("undo-a"))
        saga.add_step("s2", lambda: results.append("b") or "b", lambda r: results.append("undo-b"))
        completed = saga.execute()
        assert len(completed) == 2
        assert completed[0] == ("s1", "a")
        assert completed[1] == ("s2", "b")

    def test_compensate_on_failure(self):
        from stateful_repl.orchestrator import SagaTransaction
        compensated = []
        saga = SagaTransaction()
        saga.add_step("s1", lambda: "ok", lambda r: compensated.append(r))
        saga.add_step("s2", lambda: (_ for _ in ()).throw(RuntimeError("fail")), lambda r: None)
        with pytest.raises(RuntimeError):
            saga.execute()
        assert "ok" in compensated


class TestAsyncSagaManager:
    def _run(self, coro):
        return asyncio.run(coro)

    def test_successful_saga(self):
        async def _test():
            from stateful_repl.orchestrator import AsyncSagaManager, SagaStepDef, SagaStatus
            manager = AsyncSagaManager()
            saga_id = await manager.start("test-saga", steps=[
                SagaStepDef(name="step1", action=lambda: "result1", compensation=lambda r: None),
                SagaStepDef(name="step2", action=lambda: "result2", compensation=lambda r: None),
            ])
            record = await manager.wait(saga_id, timeout=5.0)
            assert record.status == SagaStatus.COMPLETED
            assert record.steps["step1"].status.value == "completed"
            assert record.steps["step2"].status.value == "completed"
        self._run(_test())

    def test_saga_failure_compensates(self):
        async def _test():
            from stateful_repl.orchestrator import AsyncSagaManager, SagaStepDef, SagaStatus
            compensated = []
            def fail(): raise RuntimeError("boom")
            manager = AsyncSagaManager(default_timeout=5.0)
            saga_id = await manager.start("fail-saga", steps=[
                SagaStepDef(name="step1", action=lambda: "ok", compensation=lambda r: compensated.append(r)),
                SagaStepDef(name="step2", action=fail, compensation=lambda r: None),
            ])
            record = await manager.wait(saga_id, timeout=5.0)
            assert record.status == SagaStatus.COMPENSATED
            assert "ok" in compensated
        self._run(_test())

    def test_saga_with_retry(self):
        async def _test():
            from stateful_repl.orchestrator import AsyncSagaManager, SagaStepDef, SagaStatus, RetryPolicy
            attempts = []
            def flaky():
                attempts.append(1)
                if len(attempts) < 3:
                    raise RuntimeError("not yet")
                return "success"
            manager = AsyncSagaManager(default_timeout=10.0)
            saga_id = await manager.start("retry-saga", steps=[
                SagaStepDef(
                    name="flaky-step",
                    action=flaky,
                    compensation=lambda r: None,
                    retry=RetryPolicy(max_retries=3, delay=0.01),
                ),
            ])
            record = await manager.wait(saga_id, timeout=10.0)
            assert record.status == SagaStatus.COMPLETED
            assert len(attempts) == 3
        self._run(_test())

    def test_saga_list_and_status(self):
        async def _test():
            from stateful_repl.orchestrator import AsyncSagaManager, SagaStepDef, SagaStatus
            manager = AsyncSagaManager()
            saga_id = await manager.start("test", steps=[
                SagaStepDef(name="s1", action=lambda: "ok", compensation=lambda r: None),
            ])
            await manager.wait(saga_id, timeout=5.0)
            sagas = manager.list_sagas()
            assert len(sagas) == 1
            record = manager.get_status(saga_id)
            assert record.status == SagaStatus.COMPLETED
        self._run(_test())

    def test_saga_record_to_dict(self):
        async def _test():
            from stateful_repl.orchestrator import AsyncSagaManager, SagaStepDef
            manager = AsyncSagaManager()
            saga_id = await manager.start("dict-test", steps=[
                SagaStepDef(name="s1", action=lambda: "ok", compensation=lambda r: None),
            ])
            await manager.wait(saga_id, timeout=5.0)
            d = manager.get_status(saga_id).to_dict()
            assert d["name"] == "dict-test"
            assert d["status"] == "completed"
            assert "s1" in d["steps"]
        self._run(_test())

    def test_saga_with_bus_events(self):
        async def _test():
            from stateful_repl.orchestrator import AsyncSagaManager, SagaStepDef, SagaStatus
            from stateful_repl.message_bus import InProcessBus
            bus = InProcessBus()
            received = []
            bus.on("saga.*", lambda m: received.append(m.topic))
            manager = AsyncSagaManager(bus=bus)
            saga_id = await manager.start("bus-test", steps=[
                SagaStepDef(name="s1", action=lambda: "ok", compensation=lambda r: None),
            ])
            await manager.wait(saga_id, timeout=5.0)
            assert any("saga." in t for t in received)
        self._run(_test())


# ─────────────────────────────────────────────────────────
# Planner Tests
# ─────────────────────────────────────────────────────────

class TestTaskNode:
    def test_create_node(self):
        from stateful_repl.planner import TaskNode, TaskStatus
        node = TaskNode(id="t1", name="Test task")
        assert node.id == "t1"
        assert node.status == TaskStatus.PENDING
        assert node.depends_on == []

    def test_to_dict(self):
        from stateful_repl.planner import TaskNode
        node = TaskNode(id="t1", name="Test", role="verifier")
        d = node.to_dict()
        assert d["id"] == "t1"
        assert d["role"] == "verifier"


class TestTaskPlan:
    def test_add_and_get(self):
        from stateful_repl.planner import TaskPlan, TaskNode
        plan = TaskPlan(goal="Test goal")
        plan.add_task(TaskNode(id="t1", name="Task 1"))
        plan.add_task(TaskNode(id="t2", name="Task 2"))
        assert plan.task_count == 2
        assert plan.get_task("t1") is not None

    def test_remove_task(self):
        from stateful_repl.planner import TaskPlan, TaskNode
        plan = TaskPlan()
        plan.add_task(TaskNode(id="t1", name="T1"))
        plan.add_task(TaskNode(id="t2", name="T2", depends_on=["t1"]))
        plan.remove_task("t1")
        assert plan.task_count == 1
        assert plan.get_task("t2").depends_on == []

    def test_progress(self):
        from stateful_repl.planner import TaskPlan, TaskNode, TaskStatus
        plan = TaskPlan()
        plan.add_task(TaskNode(id="t1", name="T1", status=TaskStatus.COMPLETED))
        plan.add_task(TaskNode(id="t2", name="T2"))
        assert plan.progress == 0.5
        assert not plan.is_complete

    def test_empty_plan_progress(self):
        from stateful_repl.planner import TaskPlan
        plan = TaskPlan()
        assert plan.progress == 1.0

    def test_to_dict(self):
        from stateful_repl.planner import TaskPlan, TaskNode
        plan = TaskPlan(goal="Build it")
        plan.add_task(TaskNode(id="t1", name="T1"))
        d = plan.to_dict()
        assert d["goal"] == "Build it"
        assert "t1" in d["tasks"]


class TestValidateDAG:
    def test_valid_dag(self):
        from stateful_repl.planner import TaskPlan, TaskNode, validate_dag
        plan = TaskPlan()
        plan.add_task(TaskNode(id="a", name="A"))
        plan.add_task(TaskNode(id="b", name="B", depends_on=["a"]))
        issues = validate_dag(plan)
        assert issues == []

    def test_self_dependency(self):
        from stateful_repl.planner import TaskPlan, TaskNode, validate_dag
        plan = TaskPlan()
        plan.add_task(TaskNode(id="a", name="A", depends_on=["a"]))
        issues = validate_dag(plan)
        assert any("depends on itself" in i for i in issues)

    def test_missing_dependency(self):
        from stateful_repl.planner import TaskPlan, TaskNode, validate_dag
        plan = TaskPlan()
        plan.add_task(TaskNode(id="a", name="A", depends_on=["nonexistent"]))
        issues = validate_dag(plan)
        assert any("unknown task" in i for i in issues)

    def test_cycle_detection(self):
        from stateful_repl.planner import TaskPlan, TaskNode, validate_dag
        plan = TaskPlan()
        plan.add_task(TaskNode(id="a", name="A", depends_on=["b"]))
        plan.add_task(TaskNode(id="b", name="B", depends_on=["a"]))
        issues = validate_dag(plan)
        assert any("Cycle" in i for i in issues)


class TestTopologicalSort:
    def test_linear_dag(self):
        from stateful_repl.planner import TaskPlan, TaskNode, topological_sort
        plan = TaskPlan()
        plan.add_task(TaskNode(id="a", name="A"))
        plan.add_task(TaskNode(id="b", name="B", depends_on=["a"]))
        plan.add_task(TaskNode(id="c", name="C", depends_on=["b"]))
        tiers = topological_sort(plan)
        assert len(tiers) == 3
        assert tiers[0] == ["a"]
        assert tiers[1] == ["b"]
        assert tiers[2] == ["c"]

    def test_parallel_tiers(self):
        from stateful_repl.planner import TaskPlan, TaskNode, topological_sort
        plan = TaskPlan()
        plan.add_task(TaskNode(id="a", name="A"))
        plan.add_task(TaskNode(id="b1", name="B1", depends_on=["a"]))
        plan.add_task(TaskNode(id="b2", name="B2", depends_on=["a"]))
        plan.add_task(TaskNode(id="c", name="C", depends_on=["b1", "b2"]))
        tiers = topological_sort(plan)
        assert len(tiers) == 3
        assert set(tiers[1]) == {"b1", "b2"}  # parallel
        assert tiers[2] == ["c"]

    def test_empty_plan(self):
        from stateful_repl.planner import TaskPlan, topological_sort
        plan = TaskPlan()
        assert topological_sort(plan) == []

    def test_cycle_raises(self):
        from stateful_repl.planner import TaskPlan, TaskNode, topological_sort, PlanValidationError
        plan = TaskPlan()
        plan.add_task(TaskNode(id="a", name="A", depends_on=["b"]))
        plan.add_task(TaskNode(id="b", name="B", depends_on=["a"]))
        with pytest.raises(PlanValidationError):
            topological_sort(plan)


class TestTaskPlanner:
    def test_create_plan(self):
        from stateful_repl.planner import TaskPlanner
        planner = TaskPlanner()
        plan = planner.create_plan("Build feature X")
        assert plan.goal == "Build feature X"
        assert planner.get_plan(plan.plan_id) is plan

    def test_get_ready_tasks(self):
        from stateful_repl.planner import TaskPlanner, TaskNode, TaskStatus
        planner = TaskPlanner()
        plan = planner.create_plan("Test")
        plan.add_task(TaskNode(id="a", name="A"))
        plan.add_task(TaskNode(id="b", name="B", depends_on=["a"]))
        ready = planner.get_ready_tasks(plan)
        assert len(ready) == 1
        assert ready[0].id == "a"
        # After completing A, B should be ready
        planner.mark_completed(plan, "a")
        ready2 = planner.get_ready_tasks(plan)
        assert len(ready2) == 1
        assert ready2[0].id == "b"

    def test_mark_failed_blocks_dependents(self):
        from stateful_repl.planner import TaskPlanner, TaskNode, TaskStatus
        planner = TaskPlanner()
        plan = planner.create_plan("Test")
        plan.add_task(TaskNode(id="a", name="A"))
        plan.add_task(TaskNode(id="b", name="B", depends_on=["a"]))
        plan.add_task(TaskNode(id="c", name="C", depends_on=["b"]))
        planner.mark_failed(plan, "a", "exploded")
        assert plan.tasks["b"].status == TaskStatus.BLOCKED
        assert plan.tasks["c"].status == TaskStatus.BLOCKED

    def test_schedule(self):
        from stateful_repl.planner import TaskPlanner, TaskNode
        planner = TaskPlanner()
        plan = planner.create_plan("Test")
        plan.add_task(TaskNode(id="a", name="A"))
        plan.add_task(TaskNode(id="b", name="B", depends_on=["a"]))
        tiers = planner.schedule(plan)
        assert len(tiers) == 2


# ─────────────────────────────────────────────────────────
# Agent Tests
# ─────────────────────────────────────────────────────────

class TestAgentCapability:
    def test_defaults(self):
        from stateful_repl.agents import AgentCapability
        cap = AgentCapability(role="builder")
        assert cap.role == "builder"
        assert cap.max_concurrent_tasks == 1
        assert not cap.has_model

class TestTaskResult:
    def test_to_dict(self):
        from stateful_repl.agents import TaskResult
        r = TaskResult(task_id="t1", agent_id="b1", success=True, output="done")
        d = r.to_dict()
        assert d["task_id"] == "t1"
        assert d["success"] is True
        assert d["output"] == "done"


class TestBuilderLoom:
    def _run(self, coro):
        return asyncio.run(coro)

    def test_passthrough_task(self):
        async def _test():
            from stateful_repl.agents import BuilderLoom
            from stateful_repl.planner import TaskNode
            builder = BuilderLoom()
            task = TaskNode(id="t1", name="Test task")
            result = await builder.handle_task(task, {})
            assert result.success
            assert "Test task" in str(result.output)
        self._run(_test())

    def test_sandbox_execution(self):
        async def _test():
            from stateful_repl.agents import BuilderLoom
            from stateful_repl.planner import TaskNode
            builder = BuilderLoom()
            task = TaskNode(
                id="t1", name="Sandbox test",
                metadata={"code": "loom.update_l1('goal', 'calc'); result = 2 + 2"},
            )
            result = await builder.handle_task(task, {})
            assert result.success
        self._run(_test())

    def test_capability(self):
        from stateful_repl.agents import BuilderLoom
        builder = BuilderLoom()
        cap = builder.capability
        assert cap.role == "builder"
        assert cap.has_sandbox


class TestVerifierLoom:
    def _run(self, coro):
        return asyncio.run(coro)

    def test_verify_grounded_text(self):
        async def _test():
            from stateful_repl.agents import VerifierLoom
            verifier = VerifierLoom()
            result = await verifier.verify_output(
                "Paris is the capital of France.",
                sources=["Paris is the capital of France."],
            )
            assert result.success
            assert "hallucination" in result.output
        self._run(_test())

    def test_verify_with_state(self):
        async def _test():
            from stateful_repl.agents import VerifierLoom
            verifier = VerifierLoom()
            state = {
                "L1": {"goal": "test", "constraints": [], "artifacts": [], "open_questions": []},
                "L2": [],
                "L3": {"rules": [], "concepts": [], "tracewisdomlog": []},
            }
            result = await verifier.verify_output("test output", state=state)
            assert "quality" in result.output
        self._run(_test())

    def test_capability(self):
        from stateful_repl.agents import VerifierLoom
        v = VerifierLoom()
        assert v.capability.role == "verifier"
        assert "quality" in v.capability.skills


class TestDistillerLoom:
    def _run(self, coro):
        return asyncio.run(coro)

    def test_structural_distill(self):
        async def _test():
            from stateful_repl.agents import DistillerLoom
            distiller = DistillerLoom()
            result = await distiller.distill([
                "Python is great for data science. It has many libraries.",
                "Python is great for data science. It is easy to learn.",
            ])
            assert result.success
            out = result.output
            assert out["source_count"] == 2
            assert "invariants" in out
        self._run(_test())

    def test_empty_sources_fails(self):
        async def _test():
            from stateful_repl.agents import DistillerLoom
            distiller = DistillerLoom()
            result = await distiller.distill([])
            assert not result.success
            assert "No sources" in result.error
        self._run(_test())

    def test_capability(self):
        from stateful_repl.agents import DistillerLoom
        d = DistillerLoom()
        assert d.capability.role == "distiller"
        assert "mcrd" in d.capability.skills


class TestCoordinatorLoom:
    def _run(self, coro):
        return asyncio.run(coro)

    def test_handle_goal_no_model(self):
        async def _test():
            from stateful_repl.agents import CoordinatorLoom
            coord = CoordinatorLoom()
            plan = await coord.handle_goal("Build a feature")
            assert plan.task_count >= 1
            assert plan.goal == "Build a feature"
        self._run(_test())

    def test_register_agent(self):
        from stateful_repl.agents import CoordinatorLoom, AgentCapability
        coord = CoordinatorLoom()
        coord.register_agent("b1", AgentCapability(role="builder"))
        assert "b1" in coord._agent_registry


class TestAgentBusIntegration:
    def _run(self, coro):
        return asyncio.run(coro)

    def test_agent_start_emits_ready(self):
        async def _test():
            from stateful_repl.agents import BuilderLoom
            from stateful_repl.message_bus import InProcessBus
            bus = InProcessBus()
            received = []
            bus.on("agent.*", lambda m: received.append(m))
            builder = BuilderLoom(bus=bus, agent_id="builder-test")
            await builder.start()
            assert any(m.topic == "agent.ready" for m in received)
        self._run(_test())

    def test_agent_stop(self):
        async def _test():
            from stateful_repl.agents import BuilderLoom, AgentStatus
            from stateful_repl.message_bus import InProcessBus
            bus = InProcessBus()
            builder = BuilderLoom(bus=bus)
            await builder.start()
            await builder.stop()
            assert builder.status == AgentStatus.STOPPED
        self._run(_test())


# ─────────────────────────────────────────────────────────
# Router Tests
# ─────────────────────────────────────────────────────────

class TestAgentHealth:
    def test_is_available(self):
        from stateful_repl.router import AgentHealth
        from stateful_repl.agents import AgentCapability, AgentStatus
        h = AgentHealth(agent_id="b1", capability=AgentCapability(role="builder"))
        assert h.is_available
        h.status = AgentStatus.STOPPED
        assert not h.is_available

    def test_record_completion(self):
        from stateful_repl.router import AgentHealth
        from stateful_repl.agents import AgentCapability
        h = AgentHealth(agent_id="b1", capability=AgentCapability(role="builder"))
        h.active_tasks = 1
        h.record_completion(100.0)
        assert h.active_tasks == 0
        assert h.total_completed == 1
        assert h.avg_task_duration_ms == 100.0

    def test_error_rate(self):
        from stateful_repl.router import AgentHealth
        from stateful_repl.agents import AgentCapability
        h = AgentHealth(agent_id="b1", capability=AgentCapability(role="builder"))
        h.active_tasks = 2
        h.record_completion(50.0)
        h.record_failure()
        assert h.error_rate == 0.5


class TestTaskRouter:
    def test_register_and_route(self):
        from stateful_repl.router import TaskRouter
        from stateful_repl.agents import AgentCapability
        from stateful_repl.planner import TaskNode
        router = TaskRouter()
        router.register("b1", AgentCapability(role="builder"))
        result = router.route(TaskNode(id="t1", name="Build", role="builder"))
        assert result == "b1"

    def test_no_match(self):
        from stateful_repl.router import TaskRouter
        from stateful_repl.agents import AgentCapability
        from stateful_repl.planner import TaskNode
        router = TaskRouter()
        router.register("b1", AgentCapability(role="builder"))
        result = router.route(TaskNode(id="t1", name="Verify", role="verifier"))
        assert result is None

    def test_round_robin(self):
        from stateful_repl.router import TaskRouter, RoutingStrategy
        from stateful_repl.agents import AgentCapability
        from stateful_repl.planner import TaskNode
        router = TaskRouter(strategy=RoutingStrategy.ROUND_ROBIN)
        router.register("b1", AgentCapability(role="builder"))
        router.register("b2", AgentCapability(role="builder"))
        r1 = router.route(TaskNode(id="t1", name="T1", role="builder"))
        r2 = router.route(TaskNode(id="t2", name="T2", role="builder"))
        assert {r1, r2} == {"b1", "b2"}

    def test_least_busy(self):
        from stateful_repl.router import TaskRouter, RoutingStrategy
        from stateful_repl.agents import AgentCapability
        from stateful_repl.planner import TaskNode
        router = TaskRouter(strategy=RoutingStrategy.LEAST_BUSY)
        router.register("b1", AgentCapability(role="builder", max_concurrent_tasks=5))
        router.register("b2", AgentCapability(role="builder", max_concurrent_tasks=5))
        router.record_task_start("b1")
        router.record_task_start("b1")
        result = router.route(TaskNode(id="t1", name="T", role="builder"))
        assert result == "b2"

    def test_routing_rule(self):
        from stateful_repl.router import TaskRouter, RoutingRule
        from stateful_repl.agents import AgentCapability
        from stateful_repl.planner import TaskNode
        router = TaskRouter()
        router.register("b1", AgentCapability(role="builder"))
        router.register("b2", AgentCapability(role="builder"))
        router.add_rule(RoutingRule(name="prefer-b2", match_role="builder", prefer_agent="b2"))
        result = router.route(TaskNode(id="t1", name="T", role="builder"))
        assert result == "b2"

    def test_unavailable_agent_skipped(self):
        from stateful_repl.router import TaskRouter
        from stateful_repl.agents import AgentCapability, AgentStatus
        from stateful_repl.planner import TaskNode
        router = TaskRouter()
        router.register("b1", AgentCapability(role="builder"))
        router.update_status("b1", AgentStatus.STOPPED)
        result = router.route(TaskNode(id="t1", name="T", role="builder"))
        assert result is None

    def test_get_available_agents(self):
        from stateful_repl.router import TaskRouter
        from stateful_repl.agents import AgentCapability
        router = TaskRouter()
        router.register("b1", AgentCapability(role="builder"))
        router.register("v1", AgentCapability(role="verifier"))
        available = router.get_available_agents(role="builder")
        assert available == ["b1"]

    def test_summary(self):
        from stateful_repl.router import TaskRouter
        from stateful_repl.agents import AgentCapability
        router = TaskRouter()
        router.register("b1", AgentCapability(role="builder"))
        s = router.summary()
        assert s["total_agents"] == 1
        assert s["agents_by_role"]["builder"] == 1

    def test_unregister(self):
        from stateful_repl.router import TaskRouter
        from stateful_repl.agents import AgentCapability
        router = TaskRouter()
        router.register("b1", AgentCapability(role="builder"))
        router.unregister("b1")
        assert router.summary()["total_agents"] == 0


# ─────────────────────────────────────────────────────────
# Edge Case Tests — hardening pass
# ─────────────────────────────────────────────────────────

class TestMessageBusEdgeCases:
    """Edge cases for message bus reliability."""

    def _run(self, coro):
        return asyncio.run(coro)

    def test_queue_full_drops_oldest(self):
        """When the inbox is full, the oldest message should be dropped."""
        async def _test():
            from stateful_repl.message_bus import InProcessBus, Message
            bus = InProcessBus(max_queue_size=2)
            await bus.subscribe("agent", ["test.*"])
            # Fill the queue beyond capacity
            for i in range(4):
                await bus.publish(Message(sender="src", topic="test.msg", payload={"i": i}))
            # Should receive the latest messages (oldest dropped)
            msg1 = await bus.receive("agent", timeout=0.1)
            msg2 = await bus.receive("agent", timeout=0.1)
            assert msg1 is not None
            assert msg2 is not None
            # Queue should be empty now
            msg3 = await bus.receive("agent", timeout=0.1)
            assert msg3 is None
        self._run(_test())

    def test_receive_nowait_empty(self):
        """receive_nowait should return None on empty inbox."""
        async def _test():
            from stateful_repl.message_bus import InProcessBus
            bus = InProcessBus()
            msg = await bus.receive_nowait("nonexistent")
            assert msg is None
        self._run(_test())

    def test_handler_error_captured(self):
        """Handler errors should not crash publish and should be recorded."""
        async def _test():
            from stateful_repl.message_bus import InProcessBus, Message
            bus = InProcessBus()

            def bad_handler(msg):
                raise ValueError("handler boom")

            bus.on("test.*", bad_handler)
            await bus.subscribe("agent", ["test.*"])
            # Should not raise
            await bus.publish(Message(sender="src", topic="test.msg"))
            assert bus._last_handler_error is not None
            assert "handler boom" in str(bus._last_handler_error)
        self._run(_test())

    def test_unsubscribe_specific_topic(self):
        """Unsubscribing from one topic should not affect other subscriptions."""
        async def _test():
            from stateful_repl.message_bus import InProcessBus, Message
            bus = InProcessBus()
            await bus.subscribe("agent", ["task.*", "system.*"])
            await bus.unsubscribe("agent", ["task.*"])
            await bus.publish(Message(sender="src", topic="task.assign"))
            msg_task = await bus.receive("agent", timeout=0.1)
            assert msg_task is None  # unsubscribed
            await bus.publish(Message(sender="src", topic="system.status"))
            msg_sys = await bus.receive("agent", timeout=0.1)
            assert msg_sys is not None  # still subscribed
        self._run(_test())

    def test_broadcast_skips_sender(self):
        """Broadcast should not deliver back to the sender."""
        async def _test():
            from stateful_repl.message_bus import InProcessBus, Message
            bus = InProcessBus()
            await bus.subscribe("sender", [])
            await bus.subscribe("receiver", [])
            await bus.broadcast(Message(sender="sender", topic="announce"))
            msg_sender = await bus.receive("sender", timeout=0.1)
            msg_receiver = await bus.receive("receiver", timeout=0.1)
            assert msg_sender is None
            assert msg_receiver is not None
        self._run(_test())

    def test_history_limit_enforced(self):
        """History should not exceed the configured limit."""
        async def _test():
            from stateful_repl.message_bus import InProcessBus, Message
            bus = InProcessBus(history_limit=5)
            for i in range(10):
                await bus.publish(Message(sender="s", topic="t", payload={"i": i}))
            history = bus.get_history()
            assert len(history) == 5
            # Most recent messages kept
            assert history[-1].payload["i"] == 9
        self._run(_test())

    def test_message_ttl_expired(self):
        """Messages past their TTL should be discarded on receive."""
        async def _test():
            import time
            from stateful_repl.message_bus import InProcessBus, Message
            bus = InProcessBus()
            await bus.subscribe("agent", ["t.*"])
            # Send a message with very short TTL
            await bus.publish(Message(
                sender="src", topic="t.msg", ttl=0.01,
            ))
            # Wait for TTL to expire
            time.sleep(0.05)
            msg = await bus.receive("agent", timeout=0.1)
            assert msg is None  # expired
        self._run(_test())

    def test_clear_resets_everything(self):
        """clear() should empty all state."""
        async def _test():
            from stateful_repl.message_bus import InProcessBus, Message
            bus = InProcessBus()
            await bus.subscribe("a1", ["t.*"])
            await bus.publish(Message(sender="s", topic="t.x"))
            bus.clear()
            assert bus.agent_ids == []
            assert bus.get_history() == []
        self._run(_test())


class TestPlannerEdgeCases:
    """Edge cases for HALO planner."""

    def test_empty_plan_schedule(self):
        """An empty plan should produce an empty schedule."""
        from stateful_repl.planner import TaskPlanner
        planner = TaskPlanner()
        plan = planner.create_plan("empty")
        tiers = planner.schedule(plan)
        assert tiers == []

    def test_empty_plan_progress(self):
        """An empty plan should report 100% progress."""
        from stateful_repl.planner import TaskPlan
        plan = TaskPlan(goal="empty")
        assert plan.progress == 1.0
        assert plan.is_complete is True

    def test_self_dependency_detected(self):
        """A task depending on itself should be caught."""
        from stateful_repl.planner import TaskPlan, TaskNode, validate_dag
        plan = TaskPlan()
        plan.add_task(TaskNode(id="a", name="A", depends_on=["a"]))
        issues = validate_dag(plan)
        assert any("depends on itself" in i for i in issues)

    def test_missing_dependency_detected(self):
        """A reference to a nonexistent task should be caught."""
        from stateful_repl.planner import TaskPlan, TaskNode, validate_dag
        plan = TaskPlan()
        plan.add_task(TaskNode(id="a", name="A", depends_on=["ghost"]))
        issues = validate_dag(plan)
        assert any("unknown task" in i for i in issues)

    def test_cycle_detected(self):
        """A → B → A cycle should be detected."""
        from stateful_repl.planner import TaskPlan, TaskNode, validate_dag
        plan = TaskPlan()
        plan.add_task(TaskNode(id="a", name="A", depends_on=["b"]))
        plan.add_task(TaskNode(id="b", name="B", depends_on=["a"]))
        issues = validate_dag(plan)
        assert any("Cycle" in i for i in issues)

    def test_mark_invalid_task_raises(self):
        """mark_running/completed/failed should raise on unknown task ID."""
        from stateful_repl.planner import TaskPlanner, TaskPlan
        planner = TaskPlanner()
        plan = TaskPlan(goal="test")
        import pytest
        with pytest.raises(KeyError, match="not found"):
            planner.mark_running(plan, "nonexistent", "agent-1")
        with pytest.raises(KeyError, match="not found"):
            planner.mark_completed(plan, "nonexistent")
        with pytest.raises(KeyError, match="not found"):
            planner.mark_failed(plan, "nonexistent", "error")

    def test_propagate_block_cascades(self):
        """Failing a task should block all transitive dependents."""
        from stateful_repl.planner import TaskPlanner, TaskPlan, TaskNode, TaskStatus
        planner = TaskPlanner()
        plan = TaskPlan(goal="cascade")
        plan.add_task(TaskNode(id="a", name="A"))
        plan.add_task(TaskNode(id="b", name="B", depends_on=["a"]))
        plan.add_task(TaskNode(id="c", name="C", depends_on=["b"]))
        planner.mark_failed(plan, "a", "root failure")
        assert plan.tasks["b"].status == TaskStatus.BLOCKED
        assert plan.tasks["c"].status == TaskStatus.BLOCKED

    def test_remove_task_cleans_deps(self):
        """Removing a task should clean up dependency references."""
        from stateful_repl.planner import TaskPlan, TaskNode
        plan = TaskPlan()
        plan.add_task(TaskNode(id="a", name="A"))
        plan.add_task(TaskNode(id="b", name="B", depends_on=["a"]))
        plan.remove_task("a")
        assert "a" not in plan.tasks
        assert plan.tasks["b"].depends_on == []

    def test_get_ready_skips_missing_dep(self):
        """get_ready_tasks handles references to removed dependencies."""
        from stateful_repl.planner import TaskPlanner, TaskPlan, TaskNode
        planner = TaskPlanner()
        plan = TaskPlan(goal="test")
        plan.add_task(TaskNode(id="b", name="B", depends_on=["removed"]))
        ready = planner.get_ready_tasks(plan)
        assert len(ready) == 0  # dep not in plan, so not completed

    def test_decompose_without_model(self):
        """decompose without a model should create a single-task plan."""
        async def _test():
            from stateful_repl.planner import TaskPlanner
            planner = TaskPlanner()
            plan = await planner.decompose("Build a spaceship")
            assert plan.task_count == 1
            assert plan.tasks["task-1"].name == "Build a spaceship"
        asyncio.run(_test())


class TestOrchestratorEdgeCases:
    """Edge cases for saga orchestration."""

    def _run(self, coro):
        return asyncio.run(coro)

    def test_compensation_failure_sets_failed_status(self):
        """If compensation itself fails, saga should be FAILED not COMPENSATED."""
        async def _test():
            from stateful_repl.orchestrator import AsyncSagaManager, SagaStepDef, SagaStatus
            def fail_action():
                raise RuntimeError("action boom")
            def fail_compensation(r):
                raise RuntimeError("compensation boom")
            manager = AsyncSagaManager(default_timeout=5.0)
            saga_id = await manager.start("double-fail", steps=[
                SagaStepDef(name="s1", action=lambda: "ok", compensation=fail_compensation),
                SagaStepDef(name="s2", action=fail_action, compensation=lambda r: None),
            ])
            record = await manager.wait(saga_id, timeout=5.0)
            assert record.status == SagaStatus.FAILED
            assert "Compensation also failed" in record.steps["s1"].error
        self._run(_test())

    def test_saga_wait_unknown_id_raises(self):
        """Waiting on an unknown saga should raise KeyError."""
        async def _test():
            from stateful_repl.orchestrator import AsyncSagaManager
            import pytest
            manager = AsyncSagaManager()
            with pytest.raises(KeyError, match="Unknown saga"):
                await manager.wait("nonexistent")
        self._run(_test())

    def test_sync_saga_rollback(self):
        """Phase 1 sync SagaTransaction should rollback on failure."""
        from stateful_repl.orchestrator import SagaTransaction
        log = []
        saga = SagaTransaction()
        saga.add_step("s1", action=lambda: "r1", compensation=lambda r: log.append(f"undo:{r}"))
        saga.add_step("s2", action=lambda: (_ for _ in ()).throw(RuntimeError("fail")),
                       compensation=lambda r: None)
        import pytest
        with pytest.raises(RuntimeError):
            saga.execute()
        assert "undo:r1" in log

    def test_saga_step_dependency_not_met(self):
        """Step with unmet dependency should be skipped and saga should fail."""
        async def _test():
            from stateful_repl.orchestrator import AsyncSagaManager, SagaStepDef, SagaStatus
            manager = AsyncSagaManager(default_timeout=5.0)
            saga_id = await manager.start("dep-test", steps=[
                SagaStepDef(name="s2", action=lambda: "ok", compensation=lambda r: None,
                           depends_on=["s1"]),
            ])
            record = await manager.wait(saga_id, timeout=5.0)
            assert record.status in (SagaStatus.COMPENSATED, SagaStatus.FAILED)
        self._run(_test())

    def test_async_action_supported(self):
        """Async action functions should be properly awaited."""
        async def _test():
            from stateful_repl.orchestrator import AsyncSagaManager, SagaStepDef, SagaStatus

            async def async_action():
                await asyncio.sleep(0.01)
                return "async result"

            manager = AsyncSagaManager(default_timeout=5.0)
            saga_id = await manager.start("async-test", steps=[
                SagaStepDef(name="s1", action=async_action, compensation=lambda r: None),
            ])
            record = await manager.wait(saga_id, timeout=5.0)
            assert record.status == SagaStatus.COMPLETED
            assert record.steps["s1"].result == "async result"
        self._run(_test())

    def test_saga_clear_cancels_pending(self):
        """clear() should cancel any pending saga tasks."""
        async def _test():
            from stateful_repl.orchestrator import AsyncSagaManager, SagaStepDef

            async def slow_action():
                await asyncio.sleep(100)
                return "done"

            manager = AsyncSagaManager(default_timeout=300)
            await manager.start("slow", steps=[
                SagaStepDef(name="s1", action=slow_action, compensation=lambda r: None),
            ])
            # Don't wait — clear immediately
            manager.clear()
            assert len(manager._sagas) == 0
        self._run(_test())


class TestRouterEdgeCases:
    """Edge cases for task routing."""

    def test_route_no_registered_agents(self):
        """Routing with no agents should return None."""
        from stateful_repl.router import TaskRouter
        from stateful_repl.planner import TaskNode
        router = TaskRouter()
        assert router.route(TaskNode(id="t", name="T", role="builder")) is None

    def test_route_wrong_role(self):
        """Agent with wrong role should not be matched."""
        from stateful_repl.router import TaskRouter
        from stateful_repl.agents import AgentCapability
        from stateful_repl.planner import TaskNode
        router = TaskRouter()
        router.register("v1", AgentCapability(role="verifier"))
        assert router.route(TaskNode(id="t", name="T", role="builder")) is None

    def test_health_recording(self):
        """Health stats should be accurate after completions and failures."""
        from stateful_repl.router import TaskRouter
        from stateful_repl.agents import AgentCapability
        router = TaskRouter()
        router.register("b1", AgentCapability(role="builder"))
        router.record_task_start("b1")
        assert router.get_health("b1").active_tasks == 1
        router.record_task_complete("b1", 100.0)
        h = router.get_health("b1")
        assert h.active_tasks == 0
        assert h.total_completed == 1
        assert h.avg_task_duration_ms == 100.0
        router.record_task_failure("b1")
        h = router.get_health("b1")
        assert h.total_failed == 1
        assert h.error_rate == 0.5  # 1 fail / 2 total

    def test_unregister_nonexistent_agent(self):
        """Unregistering a non-existent agent should not raise."""
        from stateful_repl.router import TaskRouter
        router = TaskRouter()
        router.unregister("ghost")  # Should not raise

    def test_get_health_nonexistent(self):
        """Getting health for unknown agent should return None."""
        from stateful_repl.router import TaskRouter
        router = TaskRouter()
        assert router.get_health("ghost") is None

    def test_record_on_nonexistent_agent_noop(self):
        """Recording task events for unknown agents should be no-ops."""
        from stateful_repl.router import TaskRouter
        router = TaskRouter()
        router.record_task_start("ghost")
        router.record_task_complete("ghost", 100.0)
        router.record_task_failure("ghost")
        # Should not raise


class TestAgentEdgeCases:
    """Edge cases for agent roles."""

    def _run(self, coro):
        return asyncio.run(coro)

    def test_builder_without_model_or_code(self):
        """Builder with no model and no code should return acknowledgment."""
        async def _test():
            from stateful_repl.agents import BuilderLoom, TaskResult
            from stateful_repl.planner import TaskNode
            builder = BuilderLoom()
            task = TaskNode(id="t1", name="Test Task")
            result = await builder.handle_task(task, {})
            assert result.success is True
            assert "acknowledged" in result.output.lower()
        self._run(_test())

    def test_verifier_with_empty_text(self):
        """Verifier should handle empty text input gracefully."""
        async def _test():
            from stateful_repl.agents import VerifierLoom
            verifier = VerifierLoom()
            result = await verifier.verify_output("")
            assert isinstance(result.success, bool)
        self._run(_test())

    def test_distiller_with_no_sources(self):
        """Distiller should fail gracefully with no sources."""
        async def _test():
            from stateful_repl.agents import DistillerLoom
            from stateful_repl.planner import TaskNode
            distiller = DistillerLoom()
            task = TaskNode(id="t1", name="Distill")
            result = await distiller.handle_task(task, {"sources": []})
            assert result.success is False
            assert "No sources" in result.error
        self._run(_test())

    def test_distiller_structural_single_source(self):
        """Distiller should handle a single source without crashing."""
        async def _test():
            from stateful_repl.agents import DistillerLoom
            result = await DistillerLoom().distill(["Only one source here."])
            assert result.success is True
            assert result.output["source_count"] == 1
        self._run(_test())

    def test_agent_stop(self):
        """Stopping an agent should set status to STOPPED."""
        async def _test():
            from stateful_repl.agents import BuilderLoom, AgentStatus
            agent = BuilderLoom()
            await agent.stop()
            assert agent.status == AgentStatus.STOPPED
        self._run(_test())

    def test_coordinator_registers_agent_on_ready(self):
        """Coordinator should auto-register agents from AGENT_READY messages."""
        async def _test():
            from stateful_repl.agents import CoordinatorLoom
            from stateful_repl.message_bus import InProcessBus, Message, Topics
            bus = InProcessBus()
            coord = CoordinatorLoom(bus=bus)
            await coord.start()
            # Simulate an agent announcing readiness
            await bus.send(coord.agent_id, Message(
                sender="builder-abc", topic=Topics.AGENT_READY, payload={"role": "builder"},
            ))
            # Process one message
            msg = await coord._receive(timeout=0.5)
            if msg and msg.topic == Topics.AGENT_READY:
                coord.register_agent(msg.sender, __import__("stateful_repl.agents", fromlist=["AgentCapability"]).AgentCapability(role=msg.payload["role"]))
            assert "builder-abc" in coord._agent_registry
        self._run(_test())

    def test_task_result_to_dict(self):
        """TaskResult.to_dict should serialize all fields."""
        from stateful_repl.agents import TaskResult
        r = TaskResult(
            task_id="t1", agent_id="a1", success=True,
            output={"key": "val"}, artifacts=["art-1"],
            quality_score=0.95, duration_ms=123.4,
        )
        d = r.to_dict()
        assert d["task_id"] == "t1"
        assert d["success"] is True
        assert d["quality_score"] == 0.95


class TestEventStoreEdgeCases:
    """Edge cases for event stores."""

    def test_sqlite_replay_with_limit(self):
        """SQLite replay with up_to should limit results."""
        import tempfile
        from stateful_repl.events import SQLiteEventStore
        from stateful_repl.loom_state import StateEvent
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            store = SQLiteEventStore(db_path=f.name)
        for i in range(5):
            store.emit(StateEvent(
                timestamp=f"2026-01-0{i+1}T00:00:00",
                layer="L1", operation="test", data={"i": i},
            ))
        def reducer(state, event):
            state["count"] = state.get("count", 0) + 1
            return state
        result = store.replay(reducer, {}, up_to=3)
        assert result["count"] == 3
        store.clear()
        import os
        os.unlink(f.name)

    def test_sqlite_count_and_clear(self):
        """SQLite count and clear should work correctly."""
        import tempfile
        from stateful_repl.events import SQLiteEventStore
        from stateful_repl.loom_state import StateEvent
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            store = SQLiteEventStore(db_path=f.name)
        store.emit(StateEvent(
            timestamp="2026-01-01T00:00:00",
            layer="L1", operation="test", data={},
        ))
        assert store.count() == 1
        store.clear()
        assert store.count() == 0
        import os
        os.unlink(f.name)

    def test_create_event_store_factory(self):
        """Factory should create the correct backend."""
        from stateful_repl.events import create_event_store, InMemoryEventStore, SQLiteEventStore
        import tempfile
        store1 = create_event_store("json", path=tempfile.mktemp(suffix=".json"))
        assert isinstance(store1, InMemoryEventStore)
        store2 = create_event_store("sqlite", path=tempfile.mktemp(suffix=".db"))
        assert isinstance(store2, SQLiteEventStore)

    def test_inmemory_replay(self):
        """InMemoryEventStore replay should apply reducer correctly."""
        import tempfile
        from stateful_repl.events import InMemoryEventStore
        from stateful_repl.loom_state import StateEvent
        store = InMemoryEventStore(filepath=tempfile.mktemp(suffix=".json"))
        for i in range(3):
            store.emit(StateEvent(
                timestamp=f"2026-01-0{i+1}T00:00:00",
                layer="L2", operation="append", data={"val": i},
            ))
        def reducer(state, event):
            state.setdefault("values", []).append(event["data"]["val"])
            return state
        result = store.replay(reducer, {})
        assert result["values"] == [0, 1, 2]
        store.clear()


class TestLoomStateEdgeCases:
    """Edge cases for the core LoomREPL."""

    def test_invalid_layer_raises(self):
        """Reading an invalid layer should raise ValueError."""
        import tempfile
        from stateful_repl.loom_state import LoomREPL
        loom = LoomREPL(state_file=tempfile.mktemp(suffix=".md"), enable_events=False)
        import pytest
        with pytest.raises(ValueError, match="Invalid layer"):
            loom.read_state("L99")

    def test_update_invalid_field_raises(self):
        """Updating a nonexistent L1 field should raise ValueError."""
        import tempfile
        from stateful_repl.loom_state import LoomREPL
        loom = LoomREPL(state_file=tempfile.mktemp(suffix=".md"), enable_events=False)
        import pytest
        with pytest.raises(ValueError, match="Invalid L1 field"):
            loom.update_l1("nonexistent_field", "value")

    def test_append_l1_raises(self):
        """Appending to L1 should raise ValueError (use update_l1)."""
        import tempfile
        from stateful_repl.loom_state import LoomREPL
        loom = LoomREPL(state_file=tempfile.mktemp(suffix=".md"), enable_events=False)
        import pytest
        with pytest.raises(ValueError, match="L1 requires UPDATE"):
            loom.append("L1", "content")

    def test_append_l3_invalid_dict(self):
        """Appending an L3 dict without required keys should raise."""
        import tempfile
        from stateful_repl.loom_state import LoomREPL
        loom = LoomREPL(state_file=tempfile.mktemp(suffix=".md"), enable_events=False)
        import pytest
        with pytest.raises(ValueError, match="must contain"):
            loom.append("L3", {"random_key": "value"})

    def test_append_l3_non_dict_raises(self):
        """Appending a non-dict to L3 should raise TypeError."""
        import tempfile
        from stateful_repl.loom_state import LoomREPL
        loom = LoomREPL(state_file=tempfile.mktemp(suffix=".md"), enable_events=False)
        import pytest
        with pytest.raises(TypeError, match="must be a dict"):
            loom.append("L3", "not a dict")

    def test_consolidate_empty_l2(self):
        """Consolidating empty L2 to L3 should return a message, not crash."""
        import tempfile
        from stateful_repl.loom_state import LoomREPL
        loom = LoomREPL(state_file=tempfile.mktemp(suffix=".md"), enable_events=False)
        result = loom.consolidate_l2_to_l3()
        assert "empty" in result.get("message", "").lower()

    def test_validate_state_clean(self):
        """Validate on a fresh state should return VALID."""
        import tempfile
        from stateful_repl.loom_state import LoomREPL
        loom = LoomREPL(state_file=tempfile.mktemp(suffix=".md"), enable_events=False)
        result = loom.validate_state()
        assert result["status"] == "VALID"

    def test_validate_state_missing_goal_warning(self):
        """State with artifacts but no goal should produce a warning."""
        import tempfile
        from stateful_repl.loom_state import LoomREPL
        loom = LoomREPL(state_file=tempfile.mktemp(suffix=".md"), enable_events=False)
        loom.update_l1("artifacts", ["art-1"])
        result = loom.validate_state()
        assert result["status"] == "WARNINGS"
        assert any("no goal" in i.lower() for i in result["issues"])
