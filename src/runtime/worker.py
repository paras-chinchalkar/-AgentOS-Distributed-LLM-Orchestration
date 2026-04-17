"""
Background Worker — consumes tasks from Redis Streams and runs dynamic LangGraph workflows.
Supports parallel agent stages, per-agent status tracking, and retry broadcast events.
"""
import asyncio
import json
import logging

from src.core.db import AgentStatus
from src.core.message_bus import MessageBus
from src.runtime.lifecycle import LifecycleManager, MAX_RETRIES, DEFAULT_TOPOLOGY
from src.runtime.graph import build_dynamic_graph, _empty_state

logger = logging.getLogger(__name__)


def _all_agents_in_topology(topology: list[list[str]]) -> list[str]:
    """Flatten topology stages into a unique ordered list of agent names."""
    seen: set[str] = set()
    result: list[str] = []
    for stage in topology:
        for name in stage:
            if name not in seen:
                seen.add(name)
                result.append(name)
    return result


async def run_worker(bus: MessageBus, manager: LifecycleManager, worker_id: str = "worker-1") -> None:
    """
    Long-running coroutine that pulls workflow tasks from Redis Streams
    and executes the dynamic LangGraph agent pipeline.
    """
    logger.info("Worker %s started, listening on stream '%s'", worker_id, bus.stream)

    async for msg_id, payload in bus.consume(consumer_name=worker_id):
        if payload.get("type") != "workflow":
            continue

        workflow_id = payload["workflow_id"]
        goal = payload["goal"]
        name = payload["name"]
        is_retry = payload.get("is_retry", False)

        # Decode topology
        raw_topo = payload.get("topology")
        try:
            if isinstance(raw_topo, str):
                topology: list[list[str]] = json.loads(raw_topo)
            elif isinstance(raw_topo, list):
                topology = raw_topo
            else:
                topology = DEFAULT_TOPOLOGY
        except Exception:
            topology = DEFAULT_TOPOLOGY

        logger.info(
            "Worker %s picked up workflow %s (%s) | topology=%s | retry=%s",
            worker_id, workflow_id, name, topology, is_retry,
        )

        # Import agent registry
        from src.runtime.agents import AGENT_REGISTRY

        # Mark workflow as running
        await manager.update_workflow_status(workflow_id, AgentStatus.RUNNING)

        # Broadcast topology so UI can render the DAG
        await bus.broadcast_event("agent_events", {
            "event": "topology_resolved",
            "workflow_id": workflow_id,
            "topology": topology,
        })

        # Spawn agent records for every unique agent in the topology
        agent_names = _all_agents_in_topology(topology)
        agent_records: dict[str, object] = {}
        for agent_name in agent_names:
            plugin = AGENT_REGISTRY.get(agent_name)
            role = plugin["role"] if plugin else agent_name
            agent = await manager.spawn_agent(workflow_id, agent_name, role)
            agent_records[agent_name] = agent
            logger.info("Spawned agent %s (%s) for workflow %s", agent_name, role, workflow_id)

        # Mark all agents as running upfront so UI shows them active
        for agent_name in agent_names:
            await manager.update_agent_status(
                agent_records[agent_name].id,
                AgentStatus.RUNNING,
                increment_retry=(is_retry),
            )

        # ── Execute with retry ────────────────────────────────────────────
        attempt = 0
        success = False
        final_output = None

        while attempt < MAX_RETRIES and not success:
            attempt += 1
            try:
                logger.info("Workflow %s attempt %d starting", workflow_id, attempt)

                graph = build_dynamic_graph(topology)
                initial_state = _empty_state(goal)

                # Run LangGraph — collect the full final state
                # astream yields {node_name: state_dict} per node completion
                accumulated_state = dict(initial_state)
                stage_idx = 0

                async for event in graph.astream(initial_state):
                    for node_name, state_patch in event.items():
                        logger.info("Workflow %s: node '%s' completed", workflow_id, node_name)

                        if not isinstance(state_patch, dict):
                            continue

                        # Merge patch into accumulated state
                        for k, v in state_patch.items():
                            if k == "logs":
                                accumulated_state["logs"] = accumulated_state.get("logs", []) + (v if isinstance(v, list) else [v])
                            else:
                                accumulated_state[k] = v

                        # Map stage_N node back to the topology stage
                        # Node names are "stage_0", "stage_1", etc.
                        try:
                            node_stage_idx = int(node_name.split("_")[1])
                        except (IndexError, ValueError):
                            node_stage_idx = stage_idx

                        if node_stage_idx < len(topology):
                            stage_agents = topology[node_stage_idx]

                            # Broadcast stage completion
                            await bus.broadcast_event("agent_events", {
                                "event": "stage_completed",
                                "workflow_id": workflow_id,
                                "stage_index": node_stage_idx,
                                "agents": stage_agents,
                            })

                            # Update agent statuses for this stage
                            for agent_name in stage_agents:
                                if agent_name not in agent_records:
                                    continue
                                plugin = AGENT_REGISTRY.get(agent_name)
                                output_key = plugin["output_key"] if plugin else agent_name
                                output = accumulated_state.get(output_key, "")
                                await manager.update_agent_status(
                                    agent_records[agent_name].id,
                                    AgentStatus.COMPLETED,
                                    output=output or "",
                                )
                                logger.info("Agent %s completed for workflow %s", agent_name, workflow_id)

                        stage_idx += 1

                # Extract final output — prefer summary, then aggregated
                final_output = (
                    accumulated_state.get("summary")
                    or accumulated_state.get("aggregated")
                    or "Workflow complete."
                )
                logger.info("Workflow %s final output (%d chars)", workflow_id, len(final_output))
                success = True

            except Exception as exc:
                logger.error(
                    "Workflow %s attempt %d failed: %s",
                    workflow_id, attempt, exc, exc_info=True
                )
                if attempt >= MAX_RETRIES:
                    for agent_name in agent_names:
                        await manager.update_agent_status(
                            agent_records[agent_name].id,
                            AgentStatus.FAILED,
                            error=str(exc),
                        )
                    await manager.update_workflow_status(workflow_id, AgentStatus.FAILED)
                    await bus.broadcast_event("agent_events", {
                        "event": "workflow_failed",
                        "workflow_id": workflow_id,
                        "error": str(exc),
                        "attempt": attempt,
                    })
                else:
                    await manager.update_workflow_status(workflow_id, AgentStatus.RETRYING)
                    await bus.broadcast_event("agent_events", {
                        "event": "workflow_retrying",
                        "workflow_id": workflow_id,
                        "attempt": attempt,
                        "next_attempt_in": 2 ** attempt,
                    })
                    await asyncio.sleep(2 ** attempt)

        if success:
            # If the summarizer completed but produced no explicit output,
            # fall back to the workflow final output so the dashboard still
            # shows an executive summary.
            if final_output:
                summary_agent = agent_records.get("summarizer")
                if summary_agent:
                    await manager.update_agent_status(
                        summary_agent.id,
                        AgentStatus.COMPLETED,
                        output=accumulated_state.get("summary") or final_output,
                    )

            await manager.update_workflow_status(
                workflow_id, AgentStatus.COMPLETED, final_output=final_output
            )
            logger.info("Workflow %s completed successfully.", workflow_id)
