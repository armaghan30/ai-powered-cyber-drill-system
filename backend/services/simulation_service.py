import os
import uuid
import logging
import threading
import datetime as _dt
from datetime import datetime, timezone

import numpy as np
from sqlalchemy.orm import Session

from backend.config import settings
from backend.models.simulation import Simulation, SimulationStep, SimulationStatus
from backend.models.scenario import Scenario
from backend.schemas.simulation import SimulationCreate

logger = logging.getLogger(__name__)

# In-memory store: session_id -> Orchestrator instance
_sessions = {}
_sessions_lock = threading.Lock()


def _get_topology_path(scenario: Scenario) -> str:
    """Return topology path as string, matching orchestrator script expectations."""
    return str(settings.TOPOLOGY_DIR / scenario.filename)


def _topo_tag_from_path(topology_path: str) -> str:
    """Derive the topology tag used in model filenames.

    Training scripts save models as ``sb3_dqn_{role}_{topo_tag}.zip`` where
    *topo_tag* is the topology filename without its extension.
    """
    return os.path.splitext(os.path.basename(topology_path))[0]


def _try_load_sb3_models(topology_path: str):
    try:
        from stable_baselines3 import DQN
    except ImportError:
        logger.warning("stable_baselines3 not installed – falling back to rule-based agents")
        return None, None

    topo_tag = _topo_tag_from_path(topology_path)
    red_path = settings.MODELS_DIR / f"sb3_dqn_red_{topo_tag}.zip"
    blue_path = settings.MODELS_DIR / f"sb3_dqn_blue_{topo_tag}.zip"

    if not red_path.exists() or not blue_path.exists():
        logger.info(
            "Trained models not found for topology '%s' (looked for %s and %s) – "
            "falling back to rule-based agents",
            topo_tag, red_path, blue_path,
        )
        return None, None

    logger.info("Loading SB3 DQN models for topology '%s'", topo_tag)
    red_model = DQN.load(str(red_path))
    blue_model = DQN.load(str(blue_path))
    return red_model, blue_model


# ---------------------------------------------------------------------------
# Action decode helpers (mirror rl_env_red / rl_env_blue)
# ---------------------------------------------------------------------------

def _decode_red_action(index: int, host_order, red_agent):

    n = len(host_order)
    if index < n:
        return red_agent.scan(host_order[index])
    elif index < 2 * n:
        return red_agent.exploit(host_order[index - n])
    elif index < 3 * n:
        return red_agent.escalate_privileges(host_order[index - 2 * n])
    elif index < 4 * n:
        return red_agent.lateral_move(host_order[index - 3 * n])
    elif index < 5 * n:
        return red_agent.exfiltrate(host_order[index - 4 * n])
    else:
        return {"action": "idle", "target": None}


def _decode_blue_action(index: int, host_order, blue_agent):
    n = len(host_order)
    if index < n:
        return {"action": "patch", "target": host_order[index]}
    elif index < 2 * n:
        return {"action": "isolate", "target": host_order[index - n]}
    elif index < 3 * n:
        return {"action": "restore", "target": host_order[index - 2 * n]}
    elif index < 4 * n:
        return blue_agent.make_detect_action(host_order[index - 3 * n])
    elif index < 5 * n:
        return {"action": "harden", "target": host_order[index - 4 * n]}
    else:
        return {"action": "idle", "target": None}


# ---------------------------------------------------------------------------
# SB3-powered simulation loop
# ---------------------------------------------------------------------------

def _run_sb3_simulation_step(orch, red_model, blue_model, host_order):
   
    from orchestrator.state_vectors import flatten_red_state, flatten_blue_state

    # 1. Build observations
    red_state = orch._build_red_state()
    blue_state = orch._build_blue_state()

    red_obs = flatten_red_state(red_state, host_order)
    blue_obs = flatten_blue_state(blue_state, host_order)

    # 2. Model predictions
    red_action_idx, _ = red_model.predict(red_obs, deterministic=True)
    blue_action_idx, _ = blue_model.predict(blue_obs, deterministic=True)

    # 3. Decode to action dicts (using agent methods for execution side-effects)
    red_action = _decode_red_action(int(red_action_idx), host_order, orch.red_agent)
    blue_action = _decode_blue_action(int(blue_action_idx), host_order, orch.blue_agent)

    # 4. Snapshot before step (for reward computation)
    if orch.logs:
        prev_state = orch.logs[-1]["environment"]
    else:
        prev_state = orch._snapshot_environment()

    # 5. Step the environment
    env_state = orch.environment.step(red_action, blue_action)

    # 6. Compute rewards
    red_reward, blue_reward = orch.reward_engine.compute_rewards(
        prev_state, env_state, red_action, blue_action
    )

    # 7. Build log entry (same schema as run_single_step)
    log_entry = {
        "step": len(orch.logs) + 1,
        "timestamp": _dt.datetime.now().isoformat(),
        "red_state": red_state,
        "blue_state": blue_state,
        "red_action": red_action,
        "blue_action": blue_action,
        "red_reward": red_reward,
        "blue_reward": blue_reward,
        "environment": env_state,
    }
    orch.logs.append(log_entry)
    return log_entry


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def create_simulation(db: Session, payload: SimulationCreate) -> Simulation:
    
    from orchestrator.orchestrator_core import Orchestrator

    scenario = db.query(Scenario).filter(Scenario.id == payload.scenario_id).first()
    if not scenario:
        raise ValueError(f"Scenario {payload.scenario_id} not found")

    session_id = str(uuid.uuid4())
    topology_path = _get_topology_path(scenario)

    # Create and initialize orchestrator (agents needed for action execution)
    orch = Orchestrator(topology_path)
    orch.load_topology()
    orch.build_environment()
    orch.init_red_agent()
    orch.init_blue_agent()

    with _sessions_lock:
        _sessions[session_id] = orch

    simulation = Simulation(
        scenario_id=payload.scenario_id,
        session_id=session_id,
        status=SimulationStatus.RUNNING,
        max_steps=payload.max_steps,
    )
    db.add(simulation)
    db.commit()
    db.refresh(simulation)
    return simulation


def run_step(db: Session, session_id: str) -> dict:
    simulation = db.query(Simulation).filter(
        Simulation.session_id == session_id
    ).first()
    if not simulation:
        raise ValueError(f"Simulation not found for session {session_id}")
    if simulation.status != SimulationStatus.RUNNING:
        raise ValueError(f"Simulation is {simulation.status.value}, not running")

    with _sessions_lock:
        orch = _sessions.get(session_id)
    if not orch:
        raise ValueError(f"No active orchestrator for session {session_id}")

    # Execute one step
    log_entry = orch.run_single_step()

    step = SimulationStep(
        simulation_id=simulation.id,
        step_number=log_entry["step"],
        red_action=log_entry["red_action"],
        blue_action=log_entry["blue_action"],
        red_reward=log_entry["red_reward"],
        blue_reward=log_entry["blue_reward"],
        red_state=log_entry["red_state"],
        blue_state=log_entry["blue_state"],
        environment_state=log_entry["environment"],
    )
    db.add(step)

    simulation.total_steps = log_entry["step"]
    simulation.total_red_reward += log_entry["red_reward"]
    simulation.total_blue_reward += log_entry["blue_reward"]

    # Check completion
    steps_remaining = simulation.max_steps - simulation.total_steps
    if steps_remaining <= 0:
        simulation.status = SimulationStatus.COMPLETED
        simulation.completed_at = datetime.now(timezone.utc)
        simulation.final_state = log_entry["environment"]
        with _sessions_lock:
            _sessions.pop(session_id, None)

    db.commit()
    db.refresh(step)

    return {
        "step": step,
        "simulation_status": simulation.status.value,
        "steps_remaining": max(0, steps_remaining),
    }


def run_full_simulation(db: Session, payload: SimulationCreate) -> Simulation:
    
    from orchestrator.orchestrator_core import Orchestrator

    # --- Resolve scenario & topology ------------------------------------------
    scenario = db.query(Scenario).filter(Scenario.id == payload.scenario_id).first()
    if not scenario:
        raise ValueError(f"Scenario {payload.scenario_id} not found")

    session_id = str(uuid.uuid4())
    topology_path = _get_topology_path(scenario)

    # --- Build orchestrator (agents always needed for action execution) --------
    orch = Orchestrator(topology_path)
    orch.load_topology()
    orch.build_environment()
    orch.init_red_agent()
    orch.init_blue_agent()

    with _sessions_lock:
        _sessions[session_id] = orch

    # --- Persist the simulation record ----------------------------------------
    simulation = Simulation(
        scenario_id=payload.scenario_id,
        session_id=session_id,
        status=SimulationStatus.RUNNING,
        max_steps=payload.max_steps,
    )
    db.add(simulation)
    db.commit()
    db.refresh(simulation)

    # --- Try to load trained SB3 models ---------------------------------------
    red_model, blue_model = _try_load_sb3_models(topology_path)
    use_sb3 = red_model is not None and blue_model is not None

    if use_sb3:
        host_order = list(orch.environment.hosts.keys())
        logger.info(
            "Running SB3-powered simulation (session=%s, topology=%s, hosts=%d)",
            session_id, _topo_tag_from_path(topology_path), len(host_order),
        )

    # --- Simulation loop ------------------------------------------------------
    for _ in range(payload.max_steps):
        if use_sb3:
            log_entry = _run_sb3_simulation_step(orch, red_model, blue_model, host_order)
        else:
            log_entry = orch.run_single_step()

        # Persist step
        step = SimulationStep(
            simulation_id=simulation.id,
            step_number=log_entry["step"],
            red_action=log_entry["red_action"],
            blue_action=log_entry["blue_action"],
            red_reward=log_entry["red_reward"],
            blue_reward=log_entry["blue_reward"],
            red_state=log_entry["red_state"],
            blue_state=log_entry["blue_state"],
            environment_state=log_entry["environment"],
        )
        db.add(step)

        simulation.total_steps = log_entry["step"]
        simulation.total_red_reward += log_entry["red_reward"]
        simulation.total_blue_reward += log_entry["blue_reward"]

    # --- Finalize -------------------------------------------------------------
    simulation.status = SimulationStatus.COMPLETED
    simulation.completed_at = datetime.now(timezone.utc)
    simulation.final_state = orch.logs[-1]["environment"] if orch.logs else None

    with _sessions_lock:
        _sessions.pop(session_id, None)

    db.commit()
    db.refresh(simulation)
    return simulation


# ---------------------------------------------------------------------------
# Read-only helpers (unchanged)
# ---------------------------------------------------------------------------

def get_simulation(db: Session, simulation_id: int) -> Simulation:
    return db.query(Simulation).filter(Simulation.id == simulation_id).first()


def list_simulations(db: Session, scenario_id: int = None) -> list:
    query = db.query(Simulation).order_by(Simulation.created_at.desc())
    if scenario_id:
        query = query.filter(Simulation.scenario_id == scenario_id)
    return query.all()


def cleanup_session(session_id: str) -> None:
    with _sessions_lock:
        _sessions.pop(session_id, None)
