# rl_env_red.py
"""
RedRLEnvironment: A Gymnasium-compatible RL environment wrapper
for training the Red agent, using the existing Orchestrator.

Usage:

    from orchestrator.rl_env_red import RedRLEnvironment

    env = RedRLEnvironment("orchestrator/sample_topology.yaml", max_steps=20)
    obs, info = env.reset()

    done = False
    while not done:
        action_id = policy.select_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action_id)
        done = terminated or truncated
        obs = next_obs

Notes:
- Blue remains rule-based (heuristic).
- Red uses DQN to decide WHICH host to scan/exploit,
  but the actual scan/exploit logic is still handled by RedAgent.
"""

from __future__ import annotations

import random
from typing import List, Dict, Any, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from orchestrator.orchestrator_core import Orchestrator


# --------------------------------------------------
# Helper: flatten red_state dict -> numeric vector
# --------------------------------------------------
def flatten_red_state(state: dict, host_order: List[str]) -> np.ndarray:
    """
    Convert structured red_state dict into a flat numeric vector.

    red_state format:
    {
        "timestep": int,
        "hosts": {
            host_name: {
                "scanned": int,
                "vulnerabilities": int,
                "services": int,
                "is_compromised": int,
                "access_level": int,
            },
            ...
        },
        "num_hosts": int,
        "num_compromised": int
    }

    For N hosts, this yields 5*N + 3 features.
    """
    hosts = state["hosts"]
    features = []

    # Per-host features in fixed order
    for name in host_order:
        h = hosts.get(name)
        if h is None:
            features.extend([0, 0, 0, 0, 0])
        else:
            features.extend([
                h["scanned"],
                h["vulnerabilities"],
                h["services"],
                h["is_compromised"],
                h["access_level"],
            ])

    # Global features
    features.append(state["timestep"])
    features.append(state["num_hosts"])
    features.append(state["num_compromised"])

    return np.array(features, dtype=np.float32)


class RedRLEnvironment(gym.Env):
    """
    Gymnasium-compatible environment wrapping your Orchestrator pipeline
    for Red-agent RL training.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, topology_path: str, max_steps: int = 20):
        """
        :param topology_path: path to the topology YAML file
        :param max_steps: maximum steps per episode
        """
        super().__init__()

        self.topology_path = topology_path
        self.max_steps = max_steps

        self.orch: Orchestrator | None = None
        self.current_step = 0

        self.num_red_actions: int | None = None
        self.host_order: list[str] | None = None

        # Gymnasium spaces will be initialized lazily on first reset(),
        # when we know how many hosts there are.
        self.observation_space: spaces.Box | None = None
        self.action_space: spaces.Discrete | None = None

    # -----------------------------------------
    # RESET: start a fresh episode
    # -----------------------------------------
    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, dict]:
        """
        Initializes a fresh orchestrator + environment + agents.
        Returns the initial Red observation (flattened vector) and info.
        """
        super().reset(seed=seed)

        self.orch = Orchestrator(self.topology_path)
        self.orch.load_topology()
        self.orch.build_environment()
        self.orch.init_red_agent()
        self.orch.init_blue_agent()

        self.current_step = 0

        # Host order for mapping action IDs to (host, action_type)
        self.host_order = list(self.orch.environment.hosts.keys())

        # Red action space: for N hosts ->
        # 0..N-1   : scan host_i
        # N..2N-1  : exploit host_i
        n_hosts = len(self.host_order)
        self.num_red_actions = 2 * n_hosts

        # Initial observation (before any action) as dict
        red_state = self.orch.get_red_state()
        obs_vec = flatten_red_state(red_state, self.host_order)

        # Lazily define Gymnasium spaces
        if self.observation_space is None:
            state_dim = obs_vec.shape[0]
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(state_dim,),
                dtype=np.float32,
            )

        if self.action_space is None:
            self.action_space = spaces.Discrete(self.num_red_actions)

        info: dict = {}
        return obs_vec, info

    # -----------------------------------------
    # STEP: apply one Red action and advance env
    # -----------------------------------------
    def step(self, action_id: int):
        """
        Take one environment step using a discrete Red action_id.
        Blue uses its existing rule-based policy.

        Returns (obs, reward, terminated, truncated, info)
        following Gymnasium API.
        """
        if self.orch is None:
            raise RuntimeError("Environment not reset. Call reset() first.")
        if self.host_order is None or self.num_red_actions is None:
            raise RuntimeError("Environment not properly initialized. Call reset() first.")

        # 1) Snapshot environment BEFORE actions (for reward computation)
        prev_state = self.orch._snapshot_environment()

        # 2) Decode action_id into (type, target_host),
        #    then call REAL RedAgent methods (scan/exploit)
        n = len(self.host_order)
        if action_id < 0 or action_id >= self.num_red_actions:
            raise ValueError(f"Invalid action_id {action_id}, valid range is [0, {self.num_red_actions - 1}]")

        if action_id < n:
            # SCAN
            target = self.host_order[action_id]
            red_action = self.orch.red_agent.scan(target)
        else:
            # EXPLOIT
            idx = action_id - n
            target = self.host_order[idx]
            red_action = self.orch.red_agent.exploit(target)

        # 3) Blue responds using its rule-based policy
        blue_action = self.orch.blue_agent.choose_action(red_action)

        # 4) Apply environment step using red_action + blue_action
        env_state = self.orch.environment.step(red_action, blue_action)

        # 5) Compute Red reward (we ignore Blue reward here)
        red_reward, _ = self.orch.reward_engine.compute_rewards(
            prev_state, env_state, red_action, blue_action
        )

        # 6) Build next Red observation (dict -> vector)
        red_state_next = self.orch.get_red_state()
        next_obs_vec = flatten_red_state(red_state_next, self.host_order)

        # 7) Episode termination condition (simple time limit here)
        self.current_step += 1
        done = self.current_step >= self.max_steps

        # Gymnasium splits done into terminated vs truncated.
        # Here our "done" is purely a time-limit, so we treat it as truncation.
        terminated = False           # no terminal success/failure condition modeled (yet)
        truncated = done

        # 8) Optional info for debugging / logging
        info = {
            "step": self.current_step,
            "red_action": red_action,
            "blue_action": blue_action,
            "environment": env_state,
        }

        return next_obs_vec, float(red_reward), terminated, truncated, info


# ---------------------------------------------------
# Simple manual test when running this file directly
# ---------------------------------------------------
if __name__ == "__main__":
    topo = "orchestrator/sample_topology.yaml"
    env = RedRLEnvironment(topo, max_steps=5)

    obs, info = env.reset()
    print("[TEST] Initial state vector shape:", obs.shape)

    done = False
    total_reward = 0.0

    while not done:
        # Random policy for testing: pick random action ID
        action_id = random.randint(0, env.num_red_actions - 1)
        next_obs, reward, terminated, truncated, info = env.step(action_id)
        done = terminated or truncated
        total_reward += reward
        print(
            f"[STEP {info['step']}] action_id={action_id}, "
            f"red_action={info['red_action']}, "
            f"reward={reward}, done={done}"
        )

        obs = next_obs

    print("[TEST] Episode finished. Total Red reward:", total_reward)
