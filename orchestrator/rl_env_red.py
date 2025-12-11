# orchestrator/rl_env_red.py

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
from gymnasium import spaces

from orchestrator.orchestrator_core import Orchestrator
from orchestrator.state_vectors import flatten_red_state
from orchestrator.reward_engine import RewardEngine


class RedRLEnvironment:
    """
    Gymnasium-style RL environment for the RED DQN agent.

    Observation: flat vector from flatten_red_state(...)

    Discrete actions (for N hosts):
      0 .. N-1      : scan host_i
      N .. 2N-1     : exploit host_(i-N)
      2N            : idle
    """

    def __init__(self, topology_path: str, max_steps: int = 20, seed: int | None = None):
        self.topology_path = topology_path
        self.max_steps = max_steps
        self.current_step = 0

        # Build orchestrator + environment once; reset() will rebuild for each episode
        self._build_orchestrator()

        self.reward_engine = RewardEngine()
        self.np_random = np.random.default_rng(seed)

        # Determine state_dim / action_dim from example state
        red_state = self.orch.get_red_state()
        self.host_order: List[str] = list(self.environment.hosts.keys())

        example_vec = flatten_red_state(red_state, self.host_order)
        self.state_dim = int(example_vec.shape[0])

        # Actions = 2 * n_hosts + 1 (scan, exploit, idle)
        self.action_dim = 2 * len(self.host_order) + 1

        self.observation_space = spaces.Box(
            low=0.0, high=10.0, shape=(self.state_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(self.action_dim)

        print(f"[INFO] RED RL Env -> state_dim={self.state_dim}, action_dim={self.action_dim}")

    # ------------------------------------------------------------
    def _build_orchestrator(self):
        """Create a fresh orchestrator + environment."""
        self.orch = Orchestrator(self.topology_path)
        self.orch.load_topology()
        self.orch.build_environment()
        self.orch.init_red_agent()
        self.orch.init_blue_agent()

        self.environment = self.orch.environment
        self.red_agent = self.orch.red_agent
        self.blue_agent = self.orch.blue_agent

    # ------------------------------------------------------------
    def _decode_red_action(self, index: int) -> Dict[str, Any]:
        """
        Convert an action index to a concrete red_action dict,
        using the same RedAgent as the orchestrator so that:
          - red_agent.known_vulns is updated on scans
          - exploits get a realistic 'success' flag
        All actions use key 'action' (NOT 'type').
        """
        n = len(self.host_order)

        if index < 0 or index >= self.action_dim:
            raise ValueError(f"Invalid RED action index: {index}")

        if index < n:
            # Scan host_i
            host = self.host_order[index]
            return self.red_agent.scan(host)
        elif index < 2 * n:
            # Exploit host_(i-n)
            host = self.host_order[index - n]
            return self.red_agent.exploit(host)
        else:
            # Idle
            return {"action": "idle", "target": None}

    # ------------------------------------------------------------
    def reset(self, seed: int | None = None, options: Dict | None = None):
        """
        Gymnasium-style reset:
        returns (obs, info)
        """
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        self.current_step = 0
        # Rebuild environment each episode for a clean start
        self._build_orchestrator()

        red_state = self.orch.get_red_state()
        self.host_order = list(self.environment.hosts.keys())

        obs_vec = flatten_red_state(red_state, self.host_order)
        return np.array(obs_vec, dtype=np.float32), {}

    # ------------------------------------------------------------
    def step(self, action_index: int):
        """
        Gymnasium-style step:
        returns (obs, reward, terminated, truncated, info)
        """
        self.current_step += 1

        # Decode RED action (this also updates RedAgent internal knowledge)
        red_action = self._decode_red_action(int(action_index))

        # BLUE agent uses its own policy (rule-based),
        # and expects action dicts with key 'action'
        blue_action = self.blue_agent.choose_action(red_action)

        prev_state = self.orch._snapshot_environment()

        # Apply actions to environment
        env_state = self.environment.step(red_action, blue_action)

        # Reward for RED
        red_reward, blue_reward = self.reward_engine.compute_rewards(
            prev_state, env_state, red_action, blue_action
        )

        # Build next observation
        red_state = self.orch.get_red_state()
        obs_vec = flatten_red_state(red_state, self.host_order)

        terminated = self.current_step >= self.max_steps
        truncated = False

        info = {
            "red_action": red_action,
            "blue_action": blue_action,
            "environment": env_state,
        }

        return np.array(obs_vec, dtype=np.float32), float(red_reward), terminated, truncated, info
