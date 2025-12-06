# orchestrator/rl_env_blue.py
"""
Gymnasium-compatible RL environment for training the BLUE agent.

Blue chooses: patch, isolate, or idle.
Red is controlled by a trained DQN model (if available) or falls back to random.
"""

from __future__ import annotations
import random
from typing import Dict, Any, Tuple, List

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from orchestrator.orchestrator_core import Orchestrator
from orchestrator.reward_engine import RewardEngine
from orchestrator.dqn_agent_red import DQNAgentRed
from orchestrator.rl_env_red import flatten_red_state


# -----------------------------------------------------------
# BLUE STATE ENCODING
# -----------------------------------------------------------
def flatten_blue_state(env_state: Dict[str, Any]) -> np.ndarray:
    """
    Converts environment snapshot → Blue’s observation vector.
    Now includes:
    - per-host features
    - global summarized features
    """

    hosts = env_state.get("hosts", {})
    host_names = sorted(hosts.keys())
    edges = env_state.get("edges", [])

    features: List[float] = []

    compromised_count = 0
    vulnerable_count = 0
    patched_count = 0
    total_vulns = 0

    # ------- Per-host features -------
    for name in host_names:
        h = hosts[name]

        is_comp = 1.0 if h.get("is_compromised", False) else 0.0
        vuln_list = h.get("vulnerabilities", [])
        vuln_count = float(len(vuln_list))

        compromised_count += is_comp
        total_vulns += vuln_count
        if vuln_count > 0:
            vulnerable_count += 1
        else:
            patched_count += 1

        features.extend([
            is_comp,
            vuln_count,
        ])

    # ------- Global summary features -------
    n_hosts = len(host_names)
    avg_vulns = total_vulns / n_hosts if n_hosts > 0 else 0.0

    features.extend([
        float(compromised_count),
        float(vulnerable_count),
        float(patched_count),
        float(avg_vulns),
        float(len(edges)),  # connectivity
        float(env_state.get("step", 0)),
    ])

    return np.array(features, dtype=np.float32)



# -----------------------------------------------------------
# BLUE RL ENVIRONMENT
# -----------------------------------------------------------
class BlueRLEnvironment(gym.Env):
    """
    RL environment for training the BLUE agent.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, topology_path: str, max_steps: int = 20):
        super().__init__()

        self.topology_path = topology_path
        self.max_steps = max_steps

        self.orch: Orchestrator | None = None
        self.reward_engine = RewardEngine()

        self.current_step = 0
        self.host_order: List[str] | None = None
        self.num_blue_actions: int | None = None

        self.observation_space: spaces.Box | None = None
        self.action_space: spaces.Discrete | None = None

        # Red attacker (will try to load pretrained model)
        self.red_agent: DQNAgentRed | None = None
        self.red_agent_ready = False
        self.num_red_actions: int | None = None

    # -----------------------------------------------------------
    # BLUE ACTIONS = idle, patch(H1..HN), isolate(H1..HN)
    # -----------------------------------------------------------
    def _build_action_space(self):
        """
        Action mapping:
         0                   -> idle
         1..N               -> patch host_i
         N+1..2N            -> isolate host_i
        """
        n = len(self.host_order)
        self.num_blue_actions = 1 + 2 * n
        self.action_space = spaces.Discrete(self.num_blue_actions)

    def _decode_blue_action(self, action_id: int) -> Dict[str, Any]:
        n = len(self.host_order)

        if action_id == 0:
            return {"action": "idle"}

        if 1 <= action_id <= n:
            host = self.host_order[action_id - 1]
            return {"action": "patch", "target": host}

        if n < action_id <= 2 * n:
            host = self.host_order[action_id - n - 1]
            return {"action": "isolate", "target": host}

        raise ValueError(f"Invalid action_id: {action_id}")

    # -----------------------------------------------------------
    # Load Trained RED Model (if exists)
    # -----------------------------------------------------------
    def _init_red_agent(self):
        if self.red_agent_ready:
            return

        # Construct temporary red_state
        red_state = self.orch.get_red_state()
        red_vec = flatten_red_state(red_state, self.host_order)
        state_dim = red_vec.shape[0]

        self.num_red_actions = 2 * len(self.host_order)

        try:
            print("[BLUE ENV] Loading trained RED model...")
            self.red_agent = DQNAgentRed(
                state_dim=state_dim,
                action_dim=self.num_red_actions,
            )
            self.red_agent.load("red_dqn_model.pth")
            self.red_agent_ready = True
            print("[BLUE ENV] Red model loaded successfully.")
        except Exception as e:
            print("[BLUE ENV] No trained Red model found. Using random attacker.")
            print("Error:", e)
            self.red_agent = None
            self.red_agent_ready = False

    # -----------------------------------------------------------
    # RESET
    # -----------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.orch = Orchestrator(self.topology_path)
        self.orch.load_topology()
        self.orch.build_environment()
        self.orch.init_red_agent()
        self.orch.init_blue_agent()

        self.current_step = 0
        self.host_order = list(self.orch.environment.hosts.keys())

        self._build_action_space()

        env_state = self.orch._snapshot_environment()
        obs_vec = flatten_blue_state(env_state)

        if self.observation_space is None:
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(obs_vec.shape[0],),
                dtype=np.float32,
            )

        # Red attacker initialization
        self._init_red_agent()

        return obs_vec, {}

    # -----------------------------------------------------------
    # STEP
    # -----------------------------------------------------------
    def step(self, action_id: int):
        if self.orch is None:
            raise RuntimeError("Call reset() before step().")

        prev_snapshot = self.orch._snapshot_environment()

        # 1) RED attacker moves
        if self.red_agent_ready:
            red_state = self.orch.get_red_state()
            red_vec = flatten_red_state(red_state, self.host_order)
            action_red = self.red_agent.select_action_eval(red_vec)

            n = len(self.host_order)
            if action_red < n:
                target = self.host_order[action_red]
                red_action = self.orch.red_agent.scan(target)
            else:
                target = self.host_order[action_red - n]
                red_action = self.orch.red_agent.exploit(target)
        else:
            # fallback: random attacker
            target = random.choice(self.host_order)
            if random.random() < 0.5:
                red_action = self.orch.red_agent.scan(target)
            else:
                red_action = self.orch.red_agent.exploit(target)

        # 2) BLUE RL action
        blue_action = self._decode_blue_action(action_id)

        # 3) Apply both to environment
        _ = self.orch.environment.step(red_action, blue_action)

        # 4) New snapshot
        new_snapshot = self.orch._snapshot_environment()

        # 5) Reward
        red_r, blue_r = self.reward_engine.compute_rewards(
            prev_snapshot, new_snapshot, red_action, blue_action
        )

        # 6) Next state vector
        next_obs = flatten_blue_state(new_snapshot)

        # 7) Episode done?
        self.current_step += 1
        terminated = False
        truncated = self.current_step >= self.max_steps

        info = {
            "step": self.current_step,
            "red_action": red_action,
            "blue_action": blue_action,
            "env_state": new_snapshot,
            "red_reward": red_r,
        }

        return next_obs, float(blue_r), terminated, truncated, info

    # -----------------------------------------------------------
    def render(self):
        print(self.orch._snapshot_environment())

    def close(self):
        self.orch = None
