# orchestrator/eval_both_agents.py

"""
Evaluate BOTH trained agents together:

- Red: DQN-based attacker (trained with Gymnasium-style wrapper).
- Blue: DQN-based defender (trained with its own RL environment).

This script:
  1. Builds the Orchestrator + Environment from sample_topology.yaml
  2. Loads red_dqn_model.pth and blue_dqn_model.pth
  3. Runs a fixed-length episode where:
       - Red chooses scan/exploit using its DQN policy
       - Blue chooses patch/isolate/idle using its DQN policy
       - Environment applies the actions
       - RewardEngine computes per-step rewards
  4. Prints the interaction + cumulative rewards.

You will use this for:
  - Demo to evaluators
  - Explaining MARL-style attack–defense dynamics
"""

from __future__ import annotations

import numpy as np
import torch

from .orchestrator_core import Orchestrator
from .dqn_agent_red import DQNAgentRed
from .rl_env_red import flatten_red_state
from .reward_engine import RewardEngine


# -----------------------------------------------------------------------
# LOAD TRAINED RED MODEL
# -----------------------------------------------------------------------
def load_red_agent(state_dim: int, action_dim: int) -> DQNAgentRed:
    """
    Create a DQNAgentRed with the correct dimensions and load weights
    from red_dqn_model.pth.
    """
    print("[EVAL] Loading trained RED model...")

    red = DQNAgentRed(
        state_dim=state_dim,
        action_dim=action_dim,
        epsilon_start=0.0,  # no exploration in eval
        epsilon_end=0.0,
        epsilon_decay=1,
    )

    red.load("red_dqn_model.pth")
    print("[EVAL] RED model loaded successfully.")
    return red


# -----------------------------------------------------------------------
# LOAD TRAINED BLUE MODEL
# -----------------------------------------------------------------------
def load_blue_agent() -> tuple[DQNAgentRed, int, int]:
    """
    Load blue_dqn_model.pth and create a DQNAgentRed to act as Blue's policy.

    We stored in the checkpoint:
      - state_dim
      - action_dim
      - online_state_dict
      - target_state_dict
    """
    print("[EVAL] Loading trained BLUE model...")

    checkpoint = torch.load("blue_dqn_model.pth", map_location="cpu")

    saved_state_dim = checkpoint["state_dim"]
    saved_action_dim = checkpoint["action_dim"]

    print(f"[EVAL] Blue model expects state_dim = {saved_state_dim}")
    print(f"[EVAL] Blue model expects action_dim = {saved_action_dim}")

    blue = DQNAgentRed(
        state_dim=saved_state_dim,
        action_dim=saved_action_dim,
        epsilon_start=0.0,  # no exploration in eval
        epsilon_end=0.0,
        epsilon_decay=1,
    )

    blue.policy_net.load_state_dict(checkpoint["online_state_dict"])
    blue.target_net.load_state_dict(checkpoint["target_state_dict"])

    print("[EVAL] BLUE model loaded successfully.")
    return blue, saved_state_dim, saved_action_dim


# -----------------------------------------------------------------------
# BLUE ACTION DECODER (for evaluation)
# -----------------------------------------------------------------------
def decode_blue_action(
    action_id: int,
    host_order: list[str],
    blue_action_dim: int,
) -> dict:
    """
    Map Blue's discrete action_id to a concrete action dict that your
    Environment + RewardEngine understand.

    For simplicity (and to keep it easy to explain in FYP):
      - 0 -> patch H1
      - 1 -> patch H2
      - 2 -> isolate H1
      - 3 -> isolate H2
      - 4 (and any others) -> idle

    This matches blue_action_dim = 5 (from training logs).
    If you later extend to more hosts/actions, you can refine this mapping.
    """
    n_hosts = len(host_order)

    # Safety: if we somehow have fewer than 2 hosts, just idle
    if n_hosts == 0:
        return {"action": "idle"}

    # Ensure we don't crash on unexpected dims
    if blue_action_dim < 4:
        # Very defensive: treat everything as idle if space is unknown
        return {"action": "idle"}

    # Hard-coded mapping for the current 2-host example
    # (You can generalize this later.)
    if action_id == 0:
        return {"action": "patch", "target": host_order[0]}  # H1
    elif action_id == 1:
        target = host_order[1] if n_hosts > 1 else host_order[0]
        return {"action": "patch", "target": target}         # H2
    elif action_id == 2:
        return {"action": "isolate", "target": host_order[0]}  # H1
    elif action_id == 3:
        target = host_order[1] if n_hosts > 1 else host_order[0]
        return {"action": "isolate", "target": target}         # H2
    else:
        # Any remaining index -> idle (no-op)
        return {"action": "idle"}


# -----------------------------------------------------------------------
# MAIN EVALUATION LOOP
# -----------------------------------------------------------------------
def main():
    topology_path = "orchestrator/sample_topology.yaml"

    print("Loading topology...")
    orch = Orchestrator(topology_path)
    orch.load_topology()
    orch.build_environment()
    orch.init_red_agent()
    orch.init_blue_agent()
    print("Environment Ready.")

    # Host list for mapping indices -> host names
    host_order = list(orch.environment.hosts.keys())
    n_hosts = len(host_order)

    # ---------------- RED DIMENSIONS ----------------
    # Red has 2 actions per host (scan, exploit)
    red_action_dim = 2 * n_hosts

    red_init_state = orch.get_red_state()
    red_state_vec_example = flatten_red_state(red_init_state, host_order)
    red_state_dim = red_state_vec_example.shape[0]

    # ---------------- LOAD AGENTS ----------------
    red_agent = load_red_agent(red_state_dim, red_action_dim)
    blue_agent, blue_state_dim, blue_action_dim = load_blue_agent()

    reward_engine = RewardEngine()

    # -----------------------------------------------------------------------
    # RUN ONE EVALUATION EPISODE
    # -----------------------------------------------------------------------
    max_steps = 20
    total_red_reward = 0.0
    total_blue_reward = 0.0

    print("\n=========== EVALUATION EPISODE START ===========")

    prev_snapshot = orch._snapshot_environment()

    for step in range(max_steps):
        print(f"\n--- Step {step} ---")

        # -----------------------
        # RED chooses action (DQN)
        # -----------------------
        red_state = orch.get_red_state()
        red_state_vec = flatten_red_state(red_state, host_order)
        red_state_vec = np.array(red_state_vec, dtype=np.float32)

        red_action_id = red_agent.select_action_eval(red_state_vec)

        # Map discrete action to actual RedAgent method
        if red_action_id < n_hosts:
            target = host_order[red_action_id]
            red_action = orch.red_agent.scan(target)
        else:
            target = host_order[red_action_id - n_hosts]
            red_action = orch.red_agent.exploit(target)

        print("RED →", red_action)

        # -----------------------
        # BLUE chooses action (DQN)
        # -----------------------
        blue_obs_vec = orch.get_blue_state_vector()
        blue_obs_vec = np.array(blue_obs_vec, dtype=np.float32)

        blue_action_id = blue_agent.select_action_eval(blue_obs_vec)
        blue_action = decode_blue_action(blue_action_id, host_order, blue_action_dim)

        print("BLUE →", blue_action)

        # -----------------------
        # STEP ENVIRONMENT
        # -----------------------
        env_state = orch.environment.step(red_action, blue_action)

        # Compute rewards based on before/after snapshot + actions
        red_reward, blue_reward = reward_engine.compute_rewards(
            prev_snapshot, env_state, red_action, blue_action
        )

        total_red_reward += red_reward
        total_blue_reward += blue_reward

        print(f"Red Reward = {red_reward:.2f} | Blue Reward = {blue_reward:.2f}")

        # Next step will compare against this snapshot
        prev_snapshot = env_state

    print("\n=========== EVALUATION COMPLETE ===========")
    print(f"Total RED Reward  = {total_red_reward:.2f}")
    print(f"Total BLUE Reward = {total_blue_reward:.2f}")
    print("Final environment state:")
    print(prev_snapshot)


if __name__ == "__main__":
    main()
