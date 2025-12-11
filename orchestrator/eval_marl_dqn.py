# orchestrator/eval_marl_dqn.py

"""
Multi-Agent RL evaluation:
    - RED: trained DQN attacker
    - BLUE: trained DQN defender

Runs several episodes and saves per-episode rewards to:
    - marl_rewards_red.csv
    - marl_rewards_blue.csv

Usage:
    (cyber_env) python -m orchestrator.eval_marl_dqn
"""

from __future__ import annotations

import csv
import numpy as np
import torch

from .orchestrator_core import Orchestrator
from .dqn_agent_red import DQNAgentRed
from .rl_env_red import flatten_red_state


def flatten_blue_state(blue_state: dict, host_order: list[str]) -> np.ndarray:
    """
    Same encoding idea as in eval_both_agents:
        per host: [is_comp, vulnerabilities, access, sensitivity]
        globals : [timestep, num_compromised]
    """
    hosts = blue_state.get("hosts", {})
    features = []

    for name in host_order:
        h = hosts.get(name, {})
        is_comp = float(h.get("is_compromised", 0))
        vulns = float(h.get("vulnerabilities", 0))
        access = float(h.get("access_level", 0))
        sens = float(h.get("sensitivity", 0))
        features.extend([is_comp, vulns, access, sens])

    features.append(float(blue_state.get("timestep", 0)))
    features.append(float(blue_state.get("num_compromised", 0)))

    return np.array(features, dtype=np.float32)


def load_generic_dqn(model_path: str, label: str) -> DQNAgentRed:
    """
    Generic loader for RED or BLUE DQN models using DQNAgentRed as a wrapper.
    """
    ckpt = torch.load(model_path, map_location="cpu")
    state_dim = int(ckpt["state_dim"])
    action_dim = int(ckpt["action_dim"])

    print(f"[EVAL MARL] Loading {label} DQN: state_dim={state_dim}, action_dim={action_dim}")

    agent = DQNAgentRed(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=1e-3,
        gamma=0.99,
        batch_size=64,
        buffer_capacity=1,
        epsilon_start=0.0,
        epsilon_end=0.0,
        epsilon_decay=1,
        target_update_freq=1000,
    )

    agent.policy_net.load_state_dict(ckpt["online_state_dict"])
    agent.target_net.load_state_dict(ckpt["target_state_dict"])
    agent.epsilon = 0.0

    print(f"[EVAL MARL] {label} model loaded successfully.")
    return agent


def decode_red_action_id(action_id: int, host_order: list[str], orch: Orchestrator) -> dict:
    n = len(host_order)
    if action_id < n:
        host = host_order[action_id]
        return orch.red_agent.scan(host)
    else:
        host = host_order[action_id - n]
        return orch.red_agent.exploit(host)


def decode_blue_action_id(action_id: int, host_order: list[str]) -> dict:
    n = len(host_order)

    if action_id < n:
        return {"action": "patch", "target": host_order[action_id]}
    elif action_id < 2 * n:
        return {"action": "isolate", "target": host_order[action_id - n]}
    else:
        return {"action": "idle", "target": None}


def main():
    topology_path = "orchestrator/sample_topology.yaml"
    max_steps = 20
    num_episodes = 1  # you can increase this later if you want more evaluation episodes

    red_agent = load_generic_dqn("red_dqn_model.pth", "RED")
    blue_agent = load_generic_dqn("blue_dqn_model.pth", "BLUE")

    red_rewards = []
    blue_rewards = []

    for ep in range(1, num_episodes + 1):
        orch = Orchestrator(topology_path)
        orch.load_topology()
        orch.build_environment()
        orch.init_red_agent()
        orch.init_blue_agent()

        host_order = list(orch.environment.hosts.keys())

        total_red_reward = 0.0
        total_blue_reward = 0.0

        print("\n=========== MARL EVALUATION EPISODE START ===========\n")

        for step in range(1, max_steps + 1):
            prev_env_state = orch._snapshot_environment()

            # RED side
            red_state = orch.get_red_state()
            red_vec = flatten_red_state(red_state, host_order)
            red_action_id = red_agent.select_action_eval(red_vec)
            red_action = decode_red_action_id(red_action_id, host_order, orch)

            # BLUE side
            blue_state = orch.get_blue_state()
            blue_vec = flatten_blue_state(blue_state, host_order)
            blue_action_id = blue_agent.select_action_eval(blue_vec)
            blue_action = decode_blue_action_id(blue_action_id, host_order)

            # Apply actions
            env_state = orch.environment.step(red_action, blue_action)

            red_r, blue_r = orch.reward_engine.compute_rewards(
                prev_env_state, env_state, red_action, blue_action
            )

            total_red_reward += red_r
            total_blue_reward += blue_r

            print(f"RED action : {red_action}")
            print(f"BLUE action: {blue_action}")
            print(f"Red Reward = {red_r:.2f} | Blue Reward = {blue_r:.2f}\n")

        print("=========== MARL EVALUATION COMPLETE ===========")
        print(f"Total RED Reward  = {total_red_reward:.2f}")
        print(f"Total BLUE Reward = {total_blue_reward:.2f}")
        print("Final environment state:")
        print(orch._snapshot_environment())

        red_rewards.append(total_red_reward)
        blue_rewards.append(total_blue_reward)

    # -------- Save per-episode rewards to CSV --------
    with open("marl_rewards_red.csv", "w", newline="") as f_red, open(
        "marl_rewards_blue.csv", "w", newline=""
    ) as f_blue:
        red_writer = csv.writer(f_red)
        blue_writer = csv.writer(f_blue)

        red_writer.writerow(["episode", "reward"])
        blue_writer.writerow(["episode", "reward"])

        for i, (rr, br) in enumerate(zip(red_rewards, blue_rewards), start=1):
            red_writer.writerow([i, rr])
            blue_writer.writerow([i, br])

    print("\n[MARL] Episode rewards saved to marl_rewards_red.csv and marl_rewards_blue.csv")


if __name__ == "__main__":
    main()
