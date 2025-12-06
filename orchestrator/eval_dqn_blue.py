# orchestrator/eval_dqn_blue.py

import numpy as np

from .rl_env_blue import BlueRLEnvironment
from .dqn_agent_red import DQNAgentRed


def main():
    print("\n========== BLUE DQN EVALUATION ==========\n")

    topology_path = "orchestrator/sample_topology.yaml"

    env = BlueRLEnvironment(topology_path, max_steps=20)

    # Load initial Blue observation
    obs_vec, info = env.reset()
    state_dim = obs_vec.shape[0]
    action_dim = env.num_blue_actions

    # Load trained Blue model
    agent = DQNAgentRed(state_dim=state_dim, action_dim=action_dim)
    agent.load("blue_dqn_model.pth")

    print("[INFO] Loaded Blue DQN model.")
    print(f"[INFO] Observation dim : {state_dim}")
    print(f"[INFO] Action dim       : {action_dim}")

    done = False
    step = 0
    total_reward = 0.0

    print("\n===== STARTING BLUE EVALUATION =====\n")

    while not done:
        step += 1
        print(f"--- STEP {step} ---")

        # Choose action deterministically
        action_id = agent.select_action_eval(obs_vec)
        next_obs, reward, terminated, truncated, info = env.step(action_id)

        done = terminated or truncated
        total_reward += reward
        obs_vec = next_obs

        # Pretty result
        print(f"RED ACTION : {info['red_action']}")
        print(f"BLUE ACTION: {info['blue_action']}")
        print(f"REWARD     : {reward}")
        print()

    print("\n===== BLUE EVALUATION COMPLETE =====")
    print(f"Total Reward: {total_reward:.2f}")


if __name__ == "__main__":
    main()
