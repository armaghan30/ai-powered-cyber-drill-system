# orchestrator/eval_marl_dqn.py

"""
Evaluate trained MARL Red & Blue DQN agents together on MultiAgentEnv.
"""

from orchestrator.multi_agent_env import MultiAgentEnv
from orchestrator.dqn_agent_red import DQNAgentRed


def main():
    topology_path = "orchestrator/sample_topology.yaml"
    max_steps = 20

    env = MultiAgentEnv(topology_path, max_steps=max_steps)

    # Build agents with correct dimensions
    obs, _ = env.reset()
    red_state_dim = obs["red"].shape[0]
    blue_state_dim = obs["blue"].shape[0]
    red_action_dim = env.red_action_dim
    blue_action_dim = env.blue_action_dim

    red_agent = DQNAgentRed(
        state_dim=red_state_dim,
        action_dim=red_action_dim,
    )
    blue_agent = DQNAgentRed(
        state_dim=blue_state_dim,
        action_dim=blue_action_dim,
    )

    print("[EVAL MARL] Loading trained models...")
    red_agent.load("marl_red_dqn.pth")
    blue_agent.load("marl_blue_dqn.pth")

    # NO exploration during eval
    red_agent.epsilon = 0.0
    blue_agent.epsilon = 0.0

    obs, _ = env.reset()
    red_state = obs["red"]
    blue_state = obs["blue"]

    total_red_reward = 0.0
    total_blue_reward = 0.0

    print("\n=========== MARL EVALUATION EPISODE START ===========\n")

    done = False
    step = 0
    while not done and step < max_steps:
        step += 1

        red_action_id = red_agent.select_action_eval(red_state)
        blue_action_id = blue_agent.select_action_eval(blue_state)

        next_obs, rewards, terminated, truncated, info = env.step(
            {"red": red_action_id, "blue": blue_action_id}
        )
        done = terminated or truncated

        red_state = next_obs["red"]
        blue_state = next_obs["blue"]

        r_r = rewards["red"]
        r_b = rewards["blue"]

        total_red_reward += r_r
        total_blue_reward += r_b

        print(f"--- Step {step} ---")
        print(f"RED action : {info['red_action']}")
        print(f"BLUE action: {info['blue_action']}")
        print(f"Red Reward = {r_r:.2f} | Blue Reward = {r_b:.2f}\n")

    print("=========== MARL EVALUATION COMPLETE ===========")
    print(f"Total RED Reward  = {total_red_reward:.2f}")
    print(f"Total BLUE Reward = {total_blue_reward:.2f}")
    print("Final environment state:")
    print(info["state"])


if __name__ == "__main__":
    main()
