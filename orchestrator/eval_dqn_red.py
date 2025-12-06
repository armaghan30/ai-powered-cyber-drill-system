# eval_dqn_red.py

import torch
from orchestrator.rl_env_red import RedRLEnvironment
from orchestrator.dqn_agent_red import DQNAgentRed


MODEL_PATH = "red_dqn_model.pth"
TOPOLOGY = "orchestrator/sample_topology.yaml"

EPISODES = 5     # number of evaluation episodes
MAX_STEPS = 20   # steps per episode


def evaluate_policy():
    print("Loading topology...")
    env = RedRLEnvironment(TOPOLOGY, max_steps=MAX_STEPS)
    initial_state = env.reset()

    # Compute state_dim from first state
    host_features = []
    for h in sorted(initial_state["hosts"].keys()):
        host_features.extend([
            0, 0, 0, 0, 0
        ])
    state_dim = len(host_features)

    num_actions = env.num_red_actions

    # Load agent
    agent = DQNAgentRed(state_dim, num_actions)

    print("Loading DQN checkpoint...")
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")

    # Load ONLY the online network
    agent.policy_net.load_state_dict(checkpoint["online_state_dict"])
    agent.epsilon = 0.0  # disable exploration

    print("Model loaded. Starting evaluation...\n")

    for ep in range(1, EPISODES + 1):
        state = env.reset()
        total_reward = 0
        exploit_success = 0
        exploit_attempts = 0

        for step in range(MAX_STEPS):
            action = agent.select_action(state)

            next_state, reward, done, info = env.step(action)
            total_reward += reward

            # count exploit successes
            if info["red_action"]["action"] == "exploit":
                exploit_attempts += 1
                if info["red_action"].get("success", False):
                    exploit_success += 1

            state = next_state
            if done:
                break

        success_rate = (
            exploit_success / exploit_attempts * 100
            if exploit_attempts > 0
            else 0
        )

        print(f"[EPISODE {ep}] Reward={total_reward:.2f}, "
              f"Exploit Success={success_rate:.1f}% "
              f"({exploit_success}/{exploit_attempts})")

    print("\nEvaluation complete.")


if __name__ == "__main__":
    evaluate_policy()
