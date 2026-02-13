#Train the BLUE DQN agent using the BlueRLEnvironment (Gymnasium-style)

from __future__ import annotations

import csv

from orchestrator.rl_env_blue import BlueRLEnvironment
from orchestrator.dqn_agent_blue import DQNAgentBlue


def main():
    topology_path = "orchestrator/sample_topology.yaml"

    num_episodes = 500
    max_steps_per_episode = 20

    gamma = 0.99
    lr = 1e-3
    batch_size = 64
    buffer_capacity = 10_000
    epsilon_start = 1.0
    epsilon_end = 0.05
    epsilon_decay = 5_000
    target_update_freq = 500

    env = BlueRLEnvironment(topology_path, max_steps=max_steps_per_episode)
    state_vec, _ = env.reset()
    state_dim = state_vec.shape[0]
    action_dim = env.action_dim

    print(f"[BLUE TRAIN] state_dim={state_dim}, action_dim={action_dim}")

    agent = DQNAgentBlue(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=lr,
        gamma=gamma,
        batch_size=batch_size,
        buffer_capacity=buffer_capacity,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay=epsilon_decay,
        target_update_freq=target_update_freq,
    )

    episode_rewards = []

    for ep in range(1, num_episodes + 1):
        state, _ = env.reset()
        total_reward = 0.0

        print(f"[BLUE EPISODE {ep}/{num_episodes}] Starting...")

        for t in range(1, max_steps_per_episode + 1):
            action = agent.act(state)
            next_state, reward, terminated, truncated, info = env.step(action)

            done = terminated or truncated
            agent.store_transition(state, action, reward, next_state, done)
            agent.update()

            state = next_state
            total_reward += reward

            if done:
                break

        episode_rewards.append(total_reward)
        print(
            f"[BLUE EPISODE {ep}/{num_episodes}] "
            f"Finished in {t} steps | Total Reward = {total_reward:.2f} | Epsilon = {agent.epsilon:.3f}"
        )

    agent.save("blue_dqn_model.pth")

    with open("blue_rewards.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "reward"])
        for i, r in enumerate(episode_rewards, start=1):
            writer.writerow([i, r])

    print("[BLUE INFO] Training finished. Model saved to blue_dqn_model.pth")
    print("[BLUE INFO] Episode rewards saved to blue_rewards.csv")


if __name__ == "__main__":
    main()
