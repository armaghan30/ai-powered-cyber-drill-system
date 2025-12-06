import numpy as np
import csv

from orchestrator.rl_env_red import RedRLEnvironment
from orchestrator.dqn_agent_red import DQNAgentRed


def flatten_red_state(state: dict, host_order: list[str]) -> np.ndarray:
    hosts = state["hosts"]
    features = []

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

    features.append(state["timestep"])
    features.append(state["num_hosts"])
    features.append(state["num_compromised"])

    return np.array(features, dtype=np.float32)


def main():

    topology_path = "orchestrator/sample_topology.yaml"

    max_steps_per_episode = 20
    num_episodes = 10

    agent_lr = 1e-3
    gamma = 0.99
    batch_size = 64
    buffer_capacity = 5000

    epsilon_start = 1.0
    epsilon_end = 0.05
    epsilon_decay_steps = 5000
    target_update_freq = 500

    # === CREATE ENVIRONMENT ===
    env = RedRLEnvironment(topology_path, max_steps=max_steps_per_episode)

    # === Gymnasium RESET ===
    _, _ = env.reset()

    # Get initial orchestrator state
    state_dict = env.orch.get_red_state()

    host_order = sorted(env.orch.environment.hosts.keys())
    example_vec = flatten_red_state(state_dict, host_order)

    state_dim = example_vec.shape[0]
    action_dim = env.num_red_actions

    print(f"[INFO] State dim : {state_dim}")
    print(f"[INFO] Actions   : {action_dim}")
    print(f"[INFO] Hosts     : {host_order}")

    # === CREATE AGENT ===
    agent = DQNAgentRed(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=agent_lr,
        gamma=gamma,
        batch_size=batch_size,
        buffer_capacity=buffer_capacity,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay=epsilon_decay_steps,
        target_update_freq=target_update_freq,
    )

    episode_rewards = []

    # === TRAINING LOOP ===
    for episode in range(1, num_episodes + 1):

        _, _ = env.reset()
        state_dict = env.orch.get_red_state()
        done = False
        total_reward = 0.0
        steps = 0

        print(f"\n[EPISODE {episode}/{num_episodes}] Starting...")

        while not done:
            steps += 1
            if steps > max_steps_per_episode:
                break

            state_vec = flatten_red_state(state_dict, host_order)

            action_id = agent.select_action(state_vec)

            next_obs, reward, terminated, truncated, info = env.step(action_id)
            done = terminated or truncated

            next_state_dict = env.orch.get_red_state()
            next_vec = flatten_red_state(next_state_dict, host_order)

            agent.store_transition(state_vec, action_id, reward, next_vec, done)
            agent.train_step()

            state_dict = next_state_dict
            total_reward += reward

        episode_rewards.append(total_reward)

        print(f"[EP {episode}] Reward={total_reward:.2f} | Epsilon={agent.epsilon:.3f}")

    agent.save("red_dqn_model.pth")
    print("\n[INFO] Model saved to red_dqn_model.pth")

    with open("red_rewards.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "reward"])
        for i, r in enumerate(episode_rewards, start=1):
            writer.writerow([i, r])

    print("[INFO] Rewards saved to red_rewards.csv")


if __name__ == "__main__":
    main()
