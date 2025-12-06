import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from tiny_agent.tiny_env import TinyCyberEnv
from tiny_agent.replay_buffer import ReplayBuffer
from tiny_agent.dqn import DQN     # <-- USE THE SHARED DQN MODEL


# ------------------------
# TRAIN FUNCTION
# ------------------------
def train_agent(episodes=5000):

    env = TinyCyberEnv()
    obs, _ = env.reset()

    input_dim = obs.shape[0]
    output_dim = env.action_space.n

    # USE SHARED MODEL
    policy_net = DQN(input_dim, output_dim)
    target_net = DQN(input_dim, output_dim)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    buffer = ReplayBuffer(max_size=50000)

    gamma = 0.99
    batch_size = 64
    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = 0.999

    print(f"Training started!")
    print(f"Observation size: {input_dim}, Action size: {output_dim}")

    for ep in range(1, episodes + 1):

        obs, _ = env.reset()
        total_reward = 0

        done = False
        truncated = False

        while not done and not truncated:

            # ------------------------
            # Epsilon-Greedy Action Select
            # ------------------------
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    q_vals = policy_net(torch.FloatTensor(obs))
                    action = torch.argmax(q_vals).item()

            next_obs, reward, done, truncated, _ = env.step(action)

            buffer.add(obs, action, reward, next_obs, done)

            obs = next_obs
            total_reward += reward

            # ------------------------
            # TRAINING BATCH
            # ------------------------
            if buffer.size() > batch_size:
                b_obs, b_actions, b_rewards, b_next, b_done = buffer.sample(batch_size)

                b_obs = torch.FloatTensor(b_obs)
                b_actions = torch.LongTensor(b_actions)
                b_rewards = torch.FloatTensor(b_rewards)
                b_next = torch.FloatTensor(b_next)
                b_done = torch.FloatTensor(b_done)

                q_values = policy_net(b_obs)
                q_value = q_values.gather(1, b_actions.unsqueeze(1)).squeeze(1)

                with torch.no_grad():
                    next_q_values = target_net(b_next).max(1)[0]
                    expected = b_rewards + gamma * next_q_values * (1 - b_done)

                loss = criterion(q_value, expected)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Update epsilon
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        # Update target network
        if ep % 50 == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if ep % 100 == 0:
            print(f"Episode {ep} | Reward: {total_reward:.2f} | Epsilon: {epsilon:.3f}")

    # SAVE SHARED MODEL
    torch.save(policy_net.state_dict(), "tiny_red_agent.pt")
    print("Training complete! Saved as tiny_red_agent.pt")


# ------------------------
# MAIN
# ------------------------
if __name__ == "__main__":
    train_agent(episodes=3000)
