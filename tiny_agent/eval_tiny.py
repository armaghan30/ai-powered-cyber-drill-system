import torch
import numpy as np
from tiny_agent.dqn import DQN
from tiny_agent.tiny_env import TinyCyberEnv


def evaluate_agent(episodes=1):

    env = TinyCyberEnv()

    obs, _ = env.reset()
    input_dim = obs.shape[0]
    output_dim = env.action_space.n

    # Build model
    model = DQN(input_dim, output_dim)
    state_dict = torch.load("tiny_red_agent.pt", map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    print("\n=== Loaded Tiny Red Agent ===")
    print(f"Observation size: {input_dim}, Action size: {output_dim}")
    print("=============================\n")

    for ep in range(1, episodes + 1):
        obs, _ = env.reset()
        done = False
        truncated = False
        total_reward = 0
        step = 0

        print(f"\n===== Starting Episode {ep} =====\n")

        while not (done or truncated):
            step += 1

            # choose best action
            obs_t = torch.FloatTensor(obs)
            with torch.no_grad():
                qvals = model(obs_t)
            action = torch.argmax(qvals).item()

            action_obj = env.actions[action]

            next_obs, reward, done, truncated, _ = env.step(action)
            total_reward += reward

            print(f"--- Step {step} ---")
            print(f"Action Index : {action}")
            print(f"Action Object: {action_obj}")
            print(f"Reward       : {reward}")
            print(f"Done? {done} | Truncated? {truncated}\n")

            obs = next_obs

        print(f"=== Episode {ep} finished! Total Reward: {total_reward} ===")


if __name__ == "__main__":
    evaluate_agent(episodes=1)
