

"""
Training script for Blue DQN Agent using BlueRLEnvironment (Gymnasium).
"""

import csv

from orchestrator.rl_env_blue import BlueRLEnvironment
from orchestrator.dqn_agent_red import DQNAgentRed      

def main():
    topology_path = "orchestrator/sample_topology.yaml"
    
    #------------------Hyperparameters---------------------
    max_steps_per_episode = 20
    num_episodes = 10
    gamma = 0.99
    lr = 1e-3
    batch_size = 64
    buffer_capacity = 5000
    epsilon_start = 1.0
    epsilon_end = 0.05
    epsilon_decay_steps = 5000
    target_update_freq = 500
    
    #-------------------------Environment------------------------
    
    env = BlueRLEnvironment(topology_path, max_steps=max_steps_per_episode)
    
    state_vec, info = env.reset()
    state_dim = state_vec.shape[0]
    
    if env.num_blue_actions is None:
        raise RuntimeError("Blue Action space not initialised. Check BlueRLEnvironment._build_action_space().")
    
    action_dim = env.num_blue_actions
    
    print(f"[BLUE INFO] State dim    : {state_dim}")
    print(f"[BLUE INFO] Actions      : {action_dim}")
    
    #----------------------DQN Agent (used for BLUE)----------------------
    agent = DQNAgentRed(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=lr,
        gamma=gamma,
        batch_size=batch_size,
        buffer_capacity=buffer_capacity,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay=epsilon_decay_steps,
        target_update_freq=target_update_freq,
    )
    
    episode_rewards = []
    
    #-------------------------------Training Loop-----------------------------
    for episode in range(1, num_episodes + 1):
        state_vec, info = env.reset()
        total_reward = 0.0
        done = False
        step_counter = 0
        
        print(f"\n[BLUE EPISODE {episode}/{num_episodes}] Starting.....")
        
        while not done:
            step_counter += 1
            if step_counter > max_steps_per_episode:
                print(f"[BLUE EPISODE {episode}] Safety break at {step_counter} steps.")
                break
            
            # Epsilon greedy action selection
            action_id = agent.select_action(state_vec)
            
            # Environment step
            next_vec, reward, terminated, truncated, info = env.step(action_id)
            done = terminated or truncated
            
            # Storing the transition and train
            agent.store_transition(state_vec, action_id, reward, next_vec, done)
            agent.train_step()
            
            state_vec = next_vec
            total_reward += reward
            
        episode_rewards.append(total_reward)
        
        print(
            f"[BLUE EPISODE {episode}/{num_episodes}]"
            f"Finished in {step_counter} steps | "
            f"Total Reward = {total_reward:.2f} | "
            f"Epsilon = {agent.epsilon:.3f} "
        )
        
        #------------------------Save trained Model-----------------------------
        save_path = "blue_dqn_model.pth"
        agent.save(save_path)
        print(f"\n[BLUE INFO] Training finished. Model saved to {save_path}")
        
        #------------------------Save rewards to CSV---------------------------------
        with open("blue_reward.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["episode", "reward"])
            for i, r in enumerate(episode_rewards, start = 1):
                writer.writerow([i, r])
                
        print("[BLUE INFO] Episode rewards saved to blue_rewards.csv")
        
if __name__ == "__main__":
    main()
    