from orchestrator.rl_env_red import RedRLEnvironment
import gymnasium as gym

print("=== Testing RedRLEnvironment (Gymnasium) ===")

env = RedRLEnvironment("orchestrator/sample_topology.yaml", max_steps=5)

# Gymnasium-style reset
obs, info = env.reset()

print("Initial obs shape:", obs.shape)
print("Initial obs:", obs)

done = False
step = 0

while not done:
    step += 1

    # Sample random action from Gymnasium Discrete space
    action = env.action_space.sample()

    print(f"\n-- Step {step} --")
    print("Action_id:", action)

    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    print("Obs shape:", obs.shape)
    print("Reward:", reward)
    print("Terminated:", terminated)
    print("Truncated:", truncated)

print("\n=== RedRLEnvironment test finished ===")

env.close()
