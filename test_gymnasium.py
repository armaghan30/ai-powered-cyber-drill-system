import gymnasium as gym

# Simple test to check Gymnasium installation and basic usage
env = gym.make("CartPole-v1")

obs, info = env.reset()

print("Gymnasium is working!")
print("Initial observation:", obs)

env.close()
