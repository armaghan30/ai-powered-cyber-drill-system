import matplotlib.pyplot as plt
import csv

# Read Red rewards
red_ep = []
red_rewards = []
with open("marl_rewards_red.csv", "r") as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        red_ep.append(int(row[0]))
        red_rewards.append(float(row[1]))

# Read Blue rewards
blue_ep = []
blue_rewards = []
with open("marl_rewards_blue.csv", "r") as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        blue_ep.append(int(row[0]))
        blue_rewards.append(float(row[1]))

plt.figure(figsize=(10, 5))
plt.plot(red_ep, red_rewards, label="Red Agent Reward", linewidth=2)
plt.plot(blue_ep, blue_rewards, label="Blue Agent Reward", linewidth=2)

plt.title("MARL Evaluation Reward Trend (Red vs Blue)")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig("marl_reward_plot.png", dpi=300)
plt.show()
