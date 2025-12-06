import matplotlib.pyplot as plt
import csv

def load_csv(filename):
    episodes = []
    rewards = []
    with open(filename, "r") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            episodes.append(int(row[0]))
            rewards.append(float(row[1]))
    return episodes, rewards

# --- RED ---
red_ep, red_rw = load_csv("red_rewards.csv")
plt.figure(figsize=(10, 4))
plt.plot(red_ep, red_rw, label="Red Training Reward", linewidth=2)
plt.title("Red Agent Training Reward per Episode")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.grid(True)
plt.tight_layout()
plt.savefig("red_training_plot.png", dpi=300)

# --- BLUE ---
blue_ep, blue_rw = load_csv("blue_rewards.csv")
plt.figure(figsize=(10, 4))
plt.plot(blue_ep, blue_rw, label="Blue Training Reward", linewidth=2)
plt.title("Blue Agent Training Reward per Episode")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.grid(True)
plt.tight_layout()
plt.savefig("blue_training_plot.png", dpi=300)

plt.show()
