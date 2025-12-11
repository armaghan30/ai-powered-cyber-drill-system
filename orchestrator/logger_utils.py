import csv
import matplotlib.pyplot as plt


def save_rewards(path, rewards):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "reward"])
        for i, r in enumerate(rewards, 1):
            writer.writerow([i, r])
    print(f"[LOG] Rewards saved -> {path}")


def plot_rewards(csv_path, out_img="reward_plot.png"):
    episodes = []
    rewards = []

    with open(csv_path, "r") as f:
        next(f)  # skip header
        for line in f:
            ep, r = line.strip().split(",")
            episodes.append(int(ep))
            rewards.append(float(r))

    plt.figure(figsize=(10, 5))
    plt.plot(episodes, rewards)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Training Reward Curve")
    plt.grid()
    plt.savefig(out_img)
    print(f"[PLOT] Saved to {out_img}")
