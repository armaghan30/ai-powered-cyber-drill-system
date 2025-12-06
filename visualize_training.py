import csv
import matplotlib.pyplot as plt


def load_rewards(path):
    episodes = []
    rewards = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            episodes.append(int(row["episode"]))
            rewards.append(float(row["reward"]))
    return episodes, rewards


def main():
    # Red rewards
    try:
        red_eps, red_rew = load_rewards("red_rewards.csv")
    except FileNotFoundError:
        red_eps, red_rew = [], []
        print("[WARN] red_rewards.csv not found")

    # Blue rewards
    try:
        blue_eps, blue_rew = load_rewards("blue_rewards.csv")
    except FileNotFoundError:
        blue_eps, blue_rew = [], []
        print("[WARN] blue_rewards.csv not found")

    if red_eps:
        plt.plot(red_eps, red_rew, label="Red Reward")
    if blue_eps:
        plt.plot(blue_eps, blue_rew, label="Blue Reward")

    if not red_eps and not blue_eps:
        print("No reward files found to plot.")
        return

    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Rewards (Red & Blue)")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
