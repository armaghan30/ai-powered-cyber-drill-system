import random
import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim


# -----------------------------------------------------------
# Q-Network
# -----------------------------------------------------------
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_dim)
        )

    def forward(self, x):
        return self.net(x)


# -----------------------------------------------------------
# DQN Agent
# -----------------------------------------------------------
class DQNAgentRed:
    def __init__(
        self,
        state_dim,
        action_dim,
        lr=1e-3,
        gamma=0.99,
        batch_size=64,
        buffer_capacity=5000,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=5000,
        target_update_freq=500,
    ):
        print(f"[DQN INIT] state_dim={state_dim}, action_dim={action_dim}")

        self.state_dim = state_dim
        self.action_dim = action_dim

        # Replay Memory
        self.replay_buffer = deque(maxlen=buffer_capacity)
        self.batch_size = batch_size

        # DQN hyperparams
        self.gamma = gamma
        self.lr = lr
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        self.total_steps = 0

        # Networks
        self.policy_net = QNetwork(state_dim, action_dim)
        self.target_net = QNetwork(state_dim, action_dim)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

        self.loss_fn = nn.MSELoss()

    # -----------------------------------------------------------
    # Experience Replay Memory
    # -----------------------------------------------------------
    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    # -----------------------------------------------------------
    # Training-time Action Selection (epsilon-greedy)
    # -----------------------------------------------------------
    def select_action(self, state_vec):
        """Select action during training using epsilon-greedy."""
        self.total_steps += 1

        # Decay epsilon
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon - (1.0 / self.epsilon_decay)
        )

        # Random exploration
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)

        # Greedy exploitation
        state = torch.FloatTensor(state_vec).unsqueeze(0)
        q_values = self.policy_net(state)
        action = torch.argmax(q_values, dim=1).item()
        return action

    # -----------------------------------------------------------
    # Evaluation-time Action Selection (NO exploration)
    # -----------------------------------------------------------
    def select_action_eval(self, state_vec):
        state = torch.FloatTensor(state_vec).unsqueeze(0)
        q_values = self.policy_net(state)
        return torch.argmax(q_values, dim=1).item()

    # -----------------------------------------------------------
    # DQN Training Step
    # -----------------------------------------------------------
    def train_step(self):

        if len(self.replay_buffer) < self.batch_size:
            return  # need enough samples

        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        # Q(s, a)
        q_values = self.policy_net(states).gather(1, actions)

        # max_a' Q_target(s', a')
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(dim=1, keepdim=True)[0]

        # TD target
        targets = rewards + (1 - dones) * self.gamma * next_q_values

        loss = self.loss_fn(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        if self.total_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    # -----------------------------------------------------------
    # Save & Load models
    # -----------------------------------------------------------
    def save(self, filepath: str):
        checkpoint = {
        "online_state_dict": self.policy_net.state_dict(),
        "target_state_dict": self.target_net.state_dict(),
        "optimizer_state_dict": self.optimizer.state_dict(),
        "state_dim": self.state_dim,
        "action_dim": self.action_dim,
        "epsilon": self.epsilon,
    }
        torch.save(checkpoint, filepath)
        print(f"[DQN] Model saved -> {filepath}")


    def load(self, path):
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint["online_state_dict"])
        self.target_net.load_state_dict(checkpoint["target_state_dict"])
        self.epsilon = checkpoint.get("epsilon", 0.05)
        self.total_steps = checkpoint.get("total_steps", 0)
