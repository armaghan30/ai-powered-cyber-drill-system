import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
from orchestrator.q_network import QNetwork


class DQNAgentBase:
    """
    Unified base class → both Red & Blue inherit from this.
    """
    def __init__(
        self,
        state_dim,
        action_dim,
        lr=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=5000,
        buffer_capacity=50000,
        batch_size=64,
        target_update_freq=500,
    ):

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # Epsilon-greedy exploration
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.steps_done = 0

        # Replay buffer
        self.memory = deque(maxlen=buffer_capacity)

        # Networks
        self.online = QNetwork(state_dim, action_dim)
        self.target = QNetwork(state_dim, action_dim)
        self.target.load_state_dict(self.online.state_dict())

        self.optimizer = optim.Adam(self.online.parameters(), lr=lr)

    # -----------------------------------------------
    def act(self, state):
        """Epsilon-greedy action selection."""
        self.steps_done += 1
        import numpy as np

        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)

        state_t = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.online(state_t)
        return int(torch.argmax(q_values, dim=1)[0])

    # -----------------------------------------------
    def remember(self, s, a, r, s2, done):
        self.memory.append((s, a, r, s2, done))

    # -----------------------------------------------
    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        s, a, r, s2, done = zip(*batch)

        s = torch.FloatTensor(s)
        s2 = torch.FloatTensor(s2)
        r = torch.FloatTensor(r)
        done = torch.FloatTensor(done)
        a = torch.LongTensor(a)

        q_pred = self.online(s).gather(1, a.unsqueeze(1)).squeeze()

        with torch.no_grad():
            q_next = self.target(s2).max(1)[0]
            q_target = r + self.gamma * q_next * (1 - done)

        loss = nn.MSELoss()(q_pred, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # UPDATE TARGET NETWORK
        if self.steps_done % self.target_update_freq == 0:
            self.target.load_state_dict(self.online.state_dict())

        # EPSILON DECAY
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon_start * (0.99995 ** self.steps_done)
        )

    # -----------------------------------------------
    def save(self, path):
        torch.save(
            {
                "state_dim": self.state_dim,
                "action_dim": self.action_dim,
                "online": self.online.state_dict(),
                "target": self.target.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "steps_done": self.steps_done,
            },
            path,
        )

    # -----------------------------------------------
    @classmethod
    def load(cls, path):
        ckpt = torch.load(path, map_location="cpu")

        agent = cls(
            ckpt["state_dim"],
            ckpt["action_dim"]
        )

        agent.online.load_state_dict(ckpt["online"])
        agent.target.load_state_dict(ckpt["target"])
        agent.optimizer.load_state_dict(ckpt["optimizer"])
        agent.epsilon = ckpt["epsilon"]
        agent.steps_done = ckpt["steps_done"]

        return agent
