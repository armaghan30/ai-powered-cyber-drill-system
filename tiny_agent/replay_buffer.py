import random
import numpy as np

class ReplayBuffer:
    def __init__(self, max_size=50000):
        self.max_size = max_size
        self.memory = []
        self.position = 0

    def add(self, state, action, reward, next_state, done):
        data = (state, action, reward, next_state, done)

        if len(self.memory) < self.max_size:
            self.memory.append(data)
        else:
            self.memory[self.position] = data

        self.position = (self.position + 1) % self.max_size

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
        )

    def size(self):
        return len(self.memory)
