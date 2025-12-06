import gym
from gym import spaces
import numpy as np
import random


class TinyCyberEnv(gym.Env):
    """
    Tiny multi-hop cyber range for Red Team RL.

    Hosts: H1 -> H2 -> H3 -> H4 (crown jewel)
    Must get root on parent before pivoting to child.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self):
        super().__init__()

        self.hosts = ["H1", "H2", "H3", "H4"]
        self.num_hosts = len(self.hosts)

        self.parent_map = {
            "H1": None,
            "H2": "H1",
            "H3": "H2",
            "H4": "H3",
        }

        # Build actions
        self.actions = self._build_actions()
        self.action_list = self.actions  # <-- important alias for eval
        self.action_space = spaces.Discrete(len(self.actions))

        # Obs space: reachable, scanned, user, root for each host
        self.features_per_host = 4
        obs_dim = self.num_hosts * self.features_per_host

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

        self.max_steps = 80

        self.state = None
        self.steps = 0

        self.reset()

    # ------------------------------------------------------------------
    # ACTION SET
    # ------------------------------------------------------------------
    def _build_actions(self):
        actions = []

        # 0–3 scan
        for h in self.hosts:
            actions.append({"type": "scan", "target": h})

        # 4–7 exploit
        for h in self.hosts:
            actions.append({"type": "exploit", "target": h})

        # 8–11 privesc
        for h in self.hosts:
            actions.append({"type": "privesc", "target": h})

        # 12–15 pivot
        for h in self.hosts:
            actions.append({"type": "pivot", "target": h})

        # 16 flag
        actions.append({"type": "flag", "target": "H4"})

        # 17 noop
        actions.append({"type": "noop", "target": None})

        return actions

    # ------------------------------------------------------------------
    # STATE MANAGEMENT
    # ------------------------------------------------------------------
    def _empty_state(self):
        state = {}
        for h in self.hosts:
            state[h] = {
                "reachable": False,
                "scanned": False,
                "user": False,
                "root": False,
            }
        return state

    def _encode_obs(self):
        vec = []
        for h in self.hosts:
            st = self.state[h]
            vec.extend([
                float(st["reachable"]),
                float(st["scanned"]),
                float(st["user"]),
                float(st["root"]),
            ])
        return np.array(vec, dtype=np.float32)

    # ------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        self.state = self._empty_state()
        self.steps = 0

        # Only H1 reachable
        self.state["H1"]["reachable"] = True

        return self._encode_obs(), {}

    # ------------------------------------------------------------------
    def step(self, action_idx):
        self.steps += 1
        done = False
        truncated = False
        reward = -0.1  # time penalty

        action = self.actions[action_idx]
        a_type = action["type"]
        target = action["target"]

        if a_type == "noop":
            reward -= 0.5

        elif a_type == "scan":
            reward += self._scan(target)

        elif a_type == "exploit":
            reward += self._exploit(target)

        elif a_type == "privesc":
            reward += self._privesc(target)

        elif a_type == "pivot":
            reward += self._pivot(target)

        elif a_type == "flag":
            r, done = self._flag(target)
            reward += r

        # episode end conditions
        if self.steps >= self.max_steps and not done:
            truncated = True

        obs = self._encode_obs()
        return obs, reward, done, truncated, {}

    # ------------------------------------------------------------------
    # ACTION LOGIC
    # ------------------------------------------------------------------
    def _scan(self, hname):
        h = self.state[hname]

        if not h["reachable"]:
            return -1.0

        if h["scanned"]:
            return -0.2

        h["scanned"] = True
        return 1.5

    def _exploit(self, hname):
        h = self.state[hname]

        if not h["reachable"]:
            return -2.0

        if not h["scanned"]:
            return -1.5

        if h["user"] or h["root"]:
            return -0.5

        if random.random() < 0.8:
            h["user"] = True
            return 10.0
        return -3.0

    def _privesc(self, hname):
        h = self.state[hname]

        if not (h["reachable"] and h["user"]):
            return -2.0

        if h["root"]:
            return -0.5

        if random.random() < 0.7:
            h["root"] = True
            return 15.0
        return -4.0

    def _pivot(self, hname):
        if hname == "H1":
            return -0.5

        parent = self.parent_map[hname]
        if parent is None:
            return -1.0

        if not self.state[parent]["root"]:
            return -2.0

        h = self.state[hname]

        if h["reachable"]:
            return -0.2

        # Activate this pivot hop
        h["reachable"] = True
        h["scanned"] = True  # auto basic recon

        return 5.0

    def _flag(self, hname):
        if hname != "H4":
            return -2.0, False

        if self.state["H4"]["root"]:
            return 200.0, True
        else:
            return -5.0, False

    # ------------------------------------------------------------------
    def render(self, mode="human"):
        print(f"Step={self.steps}")
        for h in self.hosts:
            s = self.state[h]
            print(h, s)
