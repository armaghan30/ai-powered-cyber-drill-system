
import json
import datetime
import copy

from orchestrator.yaml_loader import YAMLLoader
from orchestrator.env_builder import Environment
from orchestrator.agents.red_agent import RedAgent
from orchestrator.agents.blue_agent import BlueAgent
from orchestrator.reward_engine import RewardEngine   # NEW: external reward engine


class Orchestrator:
    _active_env = None
    _logs = []

    def __init__(self, topology_path: str):
        self.topology_path = topology_path
        self.topology = None
        self.environment = None
        self.red_agent = None
        self.blue_agent = None
        self.logs = []
        self.reward_engine = RewardEngine()   # <-- Uses external class

    # ------------------------------------------------------------
    # Load Topology, Build Environment
    # ------------------------------------------------------------
    def load_topology(self):
        print("Loading topology...")
        loader = YAMLLoader(self.topology_path)
        self.topology = loader.parse()
        print("Topology Loaded Successfully.")
        return self.topology

    def build_environment(self):
        if self.topology is None:
            raise ValueError("Topology not loaded.")
        print("Building environment from topology...")
        self.environment = Environment(self.topology)
        print("Environment Ready.")
        return self.environment

    def init_red_agent(self):
        self.red_agent = RedAgent(self.environment)
        print("Red Agent Ready.")
        return self.red_agent

    def init_blue_agent(self):
        self.blue_agent = BlueAgent(self.environment)
        print("Blue Agent Ready.")
        return self.blue_agent

    # ------------------------------------------------------------
    # Snapshot for reward engine
    # ------------------------------------------------------------
    def _snapshot_environment(self):
        """
        Capture current environment state as a JSON-dumpable snapshot.
        """
        hosts_state = {
            name: {
                "is_compromised": host.is_compromised,
                "access_level": host.access_level,
                "vulnerabilities": list(host.vulnerabilities),
            }
            for name, host in self.environment.hosts.items()
        }

        return {
            "step": self.environment.step_count,
            "timestamp": datetime.datetime.now().isoformat(),
            "hosts": hosts_state,
            "edges": copy.deepcopy(self.environment.edges),
        }
        # ------------------------------------------------------------
    # BLUE OBSERVATION VECTOR (used during evaluation)
    # ------------------------------------------------------------
    def get_blue_state_vector(self):
        """
        Produce the same Blue observation vector used during training.
        This keeps eval_both_agents consistent with BlueRLEnvironment.
        """
        from .rl_env_blue import flatten_blue_state

        snapshot = self._snapshot_environment()
        obs_vec = flatten_blue_state(snapshot)

        return obs_vec

    # ------------------------------------------------------------
    # Encoding helpers for RL states
    # ------------------------------------------------------------
    @staticmethod
    def _encode_access_level(level_str):
        return {"none": 0, "user": 1, "root": 2}.get(level_str, 0)

    @staticmethod
    def _encode_sensitivity(sens):
        return {"low": 0, "medium": 1, "high": 2}.get(sens, 0)

    # ------------------------------------------------------------
    # RED RL STATE (ALWAYS returns 13 features)
    # ------------------------------------------------------------
    def _build_red_state(self):
        env = self.environment
        timestep = env.step_count
        hosts = env.hosts
        knowledge = getattr(self.red_agent, "known_vulns", {})

        hosts_state = {}

        for name, host in hosts.items():
            scanned = name in knowledge

            if scanned:
                vulns = knowledge[name].get("vulnerabilities", [])
                services = knowledge[name].get("services", [])
            else:
                vulns = []
                services = []

            hosts_state[name] = {
                "scanned": int(scanned),
                "vulnerabilities": len(vulns),
                "services": len(services),
                "is_compromised": int(host.is_compromised),
                "access_level": self._encode_access_level(host.access_level),
            }

        red_state = {
            "timestep": timestep,
            "hosts": hosts_state,
            "num_hosts": len(hosts_state),
            "num_compromised": sum(h["is_compromised"] for h in hosts_state.values()),
        }

        return red_state

    # ------------------------------------------------------------
    # BLUE RL STATE
    # ------------------------------------------------------------
    def _build_blue_state(self):
        env = self.environment
        timestep = env.step_count
        hosts = env.hosts

        hosts_state = {}
        for name, host in hosts.items():
            hosts_state[name] = {
                "is_compromised": int(host.is_compromised),
                "vulnerabilities": len(host.vulnerabilities),
                "access_level": self._encode_access_level(host.access_level),
                "sensitivity": self._encode_sensitivity(host.sensitivity),
            }

        blue_state = {
            "timestep": timestep,
            "hosts": hosts_state,
            "num_compromised": sum(h["is_compromised"] for h in hosts_state.values()),
        }

        return blue_state

    def get_red_state(self):
        return self._build_red_state()

    def get_blue_state(self):
        return self._build_blue_state()

    # ------------------------------------------------------------
    # Simulation Step
    # ------------------------------------------------------------
    def run_single_step(self):

        red_state = self._build_red_state()
        blue_state = self._build_blue_state()

        if self.logs:
            prev_state = self.logs[-1]["environment"]
        else:
            prev_state = self._snapshot_environment()

        red_action = self.red_agent.choose_action()
        blue_action = self.blue_agent.choose_action(red_action)

        env_state = self.environment.step(red_action, blue_action)

        red_reward, blue_reward = self.reward_engine.compute_rewards(
            prev_state, env_state, red_action, blue_action
        )

        log_entry = {
            "step": len(self.logs) + 1,
            "timestamp": datetime.datetime.now().isoformat(),
            "red_state": red_state,
            "blue_state": blue_state,
            "red_action": red_action,
            "blue_action": blue_action,
            "red_reward": red_reward,
            "blue_reward": blue_reward,
            "environment": env_state,
        }

        self.logs.append(log_entry)
        return log_entry

    # ------------------------------------------------------------
    def run_simulation(self, max_steps=10):
        print(f"Starting simulation for {max_steps} steps...\n")
        for _ in range(max_steps):
            entry = self.run_single_step()
            print(
                f"[STEP {entry['step']}] "
                f"RED={entry['red_action']} (R={entry['red_reward']}) | "
                f"BLUE={entry['blue_action']} (R={entry['blue_reward']})"
            )
        print("Simulation complete.\n")
        return self.logs

    # ------------------------------------------------------------
    def save_logs(self, filepath="simulation_log.json"):
        with open(filepath, "w") as f:
            json.dump(self.logs, f, indent=4)
        print(f"Logs saved to {filepath}")

    # ------------------------------------------------------------
    # Dashboard API
    # ------------------------------------------------------------
    @classmethod
    def create_env(cls, topology_path: str):
        instance = cls(topology_path)
        instance.load_topology()
        instance.build_environment()
        instance.init_red_agent()
        instance.init_blue_agent()
        cls._active_env = instance
        cls._logs = instance.logs
        return instance

    @classmethod
    def step(cls):
        if cls._active_env is None:
            return {"status": "error", "message": "No active session"}

        entry = cls._active_env.run_single_step()
        cls._logs = cls._active_env.logs

        return {
            "status": "ok",
            "log": entry,
            "all_logs": cls._logs,
        }


if __name__ == "__main__":
    topology_path = "orchestrator/sample_topology.yaml"

    print("[MAIN] Starting orchestrator...")
    orch = Orchestrator(topology_path)
    orch.load_topology()
    orch.build_environment()
    orch.init_red_agent()
    orch.init_blue_agent()

    orch.run_simulation(max_steps=10)
    orch.save_logs("simulation_log.json")
