import yaml


class ScenarioBuilder:
    def __init__(self):
        self.scenario = {"hosts": {}}

    def add_host(self, name, vulnerabilities=None, is_sensitive=False):
        self.scenario["hosts"][name] = {
            "vulnerabilities": vulnerabilities or [],
            "sensitive": is_sensitive
        }

    def set_connections(self, connections):
        """
        connections: list of tuples → [("H1","H2"), ("H2","H3")]
        """
        self.scenario["connections"] = connections

    def build(self, path):
        with open(path, "w") as f:
            yaml.dump(self.scenario, f)
        print(f"[SCENARIO] Saved -> {path}")
