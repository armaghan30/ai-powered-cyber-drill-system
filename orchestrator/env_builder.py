# orchestrator/env_builder.py

import copy
import datetime


class Host:
    def __init__(self, name, os, services, vulnerabilities, sensitivity):
        self.name = name
        self.os = os
        self.services = list(services)
        self.vulnerabilities = list(vulnerabilities)  # list of CVE strings
        self.sensitivity = sensitivity

        # runtime values
        self.is_compromised = False
        self.access_level = "none"   # "none" -> "user" -> "root"
        self.is_isolated = False     # for Blue isolation actions

    def __repr__(self):
        return (
            f"<Host {self.name}, compromised={self.is_compromised}, "
            f"vulns={len(self.vulnerabilities)}, isolated={self.is_isolated}>"
        )

    def to_state_dict(self):
        """
        Snapshot of this host used in environment state and RL encodings.
        """
        return {
            "is_compromised": self.is_compromised,
            "access_level": self.access_level,
            "vulnerabilities": list(self.vulnerabilities),
            "is_isolated": self.is_isolated,
        }


class Environment:
    """
    Core environment built from YAML topology.

    This object is used by:
      - Orchestrator (for logging + reward engine)
      - RL wrappers (rl_env_red / rl_env_blue)
      - RedAgent / BlueAgent

    It applies the EFFECTS of red_action + blue_action on hosts and edges,
    keeps a step counter, and returns a clean snapshot dict.
    """

    def __init__(self, topology_data):
        self.topology_data = topology_data
        self.hosts = {}
        self.edges = []
        self.step_count = 0

        self._build()

    # -------------------------------
    # Build environment from YAML
    # -------------------------------
    def _build(self):
        """
        Expecting topology_data in the same format as before:

        {
          "network": {
            "hosts": {
              "H1": {
                "os": ...,
                "services": [...],
                "vulnerabilities": [...],
                "sensitivity": "low"/"medium"/"high"
              },
              ...
            },
            "edges": [
              ["H1", "H2"],
              ...
            ]
          }
        }
        """
        network = self.topology_data["network"]

        for host_name, details in network["hosts"].items():
            host_obj = Host(
                name=host_name,
                os=details.get("os", "unknown"),
                services=details.get("services", []),
                vulnerabilities=details.get("vulnerabilities", []),
                sensitivity=details.get("sensitivity", "low"),
            )
            self.hosts[host_name] = host_obj

        self.edges = list(network.get("edges", []))

    # -------------------------------
    # Reset
    # -------------------------------
    def reset(self):
        """
        Rebuilds environment to initial state.
        """
        self.hosts = {}
        self.edges = []
        self.step_count = 0
        self._build()
        return self._snapshot()

    # -------------------------------
    # Apply Red + Blue actions
    # -------------------------------
    def step(self, red_action, blue_action):
        """
        Apply environment-level effects of Red and Blue actions.

        red_action example (from RedAgent):
          {"action": "scan", "target": "H1", ...}
          {"action": "exploit", "success": True/False, "target": "H1", ...}

        blue_action example (from BlueAgent or RL Blue):
          {"action": "patch", "target": "H1"}
          {"action": "isolate", "target": "H1"}
          {"action": "idle", "target": None}
        """

        # 1. Apply Red Action Effects (exploit success)
        if red_action is not None and red_action.get("action") == "exploit":
            target = red_action.get("target")
            success = red_action.get("success", False)

            if target in self.hosts and success:
                host = self.hosts[target]
                # Mark as compromised + root
                host.is_compromised = True
                host.access_level = "root"

                # OPTIONAL: consume one vulnerability on success
                if host.vulnerabilities:
                    host.vulnerabilities = host.vulnerabilities[1:]

        # 2. Apply Blue Action Effects
        if blue_action is not None:
            b_type = blue_action.get("action")
            target = blue_action.get("target")

            if b_type == "patch" and target in self.hosts:
                host = self.hosts[target]
                # PATCH ONLY ONE VULNERABILITY AT A TIME
                if host.vulnerabilities:
                    host.vulnerabilities = host.vulnerabilities[1:]

            elif b_type == "isolate" and target in self.hosts:
                host = self.hosts[target]
                host.is_isolated = True

                # Remove edges containing this host
                new_edges = []
                for e in self.edges:
                    if target not in e:
                        new_edges.append(e)
                self.edges = new_edges

            # "idle" -> do nothing

        # 3. Increase step counter
        self.step_count += 1

        # 4. Return environment snapshot
        return self._snapshot()

    # -------------------------------
    # Snapshot (used by Orchestrator, RL, RewardEngine)
    # -------------------------------
    def _snapshot(self):
        return {
            "step": self.step_count,
            "timestamp": datetime.datetime.now().isoformat(),
            "hosts": {
                name: host.to_state_dict()
                for name, host in self.hosts.items()
            },
            "edges": copy.deepcopy(self.edges),
        }
