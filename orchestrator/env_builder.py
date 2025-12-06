# env_builder.py

import copy
import datetime


class Host:
    def __init__(self, name, os, services, vulnerabilities, sensitivity):
        self.name = name
        self.os = os
        self.services = services
        self.vulnerabilities = vulnerabilities  # list of CVE strings
        self.sensitivity = sensitivity

        # runtime values
        self.is_compromised = False
        self.access_level = "none"

        # NEW: whether Blue has isolated this host from the network
        self.is_isolated = False

    def __repr__(self):
        return (
            f"<Host {self.name}, compromised={self.is_compromised}, "
            f"isolated={self.is_isolated}, vulns={len(self.vulnerabilities)}>"
        )


class Environment:

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
        network = self.topology_data["network"]

        for host_name, details in network["hosts"].items():
            host_obj = Host(
                name=host_name,
                os=details.get("os", "unknown"),
                services=details.get("services", []),
                vulnerabilities=list(details.get("vulnerabilities", [])),
                sensitivity=details.get("sensitivity", "low"),
            )
            self.hosts[host_name] = host_obj

        self.edges = network.get("edges", [])

    # -------------------------------
    # Internal snapshot helper
    # -------------------------------
    def _snapshot(self):
        return {
            "step": self.step_count,
            "timestamp": datetime.datetime.now().isoformat(),
            "hosts": {
                name: {
                    "is_compromised": host.is_compromised,
                    "access_level": host.access_level,
                    "vulnerabilities": list(host.vulnerabilities),
                    "is_isolated": host.is_isolated,
                }
                for name, host in self.hosts.items()
            },
            "edges": copy.deepcopy(self.edges),
        }

    # -------------------------------
    # Apply Red + Blue actions
    # -------------------------------
    def step(self, red_action, blue_action):
        """
        Apply environment-level effects of Red and Blue actions.

        red_action format (from RedAgent):
            {"action": "scan"|"exploit",
             "target": "H1",
             "success": bool?, ...}

        blue_action format (from BlueAgent or RL agent):
            {"action": "patch"|"isolate"|"idle",
             "target": "H1"?}
        """

        # 1. Apply Red Action Effects (exploit success)
        if red_action and red_action.get("action") == "exploit":
            target = red_action.get("target")
            if target in self.hosts:
                host = self.hosts[target]

                # If host is isolated, exploit cannot succeed
                if host.is_isolated:
                    red_action["success"] = False
                else:
                    success = red_action.get("success", False)
                    if success:
                        host.is_compromised = True
                        host.access_level = "root"
                        # remove ONE exploited vulnerability if present
                        if host.vulnerabilities:
                            host.vulnerabilities.pop(0)

        # 2. Apply Blue Action Effects
        if blue_action:
            b_type = blue_action.get("action")

            if b_type == "patch":
                target = blue_action.get("target")
                if target in self.hosts:
                    host = self.hosts[target]
                    # PATCH ONLY ONE VULNERABILITY AT A TIME
                    if host.vulnerabilities:
                        host.vulnerabilities.pop(0)

            elif b_type == "isolate":
                target = blue_action.get("target")
                if target in self.hosts:
                    host = self.hosts[target]
                    host.is_isolated = True

                    # Remove all edges touching this host
                    new_edges = []
                    for e in self.edges:
                        if target not in e:
                            new_edges.append(e)
                    self.edges = new_edges

            # "idle" does nothing

        # 3. Increase step counter
        self.step_count += 1

        # 4. Return environment snapshot
        return self._snapshot()
