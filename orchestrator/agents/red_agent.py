# orchestrator/agents/red_agent.py

import random


class RedAgent:
    """
    Simple Red Agent used by:
      - Orchestrator.run_single_step()  (for rule-based sims)
      - RedRLEnvironment (for DQN training via scan/exploit)

    It *does not* directly mutate the environment for exploits.
    The actual state change (compromise, edges, etc.) is applied
    inside Environment.step(red_action, blue_action).
    """

    def __init__(self, environment):
        """
        :param environment: orchestrator.env_builder.Environment
        """
        self.env = environment
        # what we have learned via scans
        self.known_vulns = {}
        self.compromised_hosts = []

    # ------------------------------------------------------------------
    # SCAN
    # ------------------------------------------------------------------
    def scan(self, host_name: str):
        """
        "Passive" information-gathering.
        Environment state is NOT changed here.
        """
        host = self.env.hosts[host_name]

        found_vulns = list(getattr(host, "vulnerabilities", []))
        found_services = list(getattr(host, "services", []))

        # Update local knowledge
        self.known_vulns[host_name] = {
            "vulnerabilities": found_vulns,
            "services": found_services,
        }

        print(f" RED: Scanned {host_name} → vulns={found_vulns}")

        return {
            "action": "scan",
            "target": host_name,
            "vulnerabilities": found_vulns,
            "services": found_services,
        }

    # ------------------------------------------------------------------
    # EXPLOIT
    # ------------------------------------------------------------------
    def exploit(self, host_name: str):
        """
        Attempt to exploit a host.

        - Success probability > 0 if host has at least 1 vulnerability.
        - Does NOT directly modify host state; Environment.step() will
          apply compromise when it sees {"action":"exploit", "success":True,...}.
        """
        host = self.env.hosts[host_name]
        vulns = list(getattr(host, "vulnerabilities", []))

        if not vulns:
            print(f" RED: Exploit failed on {host_name}. No vulnerabilities.")
            return {
                "action": "exploit",
                "success": False,
                "target": host_name,
                "vulnerabilities": vulns,
            }

        # simple success probability
        success = random.random() < 0.7

        if success:
            print(f"RED: Successfully exploited {host_name}!")
            if host_name not in self.compromised_hosts:
                self.compromised_hosts.append(host_name)
        else:
            print(f"RED: Exploit failed on {host_name}.")

        return {
            "action": "exploit",
            "success": success,
            "target": host_name,
            "vulnerabilities": vulns,
        }

    # ------------------------------------------------------------------
    # SIMPLE RULE-BASED POLICY (used by Orchestrator.run_simulation)
    # ------------------------------------------------------------------
    def choose_action(self):
        """
        Simple random policy:
          - Pick a random host
          - 50% scan, 50% exploit
        This is used only in the *non-RL* orchestrator demo.
        """
        host_names = list(self.env.hosts.keys())
        if not host_names:
            return {"action": "idle"}

        target = random.choice(host_names)

        if random.random() < 0.5:
            return self.scan(target)
        else:
            return self.exploit(target)
