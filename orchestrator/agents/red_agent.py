# red_agent.py

import random


class RedAgent:

    def __init__(self, environment):
        self.env = environment

        # what the red agent has discovered
        self.known_vulns = {}
        self.compromised_hosts = []

    # ---------------- SCAN HOST -----------------
    def scan(self, host_name):
        host = self.env.hosts[host_name]

        found_vulns = list(host.vulnerabilities)
        found_services = list(host.services)

        # Store knowledge in RL-friendly format
        self.known_vulns[host_name] = {
            "vulnerabilities": found_vulns,
            "services": found_services
        }

        print(f" RED: Scanned {host_name} → vulns={found_vulns}")

        # MUST RETURN RL-COMPATIBLE FORMAT
        return {
            "action": "scan",
            "target": host_name,
            "vulnerabilities": found_vulns,   # FIXED KEY NAME
            "services": found_services        # FIXED KEY NAME
        }

    # ---------------- EXPLOIT HOST -----------------
    def exploit(self, host_name):
        host = self.env.hosts[host_name]

        if not host.vulnerabilities:
            print(f" RED: Exploit failed on {host_name}. No vulnerabilities.")
            return {
                "action": "exploit",
                "success": False,
                "target": host_name,
                "vulnerabilities": list(host.vulnerabilities)
            }

        success = random.random() < 0.7  # 70% success rate

        if success:
            host.is_compromised = True
            host.access_level = "root"
            if host_name not in self.compromised_hosts:
                self.compromised_hosts.append(host_name)

            print(f"RED: Successfully exploited {host_name}!")

            return {
                "action": "exploit",
                "success": True,
                "target": host_name,
                "vulnerabilities": list(host.vulnerabilities)
            }

        else:
            print(f"RED: Exploit failed on {host_name}.")
            return {
                "action": "exploit",
                "success": False,
                "target": host_name,
                "vulnerabilities": list(host.vulnerabilities)
            }

    # ---------------- SIMPLE POLICY (not used for RL) -----------------
    def choose_action(self):
        target = random.choice(list(self.env.hosts.keys()))

        # Basic heuristic: random scan 50%, exploit 50%
        if random.random() < 0.5:
            return self.scan(target)
        return self.exploit(target)
