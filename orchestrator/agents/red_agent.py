# orchestrator/agents/red_agent.py

import random


class RedAgent:
    
    def __init__(self, environment):
       
        self.env = environment
        # what we have learned via scans
        self.known_vulns = {}
        self.compromised_hosts = []

    # -----------------------SCAN------------------------------------
    
    def scan(self, host_name: str):
        
        host = self.env.hosts[host_name]

        found_vulns = list(getattr(host, "vulnerabilities", []))
        found_services = list(getattr(host, "services", []))

        # Update local knowledge
        self.known_vulns[host_name] = {
            "vulnerabilities": found_vulns,
            "services": found_services,
        }

        print(f" RED: Scanned {host_name} -> vulns={found_vulns}")

        return {
            "action": "scan",
            "target": host_name,
            "vulnerabilities": found_vulns,
            "services": found_services,
        }

    # ----------------------------EXPLOIT-----------------------------
    
    def exploit(self, host_name: str):
        
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


    def choose_action(self):
        
        host_names = list(self.env.hosts.keys())
        if not host_names:
            return {"action": "idle"}

        target = random.choice(host_names)

        if random.random() < 0.5:
            return self.scan(target)
        else:
            return self.exploit(target)
