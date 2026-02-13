
from __future__ import annotations
from typing import Dict, Any, Tuple


class RewardEngine:

    def compute_rewards(
        self,
        prev_state: Dict[str, Any] | None,
        new_state: Dict[str, Any],
        red_action: Dict[str, Any] | None,
        blue_action: Dict[str, Any] | None,
    ) -> Tuple[float, float]:

        red_r = 0.0
        blue_r = 0.0

        prev_hosts = prev_state.get("hosts", {}) if prev_state else {}
        new_hosts = new_state.get("hosts", {})

        # ---------------------RED ACTION REWARD-------------------
        if red_action:
            a_type = red_action.get("action")  #  IMPORTANT: use "action"
            target = red_action.get("target")

            if a_type == "scan" and target in new_hosts:
                # small info reward for scanning
                red_r += 1.0

                vulns = new_hosts[target].get("vulnerabilities", [])
                if len(vulns) > 0:
                    # scanned a host that actually has vulns
                    red_r += 2.0
                else:
                    # scanned a clean host
                    red_r -= 1.0

            elif a_type == "exploit" and target in new_hosts:
                new_comp = bool(new_hosts[target].get("is_compromised", False))
                prev_comp = bool(prev_hosts.get(target, {}).get("is_compromised", False))

                if new_comp and not prev_comp:
                    # successful exploit
                    red_r += 20.0
                else:
                    # failed exploit
                    red_r -= 5.0

        # --------------------GLOBAL: NEW COMPROMISED HOSTS-----------------------
        for host_name, new_h in new_hosts.items():
            new_comp = bool(new_h.get("is_compromised", False))
            prev_comp = bool(prev_hosts.get(host_name, {}).get("is_compromised", False))

            if new_comp and not prev_comp:
                red_r += 25.0      
                blue_r -= 10.0     

        # -------------BLUE ACTION REWARD-----------------------
        if blue_action:
            b_type = blue_action.get("action")
            target = blue_action.get("target")

            if b_type == "patch" and target in new_hosts:
                prev_vulns = prev_hosts.get(target, {}).get("vulnerabilities", [])
                new_vulns = new_hosts.get(target, {}).get("vulnerabilities", [])

                # if vulnerabilities decreased, good patch
                if len(new_vulns) < len(prev_vulns):
                    blue_r += 6.0
                else:
                    # wasted patch
                    blue_r -= 1.0

            elif b_type == "isolate" and target in new_hosts:
                
                blue_r += 4.0
                red_r -= 3.0

            elif b_type == "idle":
                pass

        return red_r, blue_r
