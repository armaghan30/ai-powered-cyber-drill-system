# orchestrator/reward_engine.py

from __future__ import annotations
from typing import Dict, Any, Tuple


class RewardEngine:
    """
    Reward engine for Red and Blue with simple shaping.

    It expects environment snapshots in the format:

    {
        "step": int,
        "hosts": {
            host_name: {
                "is_compromised": bool,
                "access_level": str,
                "vulnerabilities": [str, ...],
                "is_isolated": bool,
            },
            ...
        },
        "edges": [...]
    }

    and actions like:
    red_action = {"action": "scan"|"exploit", "target": "H1", "success": bool?}
    blue_action = {"action": "patch"|"isolate"|"idle", "target": "H1"?}
    """

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

        # ------------------ RED ACTION REWARDS ------------------
        if red_action:
            a_type = red_action.get("action")
            target = red_action.get("target")

            if a_type == "scan":
                # Red gets a small reward for scanning
                red_r += 1.0

                if target in prev_hosts:
                    prev_vulns = prev_hosts[target].get("vulnerabilities", [])
                    if len(prev_vulns) > 0:
                        # Scanned a host that actually had vulnerabilities
                        red_r += 2.0
                    else:
                        # Scanned a clean host
                        red_r -= 1.0

            elif a_type == "exploit":
                success = red_action.get("success", False)
                if success:
                    # Successful exploit
                    red_r += 15.0
                else:
                    # Failed exploit
                    red_r -= 3.0
                    if target in prev_hosts:
                        prev_vulns = prev_hosts[target].get("vulnerabilities", [])
                        if len(prev_vulns) == 0:
                            # Tried to exploit a host with no vulns
                            red_r -= 2.0

        # ------------------ NEW COMPROMISE BONUS ------------------
        # If a host is newly compromised, reward Red and penalize Blue
        for host, new_h in new_hosts.items():
            new_comp = new_h.get("is_compromised", False)
            prev_comp = prev_hosts.get(host, {}).get("is_compromised", False)

            if new_comp and not prev_comp:
                red_r += 25.0
                blue_r -= 10.0

        # ------------------ BLUE ACTION REWARDS ------------------
        if blue_action:
            b_type = blue_action.get("action")
            target = blue_action.get("target")

            if b_type == "patch" and target in prev_hosts:
                prev_vulns = prev_hosts[target].get("vulnerabilities", [])
                if len(prev_vulns) > 0:
                    # Patch a vulnerable host -> good
                    blue_r += 5.0
                else:
                    # Wasted patch
                    blue_r -= 1.0

            elif b_type == "isolate":
                # Isolation helps stop spread, but can be disruptive
                blue_r += 2.0   # lower than before
                red_r -= 5.0

            # "idle" → zero reward

        return red_r, blue_r
