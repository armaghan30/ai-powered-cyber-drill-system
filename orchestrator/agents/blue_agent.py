# blue_agent.py

import random


class BlueAgent:
    """
    Simple rule-based defender with:
      - detection of successful exploits
      - patching with cooldown (every 3 steps)
      - patch ONE vulnerability per patch
      - isolation of compromised hosts
    """

    def __init__(self, environment, patch_cooldown_steps: int = 3):
        self.env = environment
        self.patch_cooldown_steps = patch_cooldown_steps
        self.patch_cooldown = 0  # steps remaining until next patch is allowed

    # --------------------------------------------------
    # Cooldown helpers
    # --------------------------------------------------
    def _tick_cooldown(self):
        if self.patch_cooldown > 0:
            self.patch_cooldown -= 1

    def _can_patch(self) -> bool:
        return self.patch_cooldown == 0

    def _trigger_patch_cooldown(self):
        self.patch_cooldown = self.patch_cooldown_steps

    # --------------------------------------------------
    # Detection rule
    # --------------------------------------------------
    def detect(self, last_red_action):
        """
        Detect if an exploit attempt succeeded.
        Must be safe even if "success" key doesn't exist (e.g. SCAN).
        """
        if last_red_action.get("action") == "exploit" and last_red_action.get("success", False):
            return True
        return False

    # --------------------------------------------------
    # Patch logic (env step actually applies patch)
    # --------------------------------------------------
    def make_patch_action(self, host_name):
        # Do NOT modify env here; env_builder.Environment.step() will handle it.
        print(f"BLUE: Decided to patch {host_name}")
        # Start cooldown
        self._trigger_patch_cooldown()
        return {"action": "patch", "target": host_name}

    # --------------------------------------------------
    # Isolation logic
    # --------------------------------------------------
    def make_isolate_action(self, host_name):
        # Isolation effect (removing edges) will be applied in env.step()
        print(f"BLUE: Decided to isolate {host_name}")
        return {"action": "isolate", "target": host_name}

    # --------------------------------------------------
    # Defender policy
    # --------------------------------------------------
    def choose_action(self, last_red_action):
        """
        Rule-based defender:
            1) Tick cooldown (time passes each step)
            2) If exploit success detected -> isolate that host (no cooldown).
            3) Else, if cooldown allows, patch ONE vulnerable host.
            4) Otherwise, idle.
        """

        # 1) Time passes
        self._tick_cooldown()

        # 2) Detection: successful exploit -> isolate compromised host
        if self.detect(last_red_action):
            target = last_red_action.get("target")
            if target:
                return self.make_isolate_action(target)

        # 3) If patch is allowed (cooldown finished), patch one vulnerable host
        if self._can_patch():
            vulnerable_hosts = [
                name for name, host in self.env.hosts.items()
                if host.vulnerabilities  # host still has at least one vuln
            ]

            if vulnerable_hosts:
                # you can choose host by some heuristic; here random
                target = random.choice(vulnerable_hosts)
                return self.make_patch_action(target)

        # 4) Otherwise idle
        return {"action": "idle"}
