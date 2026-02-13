
import random


class BlueAgent:
    

    def __init__(self, environment, patch_cooldown_steps: int = 5):
        
        self.env = environment
        self.patch_cooldown_steps = patch_cooldown_steps
        self.patch_cooldown = 0  

    # -----------------Cooldown helpers--------------------------
    def _tick_cooldown(self):
        if self.patch_cooldown > 0:
            self.patch_cooldown -= 1

    def _can_patch(self) -> bool:
        return self.patch_cooldown == 0

    def _trigger_patch_cooldown(self):
        self.patch_cooldown = self.patch_cooldown_steps

    
    # -----------------Detection rule------------------------------
    def detect(self, last_red_action):
        #Detect if an exploit attempt succeeded.
        
        if not last_red_action:
            return False
        if last_red_action.get("action") == "exploit" and last_red_action.get("success", False):
            return True
        return False

    # ---------------Patch logic --------------------
   
    def make_patch_action(self, host_name):
        print(f"BLUE: Decided to patch {host_name}")
        # Start cooldown
        self._trigger_patch_cooldown()
        return {"action": "patch", "target": host_name}

    # --------------------Isolation logic---------------------------
    
    def make_isolate_action(self, host_name):
      
        print(f"BLUE: Decided to isolate {host_name}")
        return {"action": "isolate", "target": host_name}

    # --------------------Defender policy---------------------------
    
    def choose_action(self, last_red_action):

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
                if host.vulnerabilities  
            ]

            if vulnerable_hosts:
                target = random.choice(vulnerable_hosts)
                return self.make_patch_action(target)

        # 4) Otherwise idle
        return {"action": "idle", "target": None}
