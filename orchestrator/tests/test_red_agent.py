# test_red_agent.py

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from orchestrator.orchestrator_core import Orchestrator

orc = Orchestrator("orchestrator/sample_topology.yaml")  # <-- FIXED PATH

orc.load_topology()
env = orc.build_environment()
red = orc.init_red_agent()

print("\n=== TEST: RED SCAN ===")
red.scan("H1")

print("\n=== TEST: RED EXPLOIT ===")
red.exploit("H1")

print("\n=== COMPROMISED HOSTS ===")
print(red.compromised_hosts)
