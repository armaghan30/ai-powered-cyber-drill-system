# test_blue_agent.py

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from orchestrator.orchestrator_core import Orchestrator

orc = Orchestrator("orchestrator/sample_topology.yaml")

orc.load_topology()
env = orc.build_environment()

red = orc.init_red_agent()
blue = orc.init_blue_agent()

print("\n=== TEST: RED SCAN ===")
scan_action = red.scan("H1")
blue_response = blue.choose_action(scan_action)
print("Blue Response:", blue_response)

print("\n=== TEST: RED EXPLOIT ===")
exploit_action = red.exploit("H1")
blue_response = blue.choose_action(exploit_action)
print("Blue Response:", blue_response)

print("\n=== ALERTS ===")
print(blue.alerts)

print("\n=== PATCHED HOSTS ===")
print(blue.patched_hosts)

print("\n=== ISOLATED HOSTS ===")
print(blue.isolated_hosts)
