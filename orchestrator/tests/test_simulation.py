import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from orchestrator.orchestrator_core import Orchestrator

orc = Orchestrator("orchestrator/sample_topology.yaml")

orc.load_topology()
orc.build_environment()
orc.init_red_agent()
orc.init_blue_agent()

logs = orc.run_simulation(max_steps=5)

print("\nFINAL LOGS:")
for entry in logs:
    print(entry)
