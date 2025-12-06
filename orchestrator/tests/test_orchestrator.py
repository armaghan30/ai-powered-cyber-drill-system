from orchestrator_core import Orchestrator

orc = Orchestrator("sample_topology.yaml")

topo = orc.load_topology()

print("\n=== TOPOLOGY FROM ORCHESTRATOR ===")
print(topo)
