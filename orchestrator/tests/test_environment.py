from orchestrator_core import Orchestrator

orc = Orchestrator("sample_topology.yaml")

# Load YAML
orc.load_topology()

# Build environment
env = orc.build_environment()

print("\n=== HOSTS IN ENVIRONMENT ===")
for h in env.hosts.values():
    print(h)

print("\n=== NETWORK EDGES ===")
print(env.edges)
