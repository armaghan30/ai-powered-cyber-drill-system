from yaml_loader import YAMLLoader

loader = YAMLLoader("sample_topology.yaml")
topology = loader.parse()

print("\n=== Parsed YAML ===")
print(topology)
