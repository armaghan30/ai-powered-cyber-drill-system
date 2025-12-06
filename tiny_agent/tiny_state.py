import numpy as np

def build_observation(hosts):
    """
    Converts host dictionary → flat observation vector.
    """

    obs = []

    for h in ["H1", "H2", "H3", "H4"]:
        data = hosts[h]

        # Binary values
        obs.append(data["discovered"])
        obs.append(data["user"])
        obs.append(data["root"])
        obs.append(data["exploited"])

        # Normalized service count (0–1)
        service_count = len(data["services"]) / 5
        obs.append(service_count)

        # Normalized host value (0–50)
        host_value = data["value"] / 50
        obs.append(host_value)

    return np.array(obs, dtype=np.float32)
