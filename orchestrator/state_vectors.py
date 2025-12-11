# orchestrator/state_vectors.py

"""
Helpers to convert orchestrator red/blue state dicts into flat vectors
for DQN agents.

Compatible with:
- Orchestrator.get_red_state()
- Orchestrator.get_blue_state()
"""

from __future__ import annotations

from typing import Dict, List
import numpy as np


def flatten_red_state(red_state: Dict, host_order: List[str]) -> np.ndarray:
    """
    Red state comes from Orchestrator.get_red_state():
    {
        "timestep": int,
        "hosts": {
            name: {
                "scanned": 0/1,
                "vuln_count": int,
                "service_count": int,
                "is_compromised": 0/1,
                "access_level": 0/1/2,
                "is_isolated": 0/1,
            }
        },
        "num_hosts": int,
        "num_compromised": int,
    }

    We flatten as:
    [timestep, num_hosts, num_compromised,
     (for each host in host_order)
       scanned, vuln_count, service_count,
       is_compromised, access_level, is_isolated]
    """
    hosts = red_state.get("hosts", {})
    timestep = float(red_state.get("timestep", 0))
    num_hosts = float(red_state.get("num_hosts", len(host_order)))
    num_compromised = float(red_state.get("num_compromised", 0))

    vec = [timestep, num_hosts, num_compromised]

    for name in host_order:
        h = hosts.get(name, {})
        vec.extend(
            [
                float(h.get("scanned", 0)),
                float(h.get("vuln_count", 0)),
                float(h.get("service_count", 0)),
                float(h.get("is_compromised", 0)),
                float(h.get("access_level", 0)),
                float(h.get("is_isolated", 0)),
            ]
        )

    return np.array(vec, dtype=np.float32)


def flatten_blue_state(blue_state: Dict, host_order: List[str]) -> np.ndarray:
    """
    Blue state comes from Orchestrator.get_blue_state():
    {
        "timestep": int,
        "hosts": {
            name: {
                "is_compromised": 0/1,
                "vulnerability_count": int,
                "access_level": 0/1/2,
                "sensitivity": 0/1/2,
                "is_isolated": 0/1,
            }
        },
        "num_compromised": int,
    }

    We flatten as:
    [timestep, num_compromised,
     (for each host in host_order)
       is_compromised, vulnerability_count,
       access_level, sensitivity, is_isolated]
    """
    hosts = blue_state.get("hosts", {})
    timestep = float(blue_state.get("timestep", 0))
    num_compromised = float(blue_state.get("num_compromised", 0))

    vec = [timestep, num_compromised]

    for name in host_order:
        h = hosts.get(name, {})
        vec.extend(
            [
                float(h.get("is_compromised", 0)),
                float(h.get("vulnerability_count", 0)),
                float(h.get("access_level", 0)),
                float(h.get("sensitivity", 0)),
                float(h.get("is_isolated", 0)),
            ]
        )

    return np.array(vec, dtype=np.float32)
