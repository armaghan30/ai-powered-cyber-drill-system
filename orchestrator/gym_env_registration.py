"""
Gymnasium Registration for Cyber Range Environments.
"""

from gymnasium.envs.registration import register


def register_envs():
    # RED agent env
    register(
        id="CyberRange-Red-v1",
        entry_point="orchestrator.rl_env_red:RedRLEnvironment",
        kwargs={"topology_path": "orchestrator/sample_topology.yaml"},
        max_episode_steps=20,
    )

    # BLUE agent env
    register(
        id="CyberRange-Blue-v1",
        entry_point="orchestrator.rl_env_blue:BlueRLEnvironment",
        kwargs={"topology_path": "orchestrator/sample_topology.yaml"},
        max_episode_steps=20,
    )

    print("[GYM] Cyber Range environments registered.")
