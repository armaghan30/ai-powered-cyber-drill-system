import pytest
from orchestrator.orchestrator_core import Orchestrator

TOPOLOGY_PATH = "orchestrator/sample_topology.yaml"


@pytest.fixture
def orchestrator():
    orch = Orchestrator(TOPOLOGY_PATH)
    orch.load_topology()
    orch.build_environment()
    orch.init_red_agent()
    orch.init_blue_agent()
    return orch


@pytest.fixture
def environment(orchestrator):
    #Return the Environment instance from the orchestrator
    
    return orchestrator.environment


@pytest.fixture
def red_agent(orchestrator):
    #Return the RedAgent instance
    
    return orchestrator.red_agent


@pytest.fixture
def blue_agent(orchestrator):
    #Return the BlueAgent instance
    
    return orchestrator.blue_agent
