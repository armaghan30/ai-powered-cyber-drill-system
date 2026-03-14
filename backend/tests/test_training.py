def _create_scenario(client):
    resp = client.post("/api/scenarios/", json={
        "name": "Train Test",
        "filename": "sample_topology.yaml",
    })
    return resp.json()["id"]


def test_start_training_returns_queued(client):
    scenario_id = _create_scenario(client)
    resp = client.post("/api/training/", json={
        "scenario_id": scenario_id,
        "agent_role": "red",
        "algorithm": "dqn",
        "total_timesteps": 1000,
    })
    assert resp.status_code == 202
    data = resp.json()
    assert data["status"] in ("queued", "running")
    assert data["agent_role"] == "red"
    assert data["algorithm"] == "dqn"


def test_list_training_runs(client):
    scenario_id = _create_scenario(client)
    client.post("/api/training/", json={
        "scenario_id": scenario_id,
        "agent_role": "blue",
        "algorithm": "dqn",
        "total_timesteps": 1000,
    })
    resp = client.get("/api/training/")
    assert resp.status_code == 200
    assert len(resp.json()) >= 1


def test_get_training_run(client):
    scenario_id = _create_scenario(client)
    resp = client.post("/api/training/", json={
        "scenario_id": scenario_id,
        "agent_role": "red",
        "algorithm": "dqn",
        "total_timesteps": 1000,
    })
    run_id = resp.json()["id"]

    resp = client.get(f"/api/training/{run_id}")
    assert resp.status_code == 200
    assert resp.json()["id"] == run_id


def test_invalid_agent_role(client):
    scenario_id = _create_scenario(client)
    resp = client.post("/api/training/", json={
        "scenario_id": scenario_id,
        "agent_role": "green",
        "algorithm": "dqn",
        "total_timesteps": 1000,
    })
    assert resp.status_code == 422


def test_invalid_algorithm(client):
    scenario_id = _create_scenario(client)
    resp = client.post("/api/training/", json={
        "scenario_id": scenario_id,
        "agent_role": "red",
        "algorithm": "invalid",
        "total_timesteps": 1000,
    })
    assert resp.status_code == 422


def test_nonexistent_training_run(client):
    resp = client.get("/api/training/999")
    assert resp.status_code == 404
