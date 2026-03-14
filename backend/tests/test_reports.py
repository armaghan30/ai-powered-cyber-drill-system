def _create_scenario(client):
    resp = client.post("/api/scenarios/", json={
        "name": "Report Test",
        "filename": "sample_topology.yaml",
    })
    return resp.json()["id"]


def test_dashboard_stats(client):
    resp = client.get("/api/reports/dashboard")
    assert resp.status_code == 200
    data = resp.json()
    assert "total_simulations" in data
    assert "completed_simulations" in data
    assert "total_training_runs" in data
    assert "completed_training_runs" in data


def test_simulation_summary(client):
    scenario_id = _create_scenario(client)
    resp = client.post("/api/simulations/run", json={
        "scenario_id": scenario_id,
        "max_steps": 5,
    })
    sim_id = resp.json()["id"]

    resp = client.get(f"/api/reports/simulations/{sim_id}/summary")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_steps"] == 5
    assert "red_action_counts" in data
    assert "blue_action_counts" in data
    assert "winner" in data
    assert data["winner"] in ("red", "blue")


def test_simulation_summary_not_found(client):
    resp = client.get("/api/reports/simulations/999/summary")
    assert resp.status_code == 404


def test_training_rewards_not_found(client):
    resp = client.get("/api/reports/training/999/rewards")
    assert resp.status_code == 404
