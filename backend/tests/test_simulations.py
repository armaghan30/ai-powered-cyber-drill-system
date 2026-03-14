def _create_scenario(client):
    resp = client.post("/api/scenarios/", json={
        "name": "Sim Test",
        "filename": "sample_topology.yaml",
    })
    return resp.json()["id"]


def test_run_full_simulation(client):
    scenario_id = _create_scenario(client)
    response = client.post("/api/simulations/run", json={
        "scenario_id": scenario_id,
        "max_steps": 5,
    })
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "completed"
    assert data["total_steps"] == 5
    assert len(data["steps"]) == 5
    # Check step structure
    step = data["steps"][0]
    assert "red_action" in step
    assert "blue_action" in step
    assert "red_reward" in step
    assert "blue_reward" in step


def test_step_by_step_simulation(client):
    scenario_id = _create_scenario(client)

    # Create session
    resp = client.post("/api/simulations/", json={
        "scenario_id": scenario_id,
        "max_steps": 3,
    })
    assert resp.status_code == 201
    session_id = resp.json()["session_id"]
    assert resp.json()["status"] == "running"

    # Run 3 steps
    for i in range(3):
        resp = client.post(f"/api/simulations/{session_id}/step")
        assert resp.status_code == 200
        data = resp.json()
        assert data["step"]["step_number"] == i + 1

    # Last step should show completed
    assert data["simulation_status"] == "completed"
    assert data["steps_remaining"] == 0


def test_step_after_completion_fails(client):
    scenario_id = _create_scenario(client)
    resp = client.post("/api/simulations/", json={
        "scenario_id": scenario_id,
        "max_steps": 1,
    })
    session_id = resp.json()["session_id"]

    # Complete it
    client.post(f"/api/simulations/{session_id}/step")

    # Try another step - should fail
    resp = client.post(f"/api/simulations/{session_id}/step")
    assert resp.status_code == 400


def test_list_simulations(client):
    scenario_id = _create_scenario(client)
    client.post("/api/simulations/run", json={
        "scenario_id": scenario_id,
        "max_steps": 3,
    })
    resp = client.get("/api/simulations/")
    assert resp.status_code == 200
    assert len(resp.json()) >= 1


def test_get_simulation_detail(client):
    scenario_id = _create_scenario(client)
    resp = client.post("/api/simulations/run", json={
        "scenario_id": scenario_id,
        "max_steps": 3,
    })
    sim_id = resp.json()["id"]

    resp = client.get(f"/api/simulations/{sim_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["steps"]) == 3
    assert data["final_state"] is not None


def test_invalid_scenario_simulation(client):
    resp = client.post("/api/simulations/run", json={
        "scenario_id": 999,
        "max_steps": 5,
    })
    assert resp.status_code == 400
