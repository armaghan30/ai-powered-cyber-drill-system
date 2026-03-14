def test_discover_topologies(client):
    response = client.get("/api/scenarios/discover")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) >= 1
    # Check structure
    filenames = [t["filename"] for t in data]
    assert "sample_topology.yaml" in filenames


def test_create_and_get_scenario(client):
    # Create
    response = client.post("/api/scenarios/", json={
        "name": "Test 2-Host",
        "filename": "sample_topology.yaml",
        "description": "Basic two-host topology",
    })
    assert response.status_code == 201
    data = response.json()
    assert data["name"] == "Test 2-Host"
    assert data["num_hosts"] == 2

    # Get detail
    scenario_id = data["id"]
    response = client.get(f"/api/scenarios/{scenario_id}")
    assert response.status_code == 200
    detail = response.json()
    assert "H1" in detail["host_names"]
    assert "H2" in detail["host_names"]
    assert detail["edge_count"] >= 1


def test_list_scenarios(client):
    client.post("/api/scenarios/", json={
        "name": "Scenario A",
        "filename": "sample_topology.yaml",
    })
    response = client.get("/api/scenarios/")
    assert response.status_code == 200
    assert len(response.json()) >= 1


def test_update_scenario(client):
    resp = client.post("/api/scenarios/", json={
        "name": "Original",
        "filename": "sample_topology.yaml",
    })
    scenario_id = resp.json()["id"]

    resp = client.patch(f"/api/scenarios/{scenario_id}", json={
        "name": "Updated Name",
    })
    assert resp.status_code == 200
    assert resp.json()["name"] == "Updated Name"


def test_delete_scenario(client):
    resp = client.post("/api/scenarios/", json={
        "name": "ToDelete",
        "filename": "sample_topology.yaml",
    })
    scenario_id = resp.json()["id"]

    resp = client.delete(f"/api/scenarios/{scenario_id}")
    assert resp.status_code == 204

    resp = client.get(f"/api/scenarios/{scenario_id}")
    assert resp.status_code == 404


def test_create_scenario_bad_file(client):
    response = client.post("/api/scenarios/", json={
        "name": "Bad",
        "filename": "nonexistent.yaml",
    })
    assert response.status_code == 404


def test_get_nonexistent_scenario(client):
    response = client.get("/api/scenarios/999")
    assert response.status_code == 404
