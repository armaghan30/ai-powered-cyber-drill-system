import yaml
from pathlib import Path
from sqlalchemy.orm import Session

from backend.config import settings
from backend.models.scenario import Scenario
from backend.schemas.scenario import ScenarioCreate, ScenarioUpdate


def _resolve_topology_path(filename: str) -> Path:
    return settings.TOPOLOGY_DIR / filename


def _parse_topology(filepath: Path) -> dict:
    with open(filepath, "r") as f:
        return yaml.safe_load(f)


def _count_hosts(topology_data: dict) -> int:
    return len(topology_data.get("network", {}).get("hosts", {}))


def discover_topologies() -> list:
    """Scan orchestrator/ for YAML topology files and return metadata."""
    results = []
    for path in settings.TOPOLOGY_DIR.glob("*.yaml"):
        if path.name.startswith("."):
            continue
        try:
            data = _parse_topology(path)
            hosts = data.get("network", {}).get("hosts", {})
            results.append({
                "filename": path.name,
                "num_hosts": len(hosts),
                "host_names": list(hosts.keys()),
            })
        except Exception:
            continue
    return results


def create_scenario(db: Session, payload: ScenarioCreate) -> Scenario:
    filepath = _resolve_topology_path(payload.filename)
    if not filepath.exists():
        raise FileNotFoundError(f"Topology file not found: {payload.filename}")

    topology_data = _parse_topology(filepath)
    num_hosts = _count_hosts(topology_data)

    scenario = Scenario(
        name=payload.name,
        filename=payload.filename,
        num_hosts=num_hosts,
        description=payload.description,
    )
    db.add(scenario)
    db.commit()
    db.refresh(scenario)
    return scenario


def get_scenario(db: Session, scenario_id: int) -> Scenario:
    return db.query(Scenario).filter(Scenario.id == scenario_id).first()


def get_scenario_detail(db: Session, scenario_id: int) -> dict:
    scenario = get_scenario(db, scenario_id)
    if not scenario:
        return None
    filepath = _resolve_topology_path(scenario.filename)
    topology_data = _parse_topology(filepath)
    hosts = topology_data.get("network", {}).get("hosts", {})
    edges = topology_data.get("network", {}).get("edges", [])
    return {
        "scenario": scenario,
        "topology_data": topology_data,
        "host_names": list(hosts.keys()),
        "edge_count": len(edges),
    }


def list_scenarios(db: Session) -> list:
    return db.query(Scenario).order_by(Scenario.created_at.desc()).all()


def update_scenario(db: Session, scenario_id: int, payload: ScenarioUpdate) -> Scenario:
    scenario = get_scenario(db, scenario_id)
    if not scenario:
        return None
    update_data = payload.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(scenario, field, value)
    db.commit()
    db.refresh(scenario)
    return scenario


def delete_scenario(db: Session, scenario_id: int) -> bool:
    scenario = get_scenario(db, scenario_id)
    if not scenario:
        return False
    db.delete(scenario)
    db.commit()
    return True
