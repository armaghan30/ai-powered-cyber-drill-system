import yaml
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session

from backend.config import settings
from backend.dependencies import get_db
from backend.schemas.scenario import (
    ScenarioCreate, ScenarioUpdate, ScenarioResponse, ScenarioDetailResponse,
)
from backend.services import scenario_service

router = APIRouter(prefix="/scenarios", tags=["Scenarios"])


@router.get("/discover")
def discover_topologies():
    """Scan the orchestrator directory for available topology YAML files."""
    return scenario_service.discover_topologies()


@router.post("/", response_model=ScenarioResponse, status_code=201)
def create_scenario(payload: ScenarioCreate, db: Session = Depends(get_db)):
    try:
        return scenario_service.create_scenario(db, payload)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/", response_model=list[ScenarioResponse])
def list_scenarios(db: Session = Depends(get_db)):
    return scenario_service.list_scenarios(db)


@router.get("/{scenario_id}", response_model=ScenarioDetailResponse)
def get_scenario(scenario_id: int, db: Session = Depends(get_db)):
    result = scenario_service.get_scenario_detail(db, scenario_id)
    if not result:
        raise HTTPException(status_code=404, detail="Scenario not found")
    scenario = result["scenario"]
    return ScenarioDetailResponse(
        id=scenario.id,
        name=scenario.name,
        filename=scenario.filename,
        num_hosts=scenario.num_hosts,
        description=scenario.description,
        created_at=scenario.created_at,
        updated_at=scenario.updated_at,
        topology_data=result["topology_data"],
        host_names=result["host_names"],
        edge_count=result["edge_count"],
    )


@router.patch("/{scenario_id}", response_model=ScenarioResponse)
def update_scenario(scenario_id: int, payload: ScenarioUpdate,
                    db: Session = Depends(get_db)):
    scenario = scenario_service.update_scenario(db, scenario_id, payload)
    if not scenario:
        raise HTTPException(status_code=404, detail="Scenario not found")
    return scenario


@router.delete("/{scenario_id}", status_code=204)
def delete_scenario(scenario_id: int, db: Session = Depends(get_db)):
    if not scenario_service.delete_scenario(db, scenario_id):
        raise HTTPException(status_code=404, detail="Scenario not found")


@router.post("/upload", response_model=ScenarioResponse, status_code=201)
async def upload_topology(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if not file.filename.endswith((".yaml", ".yml")):
        raise HTTPException(status_code=400, detail="File must be a YAML file (.yaml or .yml)")

    content = await file.read()
    try:
        data = yaml.safe_load(content)
    except yaml.YAMLError as e:
        raise HTTPException(status_code=400, detail=f"Invalid YAML: {e}")

    if not data or "network" not in data or "hosts" not in data.get("network", {}):
        raise HTTPException(status_code=400, detail="YAML must contain network.hosts structure")

    dest = settings.TOPOLOGY_DIR / file.filename
    with open(dest, "wb") as f:
        f.write(content)

    name = file.filename.replace(".yaml", "").replace(".yml", "").replace("_", " ").title()

    # if scenario with same filename already exists, just return it
    from backend.models.scenario import Scenario
    existing = db.query(Scenario).filter(Scenario.filename == file.filename).first()
    if existing:
        existing.num_hosts = len(data['network']['hosts'])
        db.commit()
        db.refresh(existing)
        return existing

    payload = ScenarioCreate(
        name=name,
        filename=file.filename,
        description=f"Uploaded topology with {len(data['network']['hosts'])} hosts",
    )
    try:
        return scenario_service.create_scenario(db, payload)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
