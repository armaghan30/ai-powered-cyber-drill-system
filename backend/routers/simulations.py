from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from backend.dependencies import get_db
from backend.schemas.simulation import (
    SimulationCreate, SimulationResponse, SimulationDetailResponse,
    StepActionResponse,
)
from backend.services import simulation_service

router = APIRouter(prefix="/simulations", tags=["Simulations"])


@router.post("/", response_model=SimulationResponse, status_code=201)
def create_simulation(payload: SimulationCreate, db: Session = Depends(get_db)):
    """Create a new simulation session for step-by-step mode."""
    try:
        return simulation_service.create_simulation(db, payload)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/run", response_model=SimulationDetailResponse)
def run_full_simulation(payload: SimulationCreate, db: Session = Depends(get_db)):
    """Run a complete simulation (all steps at once)."""
    try:
        sim = simulation_service.run_full_simulation(db, payload)
        return sim
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{session_id}/step", response_model=StepActionResponse)
def run_step(session_id: str, db: Session = Depends(get_db)):
    """Execute the next step in a step-by-step simulation."""
    try:
        result = simulation_service.run_step(db, session_id)
        return StepActionResponse(
            session_id=session_id,
            step=result["step"],
            simulation_status=result["simulation_status"],
            steps_remaining=result["steps_remaining"],
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/", response_model=list[SimulationResponse])
def list_simulations(
    scenario_id: int = Query(None),
    db: Session = Depends(get_db),
):
    return simulation_service.list_simulations(db, scenario_id)


@router.get("/{simulation_id}", response_model=SimulationDetailResponse)
def get_simulation(simulation_id: int, db: Session = Depends(get_db)):
    sim = simulation_service.get_simulation(db, simulation_id)
    if not sim:
        raise HTTPException(status_code=404, detail="Simulation not found")
    return sim


@router.delete("/{session_id}/abandon", status_code=204)
def abandon_simulation(session_id: str, db: Session = Depends(get_db)):
    """Abandon a running step-by-step simulation and free resources."""
    simulation_service.cleanup_session(session_id)
