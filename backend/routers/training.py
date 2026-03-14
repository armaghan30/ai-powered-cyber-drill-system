from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from backend.dependencies import get_db
from backend.schemas.training import TrainingCreate, TrainingResponse
from backend.services import training_service

router = APIRouter(prefix="/training", tags=["Training"])


@router.post("/", response_model=TrainingResponse, status_code=202)
def start_training(payload: TrainingCreate, db: Session = Depends(get_db)):
    try:
        return training_service.start_training(db, payload)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/", response_model=list[TrainingResponse])
def list_training_runs(
    scenario_id: int = Query(None),
    status: str = Query(None),
    db: Session = Depends(get_db),
):
    return training_service.list_training_runs(db, scenario_id, status)


@router.get("/{run_id}", response_model=TrainingResponse)
def get_training_run(run_id: int, db: Session = Depends(get_db)):
    run = training_service.get_training_run(db, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Training run not found")
    return run
