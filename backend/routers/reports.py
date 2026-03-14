from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.dependencies import get_db
from backend.services import report_service

router = APIRouter(prefix="/reports", tags=["Reports"])


@router.get("/dashboard")
def get_dashboard(db: Session = Depends(get_db)):
    """Get aggregate system statistics."""
    return report_service.get_dashboard_stats(db)


@router.get("/simulations/{simulation_id}/summary")
def get_simulation_report(simulation_id: int, db: Session = Depends(get_db)):
    """Get a simulation report with action distribution and reward timeline."""
    result = report_service.get_simulation_summary(db, simulation_id)
    if not result:
        raise HTTPException(status_code=404, detail="Simulation not found")
    return result


# Static CSV/plot endpoints (must come BEFORE the parameterized route)
@router.get("/training/csvfiles")
def list_csv_files():
    """List all available training CSV files on disk."""
    return report_service.list_csv_files()


@router.get("/training/csv/{filename}")
def get_csv_rewards(filename: str):
    """Get reward data from a CSV file by filename (no DB required)."""
    result = report_service.get_csv_rewards_by_filename(filename)
    if result is None:
        raise HTTPException(status_code=404, detail="CSV file not found")
    return result


@router.get("/training/plots")
def list_plot_files():
    """List all available training plot PNG files."""
    return report_service.list_plot_files()


@router.get("/training/{run_id}/rewards")
def get_training_rewards(run_id: int):
    """Get episode reward data from training CSV."""
    result = report_service.get_training_rewards(run_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Training rewards not found")
    return result


@router.get("/defense-analysis")
def get_defense_analysis(db: Session = Depends(get_db)):
    """Get Blue agent defense performance analysis across all simulations."""
    return report_service.get_defense_analysis(db)


@router.get("/simulations/{simulation_id}/report")
def generate_report(simulation_id: int, db: Session = Depends(get_db)):
    """Generate a comprehensive downloadable report for a simulation."""
    result = report_service.generate_report(db, simulation_id)
    if not result:
        raise HTTPException(status_code=404, detail="Simulation not found")
    return result
