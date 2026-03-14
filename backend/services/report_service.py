import csv
from pathlib import Path
from sqlalchemy.orm import Session
from sqlalchemy import func

from backend.models.simulation import Simulation, SimulationStep, SimulationStatus
from backend.models.training_run import TrainingRun, TrainingStatus


def get_simulation_summary(db: Session, simulation_id: int) -> dict:
    sim = db.query(Simulation).filter(Simulation.id == simulation_id).first()
    if not sim:
        return None

    steps = db.query(SimulationStep).filter(
        SimulationStep.simulation_id == simulation_id
    ).order_by(SimulationStep.step_number).all()

    red_rewards = [s.red_reward for s in steps]
    blue_rewards = [s.blue_reward for s in steps]

    red_actions = {}
    blue_actions = {}
    for s in steps:
        ra = s.red_action.get("action", "unknown") if s.red_action else "unknown"
        ba = s.blue_action.get("action", "unknown") if s.blue_action else "unknown"
        red_actions[ra] = red_actions.get(ra, 0) + 1
        blue_actions[ba] = blue_actions.get(ba, 0) + 1

    return {
        "simulation_id": simulation_id,
        "status": sim.status.value if hasattr(sim.status, 'value') else sim.status,
        "total_steps": sim.total_steps,
        "total_red_reward": sim.total_red_reward,
        "total_blue_reward": sim.total_blue_reward,
        "red_reward_per_step": red_rewards,
        "blue_reward_per_step": blue_rewards,
        "red_action_counts": red_actions,
        "blue_action_counts": blue_actions,
        "winner": "red" if sim.total_red_reward > sim.total_blue_reward else "blue",
    }


def get_training_rewards(run_id: int) -> list:
    from backend.database import SessionLocal
    db = SessionLocal()
    try:
        run = db.query(TrainingRun).filter(TrainingRun.id == run_id).first()
        if not run or not run.csv_path:
            return None
        csv_path = Path(run.csv_path)
        if not csv_path.exists():
            return None
        rows = []
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append({
                    "episode": int(row["episode"]),
                    "reward": float(row["reward"]),
                    "topology": row["topology"],
                })
        return rows
    finally:
        db.close()


def list_csv_files() -> list:
    from backend.config import settings
    csv_dir = settings.CSV_DIR
    files = []
    if csv_dir.exists():
        for p in sorted(csv_dir.glob("*.csv")):
            name = p.stem  # e.g. sb3_dqn_red_sample_topology
            parts = name.split("_")
            if len(parts) >= 4:
                algorithm = parts[1]
                role = parts[2]
                topology = "_".join(parts[3:])
                files.append({
                    "filename": name,
                    "agent_role": role,
                    "algorithm": algorithm,
                    "topology": topology,
                })
    return files


def get_csv_rewards_by_filename(filename: str) -> list | None:
    from backend.config import settings
    csv_path = settings.CSV_DIR / f"{filename}.csv"
    if not csv_path.exists():
        return None
    rows = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        all_rows = list(reader)
    step = max(1, len(all_rows) // 500)
    for i in range(0, len(all_rows), step):
        row = all_rows[i]
        rows.append({
            "episode": int(row["episode"]),
            "reward": float(row["reward"]),
            "topology": row["topology"],
        })
    return rows


def list_plot_files() -> list:
    from backend.config import settings
    plots_dir = settings.CSV_DIR.parent / "plots"
    files = []
    if plots_dir.exists():
        for p in sorted(plots_dir.glob("*.png")):
            name = p.stem
            files.append({"filename": name, "url": f"/plots/{p.name}"})
    return files


def generate_report(db: Session, simulation_id: int) -> dict | None:
    sim = db.query(Simulation).filter(Simulation.id == simulation_id).first()
    if not sim:
        return None

    steps = db.query(SimulationStep).filter(
        SimulationStep.simulation_id == simulation_id
    ).order_by(SimulationStep.step_number).all()

    red_rewards = [s.red_reward for s in steps]
    blue_rewards = [s.blue_reward for s in steps]

    red_actions = {}
    blue_actions = {}
    for s in steps:
        ra = s.red_action.get("action", "unknown") if s.red_action else "unknown"
        ba = s.blue_action.get("action", "unknown") if s.blue_action else "unknown"
        red_actions[ra] = red_actions.get(ra, 0) + 1
        blue_actions[ba] = blue_actions.get(ba, 0) + 1

    winner = "red" if sim.total_red_reward > sim.total_blue_reward else "blue"

    return {
        "report_type": "simulation",
        "simulation_id": simulation_id,
        "status": sim.status.value if hasattr(sim.status, 'value') else sim.status,
        "total_steps": sim.total_steps,
        "total_red_reward": round(sim.total_red_reward, 2),
        "total_blue_reward": round(sim.total_blue_reward, 2),
        "winner": winner,
        "red_mean_reward": round(sum(red_rewards) / len(red_rewards), 2) if red_rewards else 0,
        "blue_mean_reward": round(sum(blue_rewards) / len(blue_rewards), 2) if blue_rewards else 0,
        "red_max_reward": round(max(red_rewards), 2) if red_rewards else 0,
        "blue_max_reward": round(max(blue_rewards), 2) if blue_rewards else 0,
        "red_action_counts": red_actions,
        "blue_action_counts": blue_actions,
        "red_reward_per_step": [round(r, 2) for r in red_rewards],
        "blue_reward_per_step": [round(r, 2) for r in blue_rewards],
        "steps": [
            {
                "step": s.step_number,
                "red_action": s.red_action.get("action", "unknown") if s.red_action else "unknown",
                "red_target": s.red_action.get("target", "N/A") if s.red_action else "N/A",
                "blue_action": s.blue_action.get("action", "unknown") if s.blue_action else "unknown",
                "blue_target": s.blue_action.get("target", "N/A") if s.blue_action else "N/A",
                "red_reward": round(s.red_reward, 2),
                "blue_reward": round(s.blue_reward, 2),
            }
            for s in steps
        ],
    }


def get_defense_analysis(db: Session) -> dict:
    sims = db.query(Simulation).filter(
        Simulation.status == SimulationStatus.COMPLETED
    ).all()

    if not sims:
        return {
            "total_simulations": 0,
            "blue_wins": 0,
            "red_wins": 0,
            "win_rate": 0,
            "avg_blue_reward": 0,
            "defense_actions": {},
            "recent_simulations": [],
        }

    blue_wins = sum(1 for s in sims if s.total_blue_reward >= s.total_red_reward)
    red_wins = len(sims) - blue_wins
    avg_blue = sum(s.total_blue_reward for s in sims) / len(sims)

    # Get action distribution across all simulations
    all_steps = db.query(SimulationStep).filter(
        SimulationStep.simulation_id.in_([s.id for s in sims])
    ).all()

    defense_actions = {}
    for s in all_steps:
        ba = s.blue_action.get("action", "unknown") if s.blue_action else "unknown"
        defense_actions[ba] = defense_actions.get(ba, 0) + 1

    recent = sorted(sims, key=lambda s: s.created_at or s.id, reverse=True)[:10]

    return {
        "total_simulations": len(sims),
        "blue_wins": blue_wins,
        "red_wins": red_wins,
        "win_rate": round(blue_wins / len(sims) * 100, 1),
        "avg_blue_reward": round(avg_blue, 2),
        "defense_actions": defense_actions,
        "recent_simulations": [
            {
                "id": s.id,
                "total_steps": s.total_steps,
                "red_reward": round(s.total_red_reward, 2),
                "blue_reward": round(s.total_blue_reward, 2),
                "winner": "blue" if s.total_blue_reward >= s.total_red_reward else "red",
            }
            for s in recent
        ],
    }


def get_dashboard_stats(db: Session) -> dict:
    total_simulations = db.query(func.count(Simulation.id)).scalar()
    completed_simulations = db.query(func.count(Simulation.id)).filter(
        Simulation.status == SimulationStatus.COMPLETED
    ).scalar()
    total_training = db.query(func.count(TrainingRun.id)).scalar()
    completed_training = db.query(func.count(TrainingRun.id)).filter(
        TrainingRun.status == TrainingStatus.COMPLETED
    ).scalar()

    return {
        "total_simulations": total_simulations,
        "completed_simulations": completed_simulations,
        "total_training_runs": total_training,
        "completed_training_runs": completed_training,
    }
