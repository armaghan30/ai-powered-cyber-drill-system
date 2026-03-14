import os
import csv
import threading
from datetime import datetime, timezone
from sqlalchemy.orm import Session as DBSession

from backend.config import settings
from backend.database import SessionLocal
from backend.models.training_run import TrainingRun, TrainingStatus
from backend.models.scenario import Scenario
from backend.schemas.training import TrainingCreate


def _run_training_job(training_run_id: int, topology_path: str,
                      agent_role: str, algorithm: str,
                      total_timesteps: int, max_steps: int) -> None:
    db = SessionLocal()
    try:
        run = db.query(TrainingRun).filter(TrainingRun.id == training_run_id).first()
        if not run:
            return

        run.status = TrainingStatus.RUNNING
        run.started_at = datetime.now(timezone.utc)
        db.commit()

        # Determine topology tag for file naming
        topo_tag = os.path.splitext(os.path.basename(topology_path))[0]

        # Import RL components
        if algorithm == "dqn":
            from stable_baselines3 import DQN as AlgoClass
        else:
            from stable_baselines3 import PPO as AlgoClass

        if agent_role == "red":
            from orchestrator.rl_env_red import RedRLEnvironment as EnvClass
        else:
            from orchestrator.rl_env_blue import BlueRLEnvironment as EnvClass

        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.callbacks import BaseCallback

        class RewardLogger(BaseCallback):
            def __init__(self):
                super().__init__(verbose=0)
                self.episode_rewards = []

            def _on_step(self) -> bool:
                for info in self.locals.get("infos", []):
                    if "episode" in info:
                        self.episode_rewards.append(info["episode"]["r"])
                return True

        env = Monitor(EnvClass(topology_path, max_steps=max_steps))

        # Build model with same hyperparameters as existing training scripts
        if algorithm == "dqn":
            model = AlgoClass(
                "MlpPolicy", env,
                learning_rate=1e-3, buffer_size=50_000,
                learning_starts=500, batch_size=64, gamma=0.99,
                target_update_interval=500, exploration_fraction=0.5,
                exploration_initial_eps=1.0, exploration_final_eps=0.05,
                verbose=0,
            )
        else:
            model = AlgoClass(
                "MlpPolicy", env,
                learning_rate=3e-4, n_steps=128, batch_size=64,
                n_epochs=10, gamma=0.99, gae_lambda=0.95,
                clip_range=0.2, ent_coef=0.01, verbose=0,
            )

        callback = RewardLogger()
        model.learn(total_timesteps=total_timesteps, callback=callback)

        # Save model and CSV
        os.makedirs(str(settings.MODELS_DIR), exist_ok=True)
        os.makedirs(str(settings.CSV_DIR), exist_ok=True)

        model_path = str(settings.MODELS_DIR / f"sb3_{algorithm}_{agent_role}_{topo_tag}")
        csv_path = str(settings.CSV_DIR / f"sb3_{algorithm}_{agent_role}_{topo_tag}.csv")

        model.save(model_path)

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["episode", "reward", "topology"])
            for i, r in enumerate(callback.episode_rewards, start=1):
                writer.writerow([i, r, topo_tag])

        # Update DB record
        import numpy as np
        run.status = TrainingStatus.COMPLETED
        run.completed_at = datetime.now(timezone.utc)
        run.episodes_completed = len(callback.episode_rewards)
        run.mean_reward = float(np.mean(callback.episode_rewards)) if callback.episode_rewards else None
        run.model_path = model_path + ".zip"
        run.csv_path = csv_path
        db.commit()

        env.close()

    except Exception as e:
        run = db.query(TrainingRun).filter(TrainingRun.id == training_run_id).first()
        if run:
            run.status = TrainingStatus.FAILED
            run.error_message = str(e)[:1000]
            run.completed_at = datetime.now(timezone.utc)
            db.commit()
    finally:
        db.close()


def start_training(db: DBSession, payload: TrainingCreate) -> TrainingRun:
    """Create a training record and launch the job in a background thread."""
    scenario = db.query(Scenario).filter(Scenario.id == payload.scenario_id).first()
    if not scenario:
        raise ValueError(f"Scenario {payload.scenario_id} not found")

    topology_path = str(settings.TOPOLOGY_DIR / scenario.filename)

    run = TrainingRun(
        scenario_id=payload.scenario_id,
        agent_role=payload.agent_role,
        algorithm=payload.algorithm,
        total_timesteps=payload.total_timesteps,
        max_steps_per_episode=payload.max_steps_per_episode,
    )
    db.add(run)
    db.commit()
    db.refresh(run)

    # Launch background thread
    thread = threading.Thread(
        target=_run_training_job,
        args=(run.id, topology_path, payload.agent_role,
              payload.algorithm, payload.total_timesteps,
              payload.max_steps_per_episode),
        daemon=True,
    )
    thread.start()

    return run


def get_training_run(db: DBSession, run_id: int) -> TrainingRun:
    return db.query(TrainingRun).filter(TrainingRun.id == run_id).first()


def list_training_runs(db: DBSession, scenario_id: int = None,
                       status: str = None) -> list:
    query = db.query(TrainingRun).order_by(TrainingRun.created_at.desc())
    if scenario_id:
        query = query.filter(TrainingRun.scenario_id == scenario_id)
    if status:
        query = query.filter(TrainingRun.status == status)
    return query.all()
