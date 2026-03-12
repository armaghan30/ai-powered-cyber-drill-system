from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.config import settings
from backend.database import engine, Base, SessionLocal
from backend.routers import health, scenarios, simulations, training, reports, auth


def _seed_scenarios():
    from backend.models.scenario import Scenario
    from backend.services.scenario_service import discover_topologies, create_scenario
    from backend.schemas.scenario import ScenarioCreate

    db = SessionLocal()
    try:
        count = db.query(Scenario).count()
        if count > 0:
            return
        topologies = discover_topologies()
        for topo in topologies:
            name = topo["filename"].replace(".yaml", "").replace("_", " ").title()
            payload = ScenarioCreate(
                name=name,
                filename=topo["filename"],
                description=f"{topo['num_hosts']}-host network topology",
            )
            create_scenario(db, payload)
    finally:
        db.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: create database tables
    Base.metadata.create_all(bind=engine)
    # Auto-seed scenarios from YAML files
    _seed_scenarios()
    yield
    # Shutdown: cleanup if needed


app = FastAPI(
    title=settings.PROJECT_NAME,
    version="4.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for training plots
plots_dir = settings.CSV_DIR.parent / "plots"
if plots_dir.exists():
    app.mount("/plots", StaticFiles(directory=str(plots_dir)), name="plots")

# Register routers
app.include_router(auth.router, prefix=settings.API_V1_PREFIX)
app.include_router(health.router, prefix=settings.API_V1_PREFIX)
app.include_router(scenarios.router, prefix=settings.API_V1_PREFIX)
app.include_router(simulations.router, prefix=settings.API_V1_PREFIX)
app.include_router(training.router, prefix=settings.API_V1_PREFIX)
app.include_router(reports.router, prefix=settings.API_V1_PREFIX)
