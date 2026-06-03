import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

from api.routers import markets, positions, model, controls, backtest
from api.websocket import WebSocketBroadcaster, websocket_endpoint
from config.settings import get_settings
from db.session import init_db
from strategies.shared_state import SharedState
from trading.position_tracker import PositionTracker
from risk.risk_controls import RiskControls

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    settings = get_settings()

    logging.basicConfig(level=getattr(logging, settings.LOG_LEVEL, logging.INFO))

    init_db(settings.DB_URL)
    logger.info("Database initialized")

    shared_state = SharedState(
        min_edge=settings.MIN_EDGE_CENTS / 100.0,
        max_ci_width=settings.MAX_CI_WIDTH,
    )
    ws_broadcaster = WebSocketBroadcaster()
    from db.session import get_session_factory
    position_tracker = PositionTracker(db_session_factory=get_session_factory())
    risk_controls = RiskControls(settings=settings, db_session_factory=get_session_factory())

    app.state.settings = settings
    app.state.shared_state = shared_state
    app.state.ws_broadcaster = ws_broadcaster
    app.state.position_tracker = position_tracker
    app.state.risk_controls = risk_controls
    app.state.ensemble_strategy = None
    app.state.blender = None

    # Attempt to initialize Kalshi client if keys are set
    kalshi_client = None
    try:
        from trading.kalshi_client import KalshiClient
        import os
        if os.path.exists(settings.KALSHI_PRIVATE_KEY_PATH):
            kalshi_client = KalshiClient(
                api_key=settings.KALSHI_API_KEY,
                private_key_path=settings.KALSHI_PRIVATE_KEY_PATH,
                base_url=settings.KALSHI_BASE_URL,
                paper_trading=settings.PAPER_TRADING,
            )
            app.state.kalshi_client = kalshi_client
    except Exception as exc:
        logger.warning("Kalshi client init failed (trading disabled): %s", exc)

    # Load model registry from trained artifacts
    model_registry: dict = {}
    try:
        from models.blend import ModelBlender
        from models.ngboost_model import NGBoostTemperatureModel
        from models.qrf_model import QRFTemperatureModel
        from models.calibration import IsotonicCalibrator
        from models.registry import load_latest_artifact
        from config.stations import ALL_ICAO
        from pathlib import Path

        blender = ModelBlender()
        blender.compute_weights_from_log_scores(-2.9, -2.1)  # fallback; overwritten below
        app.state.blender = blender
        model_registry["blender"] = blender

        # Load per-station NGBoost + QRF
        ngboost_models, qrf_models = {}, {}
        for station in ALL_ICAO:
            try:
                ngboost_models[station] = load_latest_artifact(NGBoostTemperatureModel, "ngboost", station)
                logger.info("Loaded NGBoost for %s", station)
            except FileNotFoundError:
                logger.debug("No NGBoost artifact for %s", station)
            try:
                qrf_models[station] = load_latest_artifact(QRFTemperatureModel, "qrf", station)
                logger.info("Loaded QRF for %s", station)
            except FileNotFoundError:
                logger.debug("No QRF artifact for %s", station)

        # Load calibrators
        calibrators = {}
        cal_dir = Path("data/calibrators")
        if cal_dir.exists():
            for cal_path in cal_dir.glob("*.pkl"):
                parts = cal_path.stem.split("_", 1)
                if len(parts) == 2:
                    cal_key = f"{parts[0]}_{parts[1]}"
                    try:
                        calibrators[cal_key] = IsotonicCalibrator.load(str(cal_path))
                    except Exception:
                        pass
            logger.info("Loaded %d calibrators", len(calibrators))

        model_registry["ngboost"]    = ngboost_models
        model_registry["qrf"]        = qrf_models
        model_registry["calibrators"] = calibrators

        loaded = len(ngboost_models)
        logger.info("Model registry loaded: %d stations with NGBoost, %d with QRF, %d calibrators",
                    len(ngboost_models), len(qrf_models), len(calibrators))

    except Exception as exc:
        logger.warning("Model registry init failed: %s", exc)

    # Initialize strategies if client is available
    if kalshi_client is not None:
        from strategies.ensemble_strategy import EnsembleStrategy
        from strategies.d0_strategy import D0Strategy
        from trading.order_manager import OrderManager

        ensemble_strategy = EnsembleStrategy(
            shared_state=shared_state,
            model_registry=model_registry,
            kalshi_client=kalshi_client,
            settings=settings,
        )
        d0_strategy = D0Strategy(shared_state=shared_state, settings=settings)
        order_manager = OrderManager(
            kalshi_client=kalshi_client,
            shared_state=shared_state,
            risk_controls=risk_controls,
            settings=settings,
        )
        risk_controls.set_order_manager(order_manager)
        app.state.ensemble_strategy = ensemble_strategy

        # Start scheduler
        try:
            from scheduler.jobs import build_scheduler
            scheduler = build_scheduler(
                ensemble_strategy=ensemble_strategy,
                d0_strategy=d0_strategy,
                order_manager=order_manager,
                shared_state=shared_state,
                kalshi_client=kalshi_client,
                position_tracker=position_tracker,
                ws_broadcaster=ws_broadcaster,
                settings=settings,
            )
            scheduler.start()
            app.state.scheduler = scheduler
            logger.info("Scheduler started")
            # Fix 4: trigger first ensemble cycle 30s after startup so shared_state
            # is populated quickly (seed_active_markets runs immediately via scheduler,
            # full model cycle fires after GEFS is available)
            import asyncio as _asyncio
            async def _delayed_first_cycle():
                await _asyncio.sleep(30)
                try:
                    await ensemble_strategy.run_cycle()
                    logger.info("First ensemble cycle complete")
                except Exception as exc:
                    logger.warning("First ensemble cycle failed: %s", exc)
            _asyncio.create_task(_delayed_first_cycle())
        except Exception as exc:
            logger.warning("Scheduler failed to start: %s", exc)
    else:
        logger.warning("Running in read-only mode — no Kalshi client")

    logger.info("Startup complete")
    yield

    # Shutdown
    scheduler = getattr(app.state, "scheduler", None)
    if scheduler is not None:
        scheduler.shutdown()
        logger.info("Scheduler stopped")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Kalshi Temp Bot API",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(markets.router, prefix="/api")
    app.include_router(positions.router, prefix="/api")
    app.include_router(model.router, prefix="/api")
    app.include_router(controls.router, prefix="/api")
    app.include_router(backtest.router, prefix="/api")

    @app.websocket("/ws/live")
    async def ws_live(ws: WebSocket):
        await websocket_endpoint(ws, ws.app.state)

    return app


app = create_app()
