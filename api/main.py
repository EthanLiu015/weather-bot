import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

from api.routers import markets, positions, model, controls
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
            )
            app.state.kalshi_client = kalshi_client
    except Exception as exc:
        logger.warning("Kalshi client init failed (trading disabled): %s", exc)

    # Load model registry if artifacts exist
    model_registry: dict = {}
    try:
        from models.blend import ModelBlender
        blender = ModelBlender()
        app.state.blender = blender
        model_registry["blender"] = blender
    except Exception as exc:
        logger.warning("Blender init failed: %s", exc)

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

    @app.websocket("/ws/live")
    async def ws_live(ws: WebSocket):
        await websocket_endpoint(ws, ws.app.state)

    return app


app = create_app()
