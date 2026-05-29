from fastapi import APIRouter, Request
from db.models import ModelArtifact, CalibrationSnapshot, ForecastRun
from db.session import get_session
from datetime import datetime

router = APIRouter(prefix="/model", tags=["model"])


@router.get("/fairvalues")
async def get_fair_values(request: Request) -> dict:
    shared_state = request.app.state.shared_state
    snap = shared_state.snapshot()
    return {
        ticker: {
            "fair_a": state.get("fair_a"),
            "fair_b": state.get("fair_b"),
            "blended_fair": state.get("blended_fair"),
            "ci_width": state.get("ci_width"),
        }
        for ticker, state in snap.items()
    }


@router.get("/status")
async def get_model_status(request: Request) -> dict:
    blender = getattr(request.app.state, "blender", None)
    weights = blender.weights if blender else {"ngboost": 0.5, "qrf": 0.5}

    shared_state = request.app.state.shared_state
    snap = shared_state.snapshot()
    ci_widths = [s.get("ci_width", 0.0) for s in snap.values() if s.get("ci_width") is not None]
    mean_ci = sum(ci_widths) / len(ci_widths) if ci_widths else 0.0

    with get_session() as db:
        latest_forecast = db.query(ForecastRun).order_by(ForecastRun.created_at.desc()).first()
        latest_artifact = db.query(ModelArtifact).order_by(ModelArtifact.trained_at.desc()).first()
        cal_snaps = db.query(CalibrationSnapshot).order_by(CalibrationSnapshot.recorded_at.desc()).limit(10).all()

    cal_health = []
    for snap_row in cal_snaps:
        slope = snap_row.reliability_slope
        if slope is None:
            status = "unknown"
        elif 0.8 <= slope <= 1.2:
            status = "green"
        elif 0.6 <= slope <= 1.4:
            status = "amber"
        else:
            status = "red"
        cal_health.append({
            "station": snap_row.station,
            "lead_bucket": snap_row.lead_bucket,
            "brier_score": snap_row.brier_score,
            "reliability_slope": slope,
            "status": status,
        })

    return {
        "last_forecast_run": latest_forecast.created_at.isoformat() if latest_forecast else None,
        "last_model_trained": latest_artifact.trained_at.isoformat() if latest_artifact else None,
        "blend_weights": weights,
        "mean_ci_width": mean_ci,
        "calibration_health": cal_health,
        "num_active_tickers": len(snap),
    }
