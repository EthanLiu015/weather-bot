import logging
from datetime import datetime
from pathlib import Path
from sqlalchemy.orm import Session
from db.models import ModelArtifact
from db.session import get_session

logger = logging.getLogger(__name__)

ARTIFACT_BASE = Path("data/models")


def _artifact_path(model_type: str, station: str, version: str, ext: str = "pkl") -> Path:
    return ARTIFACT_BASE / model_type / station / f"{version}.{ext}"


def save_artifact(
    model_obj,
    model_type: str,
    station: str,
    crps_val: float | None = None,
    mae_val: float | None = None,
) -> str:
    version = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    path = _artifact_path(model_type, station, version)
    path.parent.mkdir(parents=True, exist_ok=True)
    model_obj.save(str(path))

    with get_session() as db:
        artifact = ModelArtifact(
            model_type=model_type,
            station=station,
            version=version,
            path=str(path),
            trained_at=datetime.utcnow(),
            crps_val=crps_val,
            mae_val=mae_val,
        )
        db.add(artifact)

    logger.info("Artifact saved: %s/%s v%s → %s", model_type, station, version, path)
    return str(path)


def load_latest_artifact(model_cls, model_type: str, station: str):
    with get_session() as db:
        artifact = (
            db.query(ModelArtifact)
            .filter(ModelArtifact.model_type == model_type, ModelArtifact.station == station)
            .order_by(ModelArtifact.trained_at.desc())
            .first()
        )
    if artifact is None:
        raise FileNotFoundError(f"No artifact found for {model_type}/{station}")
    return model_cls.load(artifact.path)


def list_artifacts(model_type: str | None = None, station: str | None = None) -> list[dict]:
    with get_session() as db:
        q = db.query(ModelArtifact)
        if model_type:
            q = q.filter(ModelArtifact.model_type == model_type)
        if station:
            q = q.filter(ModelArtifact.station == station)
        rows = q.order_by(ModelArtifact.trained_at.desc()).all()
    return [
        {
            "id": r.id,
            "model_type": r.model_type,
            "station": r.station,
            "version": r.version,
            "path": r.path,
            "trained_at": r.trained_at,
            "crps_val": r.crps_val,
            "mae_val": r.mae_val,
        }
        for r in rows
    ]
