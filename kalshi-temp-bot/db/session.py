from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from contextlib import contextmanager
from typing import Generator
from db.models import Base


_engine = None
_SessionLocal = None


def init_db(db_url: str) -> None:
    global _engine, _SessionLocal
    _engine = create_engine(
        db_url,
        connect_args={"check_same_thread": False} if "sqlite" in db_url else {},
        echo=False,
    )
    _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_engine)
    Base.metadata.create_all(bind=_engine)


def get_engine():
    if _engine is None:
        raise RuntimeError("DB not initialized — call init_db() first")
    return _engine


@contextmanager
def get_session() -> Generator[Session, None, None]:
    if _SessionLocal is None:
        raise RuntimeError("DB not initialized — call init_db() first")
    session = _SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_session_factory() -> sessionmaker:
    if _SessionLocal is None:
        raise RuntimeError("DB not initialized — call init_db() first")
    return _SessionLocal
