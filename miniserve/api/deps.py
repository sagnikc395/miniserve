from fastapi import Request

from miniserve.api.errors import APIError
from miniserve.engine.engine import AsyncEngine


def get_engine(request: Request) -> AsyncEngine:
    engine = getattr(request.app.state, "engine", None)
    if engine is None:
        raise APIError(503, "engine is not ready", type="server_error")
    return engine
