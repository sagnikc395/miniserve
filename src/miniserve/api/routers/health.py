from fastapi import APIRouter, Depends

from miniserve.api.deps import get_engine
from miniserve.engine.engine import AsyncEngine

router = APIRouter()


@router.get("/health")
def health(aeng: AsyncEngine = Depends(get_engine)):
    stats = aeng.engine.stats()
    return {
        "status": "ok",
        "model": stats["model"],
        "running": stats["running"],
        "waiting": stats["waiting"],
        "uptime_s": stats["uptime_s"],
    }
