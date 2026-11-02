import time

from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/v1/models")
def list_models(request: Request):
    cfg = request.app.state.cfg
    return {
        "object": "list",
        "data": [
            {
                "id": cfg.model,
                "object": "model",
                "created": int(getattr(request.app.state, "started_at", time.time())),
                "owned_by": "miniserve",
            }
        ],
    }


@router.get("/v1/models/{model}")
def get_model(model: str, request: Request):
    return {
        "id": model,
        "object": "model",
        "created": int(getattr(request.app.state, "started_at", time.time())),
        "owned_by": "miniserve",
    }
