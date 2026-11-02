import time
from contextlib import asynccontextmanager

from fastapi import FastAPI

from miniserve.api.errors import install_error_handlers
from miniserve.api.routers import chat, completions, health, metrics, models
from miniserve.api.settings import Settings
from miniserve.config import EngineConfig
from miniserve.engine.engine import AsyncEngine, Engine


def config_from_settings(settings: Settings) -> EngineConfig:
    return EngineConfig(
        model=settings.model,
        device=settings.device,
        max_model_len=settings.max_model_len,
        attention_backend=settings.attention_backend,
        block_size=settings.block_size,
        max_num_seqs=settings.max_num_seqs,
        max_num_batched_tokens=settings.max_num_batched_tokens,
        gpu_memory_utilization=settings.gpu_memory_utilization,
        enable_prefix_caching=settings.enable_prefix_caching,
        use_cuda_graphs=settings.use_cuda_graphs,
    )


def create_app(cfg: EngineConfig | None = None) -> FastAPI:
    if cfg is None:
        cfg = config_from_settings(Settings.from_env())

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        engine = AsyncEngine(Engine(cfg)).start()
        app.state.engine = engine
        app.state.cfg = cfg
        app.state.started_at = time.time()
        try:
            yield
        finally:
            engine.stop()

    app = FastAPI(
        title="miniserve",
        version="0.1.0",
        description="A tiny vLLM-style inference server",
        lifespan=lifespan,
    )

    install_error_handlers(app)

    app.include_router(health.router, tags=["health"])
    app.include_router(models.router, tags=["models"])
    app.include_router(chat.router, tags=["chat"])
    app.include_router(completions.router, tags=["completions"])
    app.include_router(metrics.router, tags=["metrics"])

    @app.get("/")
    def root():
        return {"service": "miniserve", "model": cfg.model, "docs": "/docs"}

    return app
