
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.config import (
    FRONTEND_DIR,
    MODEL_PATH,
    TOKENIZER_PATH,
    METADATA_PATH,
)
from app.services.model_service import ModelService
from app.routers import (
    frontend,
    health,
    predictions,
    generation,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_service = ModelService(
        model_path=MODEL_PATH,
        tokenizer_path=TOKENIZER_PATH,
        metadata_path=METADATA_PATH,
    )

    app.state.model_service = model_service.load()

    try:
        yield
    finally:
        app.state.model_service = None


app = FastAPI(
    title="NexText AI - Next Word Prediction System",
    description=(
        "Predict the next word or generate a text continuation "
        "using an Embedding-LSTM language model."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# API routes
app.include_router(health.router)
app.include_router(predictions.router)
app.include_router(generation.router)

# Frontend route
app.include_router(frontend.router)

# Serve CSS and JavaScript separately from the API logic
app.mount(
    "/static",
    StaticFiles(directory=str(FRONTEND_DIR)),
    name="static",
)
