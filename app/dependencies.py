
from fastapi import HTTPException, Request

from app.services.model_service import ModelService


def get_model_service(request: Request) -> ModelService:
    model_service = getattr(
        request.app.state,
        "model_service",
        None,
    )

    if model_service is None or not model_service.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="The model is not available.",
        )

    return model_service
