
from fastapi import APIRouter, Request

router = APIRouter(tags=["Health"])


@router.get("/health")
def health_check(request: Request):
    model_service = getattr(request.app.state, "model_service", None)

    if model_service is None or not model_service.is_loaded:
        return {
            "status": "unhealthy",
            "model_loaded": False
        }

    return {
        "status": "healthy",
        "model_loaded": True
    }
