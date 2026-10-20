
from fastapi import APIRouter, Depends

from app.dependencies import get_model_service
from app.schemas import GenerationRequest
from app.services.generation_service import generate_text
from app.services.model_service import ModelService

router = APIRouter(prefix="/generate", tags=["Generation"])


@router.post("")
def generate(
    request: GenerationRequest,
    model_service: ModelService = Depends(get_model_service),
):
    return generate_text(
        model_service=model_service,
        seed_text=request.seed_text,
        num_words=request.num_words,
        strategy=request.strategy,
        temperature=request.temperature,
        top_k=request.top_k,
        repetition_penalty=request.repetition_penalty,
    )
