
from fastapi import APIRouter, Depends

from app.dependencies import get_model_service
from app.schemas import PredictionRequest
from app.services.model_service import ModelService
from app.services.prediction_service import predict_next_words

router = APIRouter(prefix="/predict", tags=["Prediction"])


@router.post("")
def predict(
    request: PredictionRequest,
    model_service: ModelService = Depends(get_model_service),
):
    return predict_next_words(
        model_service=model_service,
        input_text=request.input_text,
        top_k=request.top_k,
    )
