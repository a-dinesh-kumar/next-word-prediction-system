
from typing import Literal

from pydantic import BaseModel, Field

from app.config import (
    DEFAULT_GENERATION_WORDS,
    DEFAULT_TOP_K,
    MAX_GENERATION_WORDS,
    MAX_INPUT_LENGTH,
    MAX_TOP_K,
)


class PredictionRequest(BaseModel):
    input_text: str = Field(
        min_length=1,
        max_length=MAX_INPUT_LENGTH,
        description="Text used to predict the next word.",
    )
    top_k: int = Field(
        default=DEFAULT_TOP_K,
        ge=1,
        le=MAX_TOP_K,
    )


class GenerationRequest(BaseModel):
    seed_text: str = Field(
        min_length=1,
        max_length=MAX_INPUT_LENGTH,
    )
    num_words: int = Field(
        default=DEFAULT_GENERATION_WORDS,
        ge=1,
        le=MAX_GENERATION_WORDS,
    )
    strategy: Literal["greedy", "sampled"] = "sampled"
    temperature: float = Field(default=0.8, gt=0, le=2.0)
    top_k: int = Field(default=10, ge=1, le=MAX_TOP_K)
    repetition_penalty: float = Field(default=0.7, gt=0, le=1.0)
