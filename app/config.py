
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = BASE_DIR / "next_word_artifacts"
FRONTEND_DIR = BASE_DIR / "frontend"
MODEL_PATH = ARTIFACT_DIR / "best_next_word_lstm.keras"
TOKENIZER_PATH = ARTIFACT_DIR / "text_tokenizer.pkl"
METADATA_PATH = ARTIFACT_DIR / "model_metadata.json"

DEFAULT_TOP_K = 5
MAX_TOP_K = 50
DEFAULT_GENERATION_WORDS = 20
MAX_GENERATION_WORDS = 100
MAX_INPUT_LENGTH = 5000
