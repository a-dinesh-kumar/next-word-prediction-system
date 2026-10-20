
import numpy as np

from app.services.model_service import ModelService


def predict_next_words(
    model_service: ModelService,
    input_text: str,
    top_k: int = 5,
):
    encoded = model_service.encode_text(input_text)

    probabilities = model_service.predict_probabilities(
        encoded["token_ids"]
    )

    # Do not expose padding or the OOV marker as predictions.
    probabilities[0] = 0.0
    probabilities[model_service.oov_token_id] = 0.0

    candidate_ids = np.argsort(probabilities)[::-1]

    predictions = []

    for token_id in candidate_ids:
        token_id = int(token_id)
        word = model_service.id_to_word.get(token_id)

        if not word:
            continue

        predictions.append({
            "word": word,
            "probability": round(
                float(probabilities[token_id]), 6
            ),
        })

        if len(predictions) == top_k:
            break

    return {
        "input_text": input_text,
        "input_token_count": len(encoded["token_ids"]),
        "oov_count": encoded["oov_count"],
        "oov_ratio": round(encoded["oov_ratio"], 4),
        "predictions": predictions,
    }
