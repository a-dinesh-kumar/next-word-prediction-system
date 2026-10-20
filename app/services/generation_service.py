
import numpy as np

from app.services.model_service import ModelService


def generate_text(
    model_service: ModelService,
    seed_text: str,
    num_words: int = 20,
    strategy: str = "sampled",
    temperature: float = 0.8,
    top_k: int = 10,
    repetition_penalty: float = 0.7,
):
    encoded = model_service.encode_text(seed_text)
    original_token_ids = list(encoded["token_ids"])
    generated_token_ids = list(original_token_ids)

    for _ in range(num_words):
        probabilities = model_service.predict_probabilities(
            generated_token_ids
        )

        # Prevent special tokens from appearing in generated text.
        probabilities[model_service.oov_token_id] = 0.0

        # Reduce repetition of words in the recent context.
        for token_id in set(generated_token_ids[-5:]):
            if token_id < len(probabilities):
                probabilities[token_id] *= repetition_penalty

        if probabilities.sum() <= 0:
            break

        if strategy == "greedy":
            next_token_id = int(np.argmax(probabilities))

        else:
            candidate_ids = np.argsort(probabilities)[-top_k:]
            candidate_probabilities = probabilities[candidate_ids]

            if candidate_probabilities.sum() <= 0:
                break

            # Temperature changes the relative likelihood of candidates.
            log_probabilities = np.log(
                np.maximum(candidate_probabilities, 1e-12)
            )

            adjusted_probabilities = np.exp(
                (
                    log_probabilities
                    - np.max(log_probabilities)
                ) / temperature
            )

            adjusted_probabilities /= adjusted_probabilities.sum()

            next_token_id = int(
                np.random.choice(
                    candidate_ids,
                    p=adjusted_probabilities,
                )
            )

        if probabilities[next_token_id] <= 0:
            break

        generated_token_ids.append(next_token_id)

    new_token_ids = generated_token_ids[len(original_token_ids):]

    generated_words = [
        model_service.id_to_word[token_id]
        for token_id in new_token_ids
        if token_id in model_service.id_to_word
    ]

    return {
        "seed_text": seed_text,
        "generated_text": (
            seed_text.strip() + " " + " ".join(generated_words)
        ).strip(),
        "generated_words": generated_words,
        "requested_word_count": num_words,
        "actual_generated_word_count": len(generated_words),
        "strategy": strategy,
        "input_token_count": len(original_token_ids),
        "oov_count": encoded["oov_count"],
        "oov_ratio": round(encoded["oov_ratio"], 4),
    }
