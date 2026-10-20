# NexText AI — Next Word Prediction System

An AI-powered text prediction application that predicts likely next words and generates text continuations using a trained **Embedding + LSTM neural language model**.

The project combines a model trained on news articles with a modular FastAPI backend and a responsive web interface. Users can enter a starting phrase, explore likely next words, or generate a longer continuation using different decoding strategies.

## Table of Contents

- [NexText AI — Next Word Prediction System](#nextext-ai--next-word-prediction-system)
  - [Table of Contents](#table-of-contents)
  - [Project Overview](#project-overview)
  - [Features](#features)
  - [How It Works](#how-it-works)
    - [Model Architecture](#model-architecture)
    - [Text Generation Strategies](#text-generation-strategies)
  - [Model and Evaluation](#model-and-evaluation)
    - [Held-out Test Results](#held-out-test-results)
  - [Technology Stack](#technology-stack)
  - [Project Structure](#project-structure)
  - [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [1. Clone the repository](#1-clone-the-repository)
    - [2. Create a virtual environment](#2-create-a-virtual-environment)
    - [3. Install dependencies](#3-install-dependencies)
    - [4. Verify model artifacts](#4-verify-model-artifacts)
  - [Running the Application](#running-the-application)
  - [API Documentation](#api-documentation)
    - [1. Health Check](#1-health-check)
    - [2. Predict Next Words](#2-predict-next-words)
    - [3. Generate Text](#3-generate-text)
  - [Running Tests](#running-tests)
  - [Known Limitations](#known-limitations)
  - [Future Improvements](#future-improvements)
  - [Author](#author)
    - [Areas of Interest:](#areas-of-interest)

## Project Overview

Next Word Prediction is a neural language modelling project designed to learn word patterns from a collection of news articles.

Given an input phrase such as:

`the team won the`

the model returns a ranked list of candidate next words with their predicted probabilities. It can also repeatedly predict subsequent words to generate a longer text continuation.

The application demonstrates the complete path from NLP model training and evaluation to API development, frontend integration, and automated testing.

## Features

- **Next-word prediction:** Returns the most likely next words for a user-provided phrase.
- **Text generation:** Generates a continuation using either Greedy or Sampled decoding.
- **Probability scores:** Displays model probabilities for candidate words.
- **OOV awareness:** Reports input tokens outside the model's known vocabulary.
- **Configurable generation:** Supports the number of words to generate and generation strategy.
- **Responsive frontend:** Provides a simple interface for predictions and generated text.
- **REST API:** Exposes prediction, generation, and health-check endpoints through FastAPI.
- **Input validation:** Validates request fields and rejects invalid inputs.
- **Automated tests:** Tests API behavior, response structures, validation, and frontend assets.
- **Modular architecture:** Keeps model loading, prediction logic, generation logic, API routes, and frontend files separate.

## How It Works

The application follows this flow:

1. The user enters a starting phrase in the frontend.
2. The frontend sends a request to the FastAPI backend.
3. The tokenizer converts the text into integer token IDs.
4. The input is padded or truncated to the model's expected sequence length.
5. The trained Embedding + LSTM model predicts a probability distribution over the vocabulary.
6. For next-word prediction, the API returns the highest-ranked eligible candidates.
7. For text generation, a decoding strategy selects a word, appends it to the input, and repeats the prediction process.
8. The frontend displays the results.

### Model Architecture

**Input text → Tokenization → Padding/Truncation → Embedding Layer → LSTM Layer → Dropout → Softmax Output → Word Prediction**

- **Tokenizer:** Converts words into integer IDs using the training vocabulary.
- **Embedding layer:** Learns dense vector representations of tokens.
- **LSTM layer:** Learns sequential patterns and relationships between tokens.
- **Dropout:** Helps reduce overfitting during training.
- **Dense softmax layer:** Produces probabilities for possible next tokens.

### Text Generation Strategies

**Greedy decoding**

Selects the highest-probability eligible next word at every generation step. It is deterministic for a fixed model and input, but may produce repetitive text.

**Sampled decoding**

Samples from a probability distribution rather than always choosing the top candidate. This can increase variety, although it may also produce less predictable or less coherent output.

The generation API supports temperature, top-k sampling, and a repetition penalty. Their practical effect depends on how these options are implemented in the generation service.

## Model and Evaluation

The model was trained on a news article dataset containing sports and business content.

| Configuration | Value |
|---|---|
| Model architecture | Embedding + LSTM |
| Embedding dimension | 128 |
| LSTM units | 128 |
| Dropout | 0.2 |
| Maximum vocabulary size | 20,000 |
| Maximum input sequence length | 20 tokens |
| Training epochs | 5 |
| Model parameters | Approximately 5.27 million |
| Saved model format | Keras `.keras` |

### Held-out Test Results

| Metric | Result |
|---|---:|
| Test loss | 5.3842 |
| Top-1 accuracy | 18.14% |
| Top-5 accuracy | 34.80% |
| Frequency-baseline top-1 accuracy | 5.58% |
| Frequency-baseline top-5 accuracy | 15.06% |

The LSTM outperformed the simple training-word-frequency baseline on both top-1 and top-5 accuracy.

These metrics measure next-token prediction, not the overall grammatical quality or factual correctness of generated paragraphs. The results are specific to the dataset, preprocessing, and trained model used for this project.

## Technology Stack

**Machine Learning and NLP**
- Python
- TensorFlow / Keras
- NumPy
- Tokenization and sequence preprocessing

**Backend**
- FastAPI
- Uvicorn
- Pydantic

**Frontend**
- HTML
- CSS
- JavaScript

**Testing**
- pytest
- FastAPI TestClient
- HTTPX

**Development and Deployment**
- Git
- GitHub
- Render (planned deployment target)

## Project Structure

```text
next_word_prediction_system/
│
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── config.py
│   ├── schemas.py
│   ├── dependencies.py
│   │
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── frontend.py
│   │   ├── health.py
│   │   ├── predictions.py
│   │   └── generation.py
│   │
│   └── services/
│       ├── __init__.py
│       ├── model_service.py
│       ├── prediction_service.py
│       └── generation_service.py
│
├── frontend/
│   ├── index.html
│   ├── style.css
│   ├── app.js
│   └── icons/
│       ├── github.png
│       └── linkedin.png
│
├── next_word_artifacts/
│   ├── best_next_word_lstm.keras
│   ├── text_tokenizer.pkl
│   └── model_metadata.json
│
├── tests/
│   ├── __init__.py
│   └── test_api.py
│
├── requirements.txt
├── .gitignore
└── README.md
```

*The tree reflects the current application layout. Keep it aligned with your actual repository if any files differ.*

## Getting Started

### Prerequisites

- Python installed and compatible with the project's pinned dependencies.
- Git.
- The trained model, tokenizer, and metadata files in `next_word_artifacts/`.

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/YOUR_REPOSITORY_NAME.git
cd YOUR_REPOSITORY_NAME
```

Replace the repository URL with the actual GitHub repository URL.

### 2. Create a virtual environment

**Windows:**

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Verify model artifacts

Make sure the following files exist:

```text
next_word_artifacts/best_next_word_lstm.keras
next_word_artifacts/text_tokenizer.pkl
next_word_artifacts/model_metadata.json
```

The application requires these files to load the trained model. The original training dataset and notebook are not required for inference if the artifacts are already available.

## Running the Application

Start the development server from the repository root:

```bash
python -m uvicorn app.main:app --reload
```

Open the following URLs:

| Resource | URL |
|---|---|
| Web application | http://127.0.0.1:8000/ |
| Health check | http://127.0.0.1:8000/health |
| Interactive API documentation | http://127.0.0.1:8000/docs |

The web interface and API are served by the same FastAPI application.

## API Documentation

### 1. Health Check

**Endpoint:** `GET /health`

Checks whether the model service is loaded.

Example response:

```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### 2. Predict Next Words

**Endpoint:** `POST /predict`

Example request:

```json
{
  "input_text": "the team won the",
  "top_k": 5
}
```

The response contains the input text, vocabulary-related information, and ranked next-word candidates with their probabilities.

Example response shape:

```json
{
  "input_text": "the team won the",
  "predictions": [
    {
      "word": "world",
      "probability": 0.0664
    }
  ]
}
```

This is an illustrative response shape, not a complete live response. Actual output depends on the loaded model and may contain additional fields.

### 3. Generate Text

**Endpoint:** `POST /generate`

Example request:

```json
{
  "seed_text": "the team won the",
  "num_words": 20,
  "strategy": "sampled",
  "temperature": 0.8,
  "top_k": 10,
  "repetition_penalty": 0.7
}
```

Supported strategies:

- `greedy`
- `sampled`

The API returns the generated continuation, generation details, and available input-token statistics.

For exact schemas and response fields, use the interactive documentation at `/docs`.

## Running Tests

The project uses pytest and FastAPI's TestClient to verify the application.

Install the testing dependencies if they are not already installed:

```bash
python -m pip install pytest httpx
```

Run the test suite from the project root:

```bash
python -m pytest tests/test_api.py -v
```

The current test suite covers:

- Health endpoint behavior.
- Homepage and static asset delivery.
- Successful next-word predictions.
- Top-k prediction limits.
- Successful text generation using both strategies.
- Invalid request validation for prediction and generation.

The test suite has passed in the local development environment.

## Known Limitations

- The model is trained on a relatively small, domain-specific news dataset.
- Generated text may repeat phrases or lack grammatical and semantic coherence.
- Next-token accuracy does not measure the factual correctness of a generated continuation.
- Words outside the known vocabulary may reduce prediction reliability.
- A word appearing in the vocabulary does not mean the model understands its meaning.
- The model is not a general-purpose conversational AI or a large language model.
- Performance on unrelated topics may be poor.
- Model quality is limited by the dataset size, vocabulary, training configuration, and sequence length.

## Future Improvements

- Deploy the application on Render.
- Improve text preprocessing and tokenization behavior.
- Investigate repeated phrases and token-spacing artifacts.
- Experiment with additional training data and model configurations.
- Compare generation strategies and evaluate text quality.
- Add more edge-case and service-level tests.
- Improve model monitoring, error handling, and deployment reliability.

## Author

**Dinesh Kumar Alagarsamy**  
*Software Quality Engineer / SDET*
-   GitHub: https://github.com/a-dinesh-kumar
-   LinkedIn: https://www.linkedin.com/in/dinesh-kumar-alagarsamy

### Areas of Interest:
* Quality Engineering
* Test Automation
* API Testing
* Artificial Intelligence
* Machine Learning
* AI-powered Testing
* Software Architecture
* GenAI & Agentic AI

---

*Built as an NLP learning and portfolio project to explore neural next-word prediction, recurrent language models, text generation, API development, and deployment.*
