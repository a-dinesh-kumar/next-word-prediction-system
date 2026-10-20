
const inputText = document.getElementById("inputText");
const charCount = document.getElementById("charCount");

const predictButton = document.getElementById("predictButton");
const generateButton = document.getElementById("generateButton");
const clearButton = document.getElementById("clearButton");

const generationOptions = document.getElementById("generationOptions");
const numWords = document.getElementById("numWords");
const strategy = document.getElementById("strategy");

const resultTitle = document.getElementById("resultTitle");
const resultBadge = document.getElementById("resultBadge");
const results = document.getElementById("results");
const message = document.getElementById("message");
const oovWarning = document.getElementById("oovWarning");

const actionButtons = [predictButton, generateButton];

inputText.addEventListener("input", () => {
    charCount.textContent = `${inputText.value.length} / 5000`;
});

predictButton.addEventListener("click", predictNextWords);
generateButton.addEventListener("click", generateText);
clearButton.addEventListener("click", clearInterface);

function showMessage(text, type = "info") {
    message.textContent = text;
    message.className = `message ${type}`;
    message.hidden = false;
}

function hideMessage() {
    message.hidden = true;
    message.textContent = "";
}

function setLoading(isLoading, action = "") {
    actionButtons.forEach((button) => {
        button.disabled = isLoading;
    });

    if (isLoading) {
        resultBadge.textContent = "Working...";
        showMessage(
            action === "predict"
                ? "Finding the most likely next words..."
                : "Generating your text continuation..."
        );
    } else {
        resultBadge.textContent = "Complete";
        hideMessage();
    }
}

function resetResultArea(title, badge = "Ready") {
    resultTitle.textContent = title;
    resultBadge.textContent = badge;
    results.replaceChildren();
    oovWarning.hidden = true;
    oovWarning.textContent = "";
    hideMessage();
}

function showEmptyState() {
    const wrapper = document.createElement("div");
    wrapper.className = "empty-state";

    const icon = document.createElement("div");
    icon.className = "empty-icon";
    icon.textContent = "✦";

    const heading = document.createElement("p");
    heading.textContent = "Start with a phrase above.";

    const description = document.createElement("span");
    description.textContent =
        "Your predictions or generated text will appear here.";

    wrapper.append(icon, heading, description);
    results.replaceChildren(wrapper);
}

async function callApi(endpoint, payload) {
    const response = await fetch(endpoint, {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify(payload)
    });

    const responseData = await response.json().catch(() => ({}));

    if (!response.ok) {
        let errorMessage = responseData.detail;

        if (Array.isArray(errorMessage)) {
            errorMessage = errorMessage
                .map((item) => item.msg || "Invalid input")
                .join("; ");
        }

        throw new Error(
            typeof errorMessage === "string"
                ? errorMessage
                : `Request failed with HTTP ${response.status}.`
        );
    }

    return responseData;
}

function showOovWarning(data) {
    const ratio = Number(data.oov_ratio ?? 0);
    const count = Number(data.oov_count ?? 0);
    const tokenCount = Number(data.token_count ?? 0);

    if (count > 0 || ratio > 0) {
        const percentage = (ratio * 100).toFixed(1);

        oovWarning.textContent =
            `${count} of ${tokenCount} input tokens are outside the model's ` +
            `known vocabulary (${percentage}%). Predictions may be less reliable.`;

        oovWarning.hidden = false;
    } else {
        oovWarning.hidden = true;
    }
}

function formatProbability(value) {
    const probability = Number(value);

    if (!Number.isFinite(probability)) {
        return "—";
    }

    return `${(probability * 100).toFixed(2)}%`;
}

async function predictNextWords() {
    const text = inputText.value.trim();

    if (!text) {
        showMessage("Please enter some starting text first.", "error");
        inputText.focus();
        return;
    }

    generationOptions.hidden = true;
    resetResultArea("Next-word predictions");
    setLoading(true, "predict");

    try {
        const data = await callApi("/predict", {
            input_text: text,
            top_k: 5
        });

        showOovWarning(data);
        displayPredictions(data);
        resultTitle.textContent = "Most likely next words";
        resultBadge.textContent = "Complete";
    } catch (error) {
        showMessage(error.message || "Unable to predict next words.", "error");
        resultBadge.textContent = "Error";
    } finally {
        actionButtons.forEach((button) => {
            button.disabled = false;
        });
    }
}

function displayPredictions(data) {
    const predictions = data.predictions || [];

    if (!predictions.length) {
        showMessage("The model did not return any predictions.", "error");
        return;
    }

    const list = document.createElement("div");
    list.className = "prediction-list";

    const maxProbability = Math.max(
        ...predictions.map((item) => Number(item.probability) || 0),
        0.000001
    );

    predictions.forEach((item) => {
        const probability = Number(item.probability) || 0;

        const row = document.createElement("div");
        row.className = "prediction-item";

        const word = document.createElement("span");
        word.className = "prediction-word";
        word.textContent = item.word;

        const track = document.createElement("div");
        track.className = "bar-track";

        const fill = document.createElement("div");
        fill.className = "bar-fill";
        fill.style.width = `${Math.min(
            (probability / maxProbability) * 100,
            100
        )}%`;

        track.appendChild(fill);

        const percent = document.createElement("span");
        percent.className = "probability";
        percent.textContent = formatProbability(probability);

        row.append(word, track, percent);
        list.appendChild(row);
    });

    results.replaceChildren(list);

    const note = document.createElement("p");
    note.className = "result-meta";
    note.textContent =
        "Probabilities show the model's relative confidence for each candidate. " +
        "They are not a guarantee that a word is correct.";

    results.appendChild(note);
}

async function generateText() {
    const text = inputText.value.trim();

    if (!text) {
        showMessage("Please enter a starting phrase first.", "error");
        inputText.focus();
        return;
    }

    generationOptions.hidden = false;
    resetResultArea("Generated continuation");
    setLoading(true, "generate");

    try {
        const data = await callApi("/generate", {
            seed_text: text,
            num_words: Number(numWords.value),
            strategy: strategy.value,
            temperature: 0.8,
            top_k: 10,
            repetition_penalty: 0.7
        });

        showOovWarning(data);
        displayGeneratedText(data);
        resultTitle.textContent = "Generated text";
        resultBadge.textContent = "Complete";
    } catch (error) {
        showMessage(error.message || "Unable to generate text.", "error");
        resultBadge.textContent = "Error";
    } finally {
        actionButtons.forEach((button) => {
            button.disabled = false;
        });
    }
}


function displayGeneratedText(data) {
    const output = document.createElement("div");
    output.className = "generated-text";
    output.textContent = data.generated_text || "No text was generated.";

    const meta = document.createElement("p");
    meta.className = "result-meta";

    const generatedWords = data.generated_words;
    let wordCount = data.actual_generated_words;

    // Some backend responses return generated_words as a list.
    if (Array.isArray(generatedWords)) {
        wordCount = generatedWords.length;
    } else if (wordCount == null && typeof generatedWords === "string") {
        wordCount = generatedWords.trim()
            ? generatedWords.trim().split(/\s+/).length
            : 0;
    }

    meta.textContent =
        `Strategy: ${data.strategy || strategy.value} · ` +
        `Generated words: ${wordCount ?? "—"}`;

    results.replaceChildren(output, meta);
}

function clearInterface() {
    inputText.value = "";
    charCount.textContent = "0 / 5000";
    generationOptions.hidden = true;

    resetResultArea("Your results will appear here", "Ready");
    showEmptyState();

    inputText.focus();
}

showEmptyState();
