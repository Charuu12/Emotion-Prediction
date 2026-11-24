from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from pathlib import Path
from transformers import AutoTokenizer

import onnxruntime as ort
import numpy as np
import re

# 1. Configuration

BASE_DIR = Path(__file__).resolve().parent

model_path = BASE_DIR / "Artifacts" / "Bert_ONNX" / "bert_model_int8.onnx"
tokenizer_path = BASE_DIR / "Artifacts" / "Bert"

max_sequence_length = 128

emotion_labels = ["sadness", "joy", "love", "anger", "fear", "surprise"]

EMOTION_EMOJIS = {
    "sadness": "😢",
    "joy": "😄",
    "love": "❤️",
    "anger": "😠",
    "fear": "😨",
    "surprise": "😲",
}

# 2. Text preprocessing

def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"'", "", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# 3. Request and response schemas

class TextInput(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The sentence to analyze",
        json_schema_extra={"example": "I feel so happy and excited"}
    )


class PredictionResponse(BaseModel):
    text: str
    predicted_emotion: str
    confidence: float
    all_probabilites: dict[str, float]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


# 4. Model loading

dl_model = {}


@asynccontextmanager
async def lifespan(app: FastAPI):

    print("Loading ONNX INT8 model and tokenizer...")

    dl_model["Tokenizer"] = AutoTokenizer.from_pretrained(
        str(tokenizer_path)
    )

    dl_model["BERT"] = ort.InferenceSession(
        str(model_path),
        providers=["CPUExecutionProvider"]
    )

    print("ONNX INT8 model and tokenizer loaded successfully!")

    yield

    dl_model.clear()


# 5. FastAPI application

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount(
    "/static",
    StaticFiles(directory=str(BASE_DIR / "static")),
    name="static"
)


# 6. API endpoints

@app.get("/", include_in_schema=False)
def server_ui():
    return FileResponse(str(BASE_DIR / "static" / "index.html"))


@app.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(
        status="Server is running",
        model_loaded=bool(dl_model)
    )


@app.post("/predict", response_model=PredictionResponse)
def predict_emotion(text_input: TextInput):

    BERT_model = dl_model.get("BERT")
    tokenizer_model = dl_model.get("Tokenizer")

    if BERT_model is None or tokenizer_model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded yet."
        )

    # 1. Preprocess input
    cleaned_text = preprocess_text(text_input.text)

    # 2. Tokenize input
    inputs = tokenizer_model(
        cleaned_text,
        return_tensors="np",
        padding="max_length",
        truncation=True,
        max_length=max_sequence_length
    )

    # 3. Prepare ONNX inputs
    onnx_inputs = {
        "input_ids": inputs["input_ids"].astype(np.int32),
        "attention_mask": inputs["attention_mask"].astype(np.int32),
        "token_type_ids": inputs["token_type_ids"].astype(np.int32)
    }

    # 4. ONNX prediction
    outputs = BERT_model.run(
        ["logits"],
        onnx_inputs
    )

    logits = outputs[0]

    # 5. Convert logits into probabilities
    logits = logits - np.max(logits, axis=-1, keepdims=True)

    exp_values = np.exp(logits)

    probabilities = (
        exp_values / np.sum(exp_values, axis=-1, keepdims=True)
    )[0]

    # 6. Find highest probability
    top_emotion_index = int(np.argmax(probabilities))

    all_probabilites = {
        label: float(prob)
        for prob, label in zip(probabilities, emotion_labels)
    }

    # 7. Return response
    return PredictionResponse(
        text=text_input.text,
        predicted_emotion=emotion_labels[top_emotion_index],
        confidence=float(probabilities[top_emotion_index]),
        all_probabilites=all_probabilites
    )