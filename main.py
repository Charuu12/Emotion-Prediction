from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from pydantic import BaseModel, Field

from pathlib import Path
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

import tensorflow as tf
import numpy as np
import re

"""
1. We are going to make some constants like:
A. Model Path (BiGRU)
B. Tokenizer Path
C. Max Sequence Length
D. Emotion Labels
E. Emotion emojis
"""

# A. BERT Model Path
# model_path = "Artifacts/Bert"
BASE_DIR = Path(__file__).resolve().parent
model_path = BASE_DIR / "Artifacts" / "Bert"

# B. Maximum Sequence Length
max_sequence_length = 128

# C. Emotion Labels
emotion_labels = ["sadness", "joy", "love", "anger", "fear", "surprise"]

# D. Emotion Emojis
EMOTION_EMOJIS = {
    "sadness": "😢",
    "joy": "😄",
    "love": "❤️",
    "anger": "😠",
    "fear": "😨",
    "surprise": "😲",
}

"""
2. Preprocess the upcoming text
Cleans raw text so it matches the format used while training.
A. Convert the text to lowercase. -done
B. Remove apostrophes (e.g can't -> cant). -done
C. Remove Special Characters and Punctuation. -done
D. Remove extra spaces -done
"""
def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"'", "", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text

"""
3. Request and Response Schemas
A. Text Input -> Input schema: the text sent by user. 
B. Prediciton Response -> Output schema: the emotion to predict. 
C. Health Response (Server health check)
"""
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


"""
4. Model Loading and LifeSpan Management
Load the model and toknizer once the server starts up.
"""
dl_model = {}

@asynccontextmanager 
async def lifespan(app: FastAPI):

    print("Loading BERT model and tokenizer...")

    # Load BERT tokenizer
    dl_model["Tokenizer"] = AutoTokenizer.from_pretrained(
        str(model_path)
    )

    # Load trained BERT model
    dl_model["BERT"] = TFAutoModelForSequenceClassification.from_pretrained(
        str(model_path)
    )

    print("BERT model and tokenizer loaded successfully!")

    yield #Pause, model is laoded and server is running and at this point model wait karega for request

    dl_model.clear()


#5. Mount the static files to the FastAPI app
app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount('/static', StaticFiles(directory="static"), name="static")

# 6. API Endpoints.
@app.get('/', include_in_schema=False)
def server_ui():
    return FileResponse('static/index.html')


@app.get('/health', response_model=HealthResponse)
def health_check():
    return HealthResponse(
        status="Server is running",
        model_loaded=bool(dl_model)
    )

@app.post('/predict', response_model=PredictionResponse)
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

    # 2. Tokenize input using BERT tokenizer
    inputs = tokenizer_model(
        cleaned_text,
        return_tensors="tf",
        padding="max_length",
        truncation=True,
        max_length=max_sequence_length
    )

    # 3. BERT prediction
    outputs = BERT_model(
        inputs,
        training=False
    )

    # 4. Convert logits into probabilities
    probabilities = tf.nn.softmax(
        outputs.logits,
        axis=-1
    ).numpy()[0]

    # 5. Find the emotion with highest probability
    top_emotion_index = int(np.argmax(probabilities))

    all_probabilites = {
        label: float(prob)
        for prob, label in zip(probabilities, emotion_labels)
    }

    # 6. Return response
    return PredictionResponse(
        text=text_input.text,
        predicted_emotion=emotion_labels[top_emotion_index],
        confidence=float(probabilities[top_emotion_index]),
        all_probabilites=all_probabilites
    )
