import json
from typing import Dict

import joblib
import nest_asyncio
import numpy as np
import torch
import torch.nn.functional as F
import uvicorn
from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pyngrok import ngrok
from transformers import AutoTokenizer, logging

from fine_tune import DialectIDModel
from preprocessing import preprocess

logging.set_verbosity_error()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class DialectRequest(BaseModel):
    text: str


class DialectResponse(BaseModel):

    confidence_scores: Dict[str, float]
    dialect: str
    confidence: float


class DLModel:
    def __init__(self):
        self._setup()

    def predict(self, text):
        text = preprocess(text, bert=True)
        encoded_input = self.tokenizer.encode_plus(
            text,
            padding="max_length",
            max_length=64,
            add_special_tokens=True,
            truncation="longest_first",
            return_tensors="pt",
        )
        with torch.no_grad():
            input_ids = encoded_input["input_ids"]
            attention_mask = encoded_input["attention_mask"]
            confidence_scores = F.softmax(
                self.model(input_ids=input_ids, attention_mask=attention_mask), dim=1
            )
        confidence, predicted_class = torch.max(confidence_scores, dim=1)
        prediction = self._get_str_label(predicted_class.item())
        confidence_scores = confidence_scores.flatten().numpy().tolist()
        return (
            dict(zip(list(self.label_map.keys()), confidence_scores)),
            confidence.item(),
            prediction,
        )

    def _get_str_label(self, predicted_class):
        for k, v in self.label_map.items():
            if predicted_class == v:
                return k
        return -1

    def _setup(self):
        self.tokenizer = AutoTokenizer.from_pretrained("UBC-NLP/MARBERT")
        self.model = DialectIDModel.load_from_checkpoint("marbert_checkpoint.ckpt")
        self.model.eval()
        with open("labels.json", "r") as f:
            self.label_map = json.load(f)


class ClassicalModel:
    def __init__(self):
        self._setup()

    def _setup(self):
        self.model = joblib.load("linear_svm.joblib")
        self.tfidf = joblib.load("tfidf.joblib")
        with open("labels.json", "r") as f:
            self.label_map = json.load(f)

    def _get_str_label(self, predicted_class):
        for k, v in self.label_map.items():
            if predicted_class == v:
                return k
        return -1

    def predict(self, text):
        text = preprocess(text)
        features = self.tfidf.transform([text])
        confidence_scores = self.model.decision_function(features)[0]
        confidence = np.max(confidence_scores)
        predicted_class = np.argmax(confidence_scores)
        prediction = self._get_str_label(predicted_class)
        return (
            dict(zip(list(self.label_map.keys()), confidence_scores)),
            confidence,
            prediction,
        )


marbert = DLModel()
svm = ClassicalModel()


def get_marbert():
    return marbert


def get_svm():
    return svm


@app.post("/predict", response_model=DialectResponse)
def predict(request: DialectRequest, model: DLModel = Depends(get_marbert)):
    confidence_scores, confidence, prediction = model.predict(request.text)
    return DialectResponse(
        dialect=prediction, confidence=confidence, confidence_scores=confidence_scores
    )


@app.post("/predict_classical", response_model=DialectResponse)
def predict(request: DialectRequest, model: ClassicalModel = Depends(get_svm)):
    confidence_scores, confidence, prediction = model.predict(request.text)
    return DialectResponse(
        dialect=prediction, confidence=confidence, confidence_scores=confidence_scores
    )


if __name__ == "__main__":
    ngrok_tunnel = ngrok.connect(8000)
    print("Public URL:", ngrok_tunnel.public_url)
    nest_asyncio.apply()
    uvicorn.run(app, port=8000)
