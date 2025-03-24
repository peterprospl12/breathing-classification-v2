from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import numpy as np
import uvicorn
import base64
import io
from typing import List, Optional

from server_dependencies.server_dependecies import (
    RealTimeAudioClassifier,
    MelTransformer,
    RATE,
    device
)
MODEL_PATH = 'server_dependencies/best_breath_seq_transformer_model_CURR_BEST.pth'
# Tworzymy aplikację FastAPI
app = FastAPI(title="Breathing classification API")

# Dodajemy middleware CORS dla obsługi żądań z różnych źródeł
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Dla produkcji dostosuj do odpowiednich źródeł
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Zmienna globalna dla klasyfikatora
classifier = None


# Model wejściowy dla surowych danych audio
class AudioData(BaseModel):
    """Model dla danych audio w formacie base64"""
    audio_data: str  # Dane audio w formacie base64
    sample_rate: int = RATE


# Model wejściowy dla danych mel spectrogramu
class MelData(BaseModel):
    """Model dla pre-obliczonego mel spectrogramu"""
    mel_data: List


# Model odpowiedzi z prognozą
class PredictionResponse(BaseModel):
    """Model odpowiedzi dla prognoz"""
    prediction: int
    prediction_name: str
    confidence: Optional[float] = None


# Funkcja uruchamiana podczas startu aplikacji
@app.on_event("startup")
async def startup_event():
    global classifier
    classifier = RealTimeAudioClassifier(
        model_path=MODEL_PATH # Używamy pre-obliczonego mel spectrogramu
    )
    print(f"Model załadowany z {MODEL_PATH}")


# Endpoint główny
@app.get("/")
async def read_root():
    return {"status": "Breath Classification API is running"}


# Endpoint do klasyfikacji surowych danych audio
@app.post("/predict/audio")
async def predict_audio(data: AudioData):
    """Klasyfikacja fazy oddychania na podstawie surowych danych audio"""
    if not classifier:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Konwersja audio z base64 na surowe dane
    try:
        audio_bytes = base64.b64decode(data.audio_data)
        audio_array = np.frombuffer(audio_bytes, dtype=np.float32)
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid audio data format")

    # Klasyfikacja
    prediction, prediction_name, confidence = classifier.predict(audio_array, data.sample_rate)

    return PredictionResponse(
        prediction=prediction,
        prediction_name=prediction_name,
        confidence=confidence
    )


# Endpoint do klasyfikacji pre-obliczonego mel spectrogramu
@app.post("/predict/mel")
async def predict_mel(data: MelData):
    """Klasyfikacja fazy oddychania na podstawie pre-obliczonego mel spectrogramu"""
    if not classifier:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Convert the list back to a tensor
        mel_tensor = torch.tensor(data.mel_data).to(device)

        # Ensure it has the correct shape for the model input
        if len(mel_tensor.shape) != 4:
            raise HTTPException(status_code=400, detail="Invalid mel spectrogram shape")

        # Klasyfikacja
        prediction, prediction_name, confidence = classifier.predict(mel_tensor, dont_calc_mel=True)

        return PredictionResponse(
            prediction=prediction,
            prediction_name=prediction_name,
            confidence=confidence
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing mel data: {str(e)}")


# Uruchomienie aplikacji FastAPI z Uvicorn
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
