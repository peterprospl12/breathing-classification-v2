# Breathing Phase Classification Project

This repository contains a comprehensive solution for real-time breathing phase classification using machine learning models. The system can detect three breath phases: inhale, exhale, and silence from audio recordings of breathing.

## 🌟 Features

- **Multiple ML Models**: Implementation of different architectures (LSTM, Transformer, Spectrum-based models)
- **Real-time Classification**: Process audio input and classify breathing phases in real-time
- **Data Generation Tools**: Scripts to create synthetic breathing sequences for training
- **API Server**: FastAPI-based server for model serving
- **Mobile Application**: Flutter app for user-friendly visualization and monitoring
- **Tensometer Data Integration**: Support for synchronizing breathing data with tensometer readings

## 🧩 Project Structure

The project is organized into the following main components:

```
breathing-classification-v2/
├── API/                    # FastAPI server for model serving
├── breathing_model/
│   ├── model/              # ML model implementations
│   │   ├── lstm_model/     # LSTM-based model
│   │   ├── spectrum_model/ # Spectrum-based model
│   │   └── transformer_model/ # Transformer-based model
│   ├── generators/         # Data generation utilities
│   └── scripts/            # Utility scripts for data processing
└── app/                    # Flutter mobile application
```

## 📋 Requirements

- Python 3.7+
- PyTorch
- torchaudio
- Flutter SDK (for app)
- Additional dependencies in requirements.txt

## 🚀 Getting Started

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/breathing-classification-v2.git
   cd breathing-classification-v2
   ```

2. Set up a Python virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 🔍 Models

### LSTM Model

The LSTM model processes audio features (MFCC) to classify breathing phases:

```python
# Example usage
from breathing_model.archive.lstm.realtime import RealTimeAudioClassifier

classifier = RealTimeAudioClassifier("model_path.pth")
prediction = classifier.predict(audio_data)
```

### Transformer Model

The Transformer-based model provides enhanced sequence modeling capabilities:

```python
# Example usage
from breathing_model.archive.transformer_model.realtime import RealTimeAudioClassifier, PredictionModes

classifier = RealTimeAudioClassifier("model_path.pth", PredictionModes.LOCAL)
prediction = classifier.predict(audio_data)
```

## 🖥️ API Server

The project includes a FastAPI server for model serving:

1. Start the server:
   ```bash
   cd API
   python server.py
   ```

2. API endpoints:
   - `/predict/audio`: For raw audio classification
   - `/predict/mel`: For pre-computed mel-spectrogram classification

## 📱 Mobile Application

A Flutter application is included for user-friendly breathing monitoring:

- Real-time visualization of breathing patterns
- Breath phase classification display
- Inhale/Exhale counting

See the [app directory](app/README.md) for more details on installation and usage.

## 🔧 Data Generation

The repository includes tools for generating synthetic breathing sequences for training:

```bash
# Create training sequences
python breathing_model/scripts/sequence_creator_2.py
```

## 📊 Training

To train a model:

```bash
# Train transformer model
python breathing_model/model/transformer_model/transformer_model.py

# Train LSTM model
python breathing_model/model/lstm/full_train_in_one_script.py
```

## 📝 License

[Your license information here]

## 👥 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.