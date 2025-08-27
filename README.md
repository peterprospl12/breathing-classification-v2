# Breathing Phase Classification Project

This repository contains a comprehensive solution for real-time breathing phase classification using machine learning models. The system can detect three breath phases: **inhale**, **exhale**, and **silence** from audio recordings, with additional support for tensometer-based data collection and analysis.

## 🌟 Features

- **Multiple ML Architectures**: Transformer, LSTM, and Spectrum-based models with different performance characteristics
- **Real-time Classification**: Live audio processing and breath phase detection
- **Tensometer Integration**: Synchronized breathing data collection using tensometer sensors via Bluetooth
- **Cross-platform Mobile App**: Flutter application for Windows, Web, Android, and iOS
- **ONNX Model Export**: Convert trained models to ONNX format for deployment
- **Advanced Data Generation**: Tools for creating labeled training data from tensometer readings
- **Model Evaluation Tools**: Comprehensive evaluation and visualization utilities
- **Multi-language Support**: Interface supports both English and Polish

## 🧩 Project Structure

```
breathing-classification-v2/
├── breathing_model/           # Core ML package
│   ├── model/                # Current model implementations
│   │   ├── transformer/      # Latest transformer architecture
│   │   ├── trained_models/   # Pre-trained model checkpoints
│   │   ├── silence_detector/ # Silence detection utilities
│   │   └── invalid_data_filter/ # Data quality filtering
│   ├── data/                 # Data processing and generation
│   │   ├── generators/       # Training data generation tools
│   │   ├── train/           # Training datasets
│   │   └── eval/            # Evaluation datasets
│   ├── archive/             # Legacy model implementations
│   │   ├── lstm/            # LSTM-based models
│   │   ├── spectrum/        # Frequency-domain models
│   │   └── transformer_model/ # Previous transformer implementation
│   └── requirements.txt     # Python dependencies
├── app/                     # Flutter mobile application
│   ├── lib/                # Dart source code
│   ├── assets/             # App resources and models
│   ├── plugins/            # Custom platform plugins
│   └── README.md           # App-specific documentation
├── docs/                   # Project documentation
└── LICENSE                 # MIT License
```

## 📋 Requirements

### Python Environment
- **Python 3.7+** (recommended: 3.9+)
- **PyTorch 2.5.1+** with torchaudio support
- **NumPy 2.0.2+** for numerical computations
- **librosa 0.10.0+** for audio processing
- **scikit-learn 1.6.0+** for ML utilities
- **ONNX Runtime 1.22.1+** for model deployment

### Flutter Environment (for mobile app)
- **Flutter SDK 3.7.0+**
- **Dart SDK** (included with Flutter)
- Platform-specific SDKs for target deployment

### Hardware Requirements
- **Microphone** for real-time audio capture
- **Tensometer device** (optional, for data collection)
- **Bluetooth support** (for tensometer connectivity)

## 🚀 Getting Started

### Python Environment Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/peterprospl12/breathing-classification-v2.git
   cd breathing-classification-v2
   ```

2. **Set up a Python virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Python dependencies:**
   ```bash
   cd breathing_model
   pip install -r requirements.txt
   ```

### Flutter App Setup

1. **Install Flutter SDK** (if not already installed):
   - Follow the official [Flutter installation guide](https://docs.flutter.dev/get-started/install)
   - Ensure Flutter is added to your system PATH

2. **Navigate to the app directory and install dependencies:**
   ```bash
   cd app
   flutter pub get
   ```

3. **Run the application:**
   ```bash
   # For development on different platforms:
   flutter run -d windows      # Windows desktop
   flutter run -d chrome       # Web browser
   flutter run -d android      # Android device/emulator
   flutter run -d ios          # iOS device/simulator
   ```

### Quick Start Example

```python
# Real-time breathing classification
from breathing_model.model.transformer.inference.main import main as run_inference

# Run real-time classification with visualization
run_inference()
```

## 🧠 Model Architectures

### Current Transformer Model
The latest transformer-based architecture provides state-of-the-art performance:

**Architecture Details:**
- **Input**: Mel-spectrogram features (128 mel bins)
- **Model**: Transformer encoder with 6 layers, 8 attention heads
- **Output**: 3-class classification (inhale, exhale, silence)
- **Sample Rate**: 44.1 kHz
- **Chunk Length**: 0.3 seconds

**Usage Example:**
```python
from breathing_model.model.transformer.inference.main import main

# Start real-time classification with GUI
main()
```

**Configuration:**
Model parameters can be adjusted in `breathing_model/model/transformer/config.yaml`

### Legacy Models (Archive)

#### LSTM Model
Traditional sequential model for breath phase classification:
```python
# Example usage (archived)
from breathing_model.archive.lstm.realtime import RealTimeAudioClassifier

classifier = RealTimeAudioClassifier("model_path.pth")
prediction = classifier.predict(audio_data)
```

#### Transformer v1 (Archive)
Previous transformer implementation with ONNX export support:
```python
# ONNX export example
from breathing_model.archive.transformer_model.export_to_onnx import export_breath_classifier_to_onnx

export_breath_classifier_to_onnx("model.pth", "model.onnx")
```

## 📊 Data Generation & Collection

### Tensometer-Based Data Collection
The project includes comprehensive tools for collecting labeled breathing data using tensometer sensors:

**Features:**
- **Bluetooth Integration**: Connects to FT7 tensometer devices
- **Synchronized Recording**: Simultaneous audio and tensometer data capture
- **Automatic Labeling**: Generates breathing phase labels from tensometer readings
- **Multi-format Output**: Supports various data formats for training

**Usage:**
```python
# Start data collection session
python breathing_model/data/generators/tenso_model_based_data_generation/data_recorder.py
```

**Data Processing:**
```python
# Process collected tensometer data
python breathing_model/data/generators/tenso_model_based_data_generation/tenso_model_based_data_gen.py
```

### Manual Data Generation
For scenarios without tensometer hardware:
```python
# Manual data generation tools
python breathing_model/data/generators/manual_data_gen.py
```

## 🎯 Model Training

### Train Current Transformer Model
```bash
cd breathing_model/model/transformer
python train.py
```

### Training Configuration
Modify training parameters in `breathing_model/model/transformer/config.yaml`:
- Batch size, learning rate, epochs
- Audio processing parameters
- Model architecture settings

### Model Evaluation
```bash
# Evaluate trained models
python breathing_model/model/transformer/inference/main.py --evaluate
```

## 📱 Mobile Application

The Flutter application provides a user-friendly interface for real-time breathing monitoring with cross-platform support.

### Features
- **Real-time Visualization**: Live breathing pattern display with smooth animations
- **Phase Classification**: Visual indicators for inhale, exhale, and silence phases
- **Breath Counting**: Automatic inhale/exhale cycle counting
- **Audio Recording**: Built-in recording capabilities with permission handling
- **Cross-platform**: Supports Windows, Web, Android, and iOS

### Platform-Specific Setup

#### Windows
```bash
cd app
flutter run -d windows
```

#### Web
```bash
flutter run -d chrome
```

#### Android (Future Release)
```bash
flutter run -d android
```

#### iOS (Future Release)
```bash
flutter run -d ios
```

### App Configuration
- **Models**: Place ONNX models in `app/assets/models/`
- **Permissions**: Microphone access is automatically requested
- **Themes**: Customizable UI themes available

For detailed app setup instructions, see [app/README.md](app/README.md).

## 🔧 ONNX Model Export

Convert trained PyTorch models to ONNX format for deployment:

```python
# Export transformer model to ONNX
from breathing_model.archive.transformer_model.export_to_onnx import export_breath_classifier_to_onnx

model_path = "breathing_model/model/trained_models/1/transformer_model.pth"
onnx_path = "breathing_model/model/trained_models/1/transformer_model.onnx"

export_breath_classifier_to_onnx(model_path, onnx_path)
```

**ONNX Benefits:**
- **Cross-platform Deployment**: Use models in different environments
- **Performance Optimization**: Optimized inference on various hardware
- **Mobile Integration**: Deploy in Flutter app for offline inference

## 🛠️ Development Tools

### Data Analysis
Jupyter notebooks for data exploration and model analysis:
```bash
# Navigate to analysis tools
cd breathing_model/archive/scripts
jupyter notebook useful-scripts.ipynb
```

### Audio Processing Utilities
```python
# Audio file trimming and preprocessing
python breathing_model/data/trim.py
```

### Model Validation
```python
# Validate model performance
from breathing_model.model.transformer.inference.model_loader import load_model

model = load_model("path/to/model.pth")
# Perform validation...
```

## 📚 Documentation

Comprehensive project documentation is available in the `docs/` directory:

- **Business Presentation**: Overview of project goals and outcomes
- **System Requirements**: Detailed technical specifications
- **Model Evaluation**: Performance analysis and benchmarks
- **Project Organization**: Development methodology and infrastructure
- **Architecture Diagrams**: Technical architecture visualizations

## 🔧 Troubleshooting

### Common Issues

#### Python Environment
```bash
# If PyAudio installation fails on Windows:
pip install pipwin
pipwin install pyaudio

# For Linux/macOS PyAudio issues:
sudo apt-get install portaudio19-dev  # Ubuntu/Debian
brew install portaudio               # macOS
```

#### Flutter Issues
```bash
# Clear Flutter cache
flutter clean
flutter pub get

# Check Flutter installation
flutter doctor
```

#### Audio Device Setup
```bash
# List available audio devices (Python)
python -c "import pyaudio; p = pyaudio.PyAudio(); [print(f'{i}: {p.get_device_info_by_index(i)}') for i in range(p.get_device_count())]"
```

### Performance Optimization
- **GPU Acceleration**: Ensure CUDA is available for PyTorch training
- **Audio Latency**: Adjust chunk size in config for real-time performance
- **Model Size**: Use ONNX quantization for mobile deployment

## 🤝 Contributing

We welcome contributions! Please follow these guidelines:

1. **Fork the repository** and create a feature branch
2. **Follow code style** consistent with existing codebase
3. **Add tests** for new functionality
4. **Update documentation** for significant changes
5. **Submit a pull request** with clear description

### Development Setup
```bash
# Install development dependencies
pip install -r breathing_model/requirements.txt

# Install pre-commit hooks (if available)
pre-commit install
```

### Coding Standards
- **Python**: Follow PEP 8 guidelines
- **Dart/Flutter**: Follow official Dart style guide
- **Documentation**: Update README and inline comments
- **Testing**: Include unit tests for new features

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License
Copyright (c) 2024 Piotr Sulewski
```

## 👨‍💻 Authors

- **Piotr Sulewski** - *Lead Developer* - [peterprospl12](https://github.com/peterprospl12)

## 🙏 Acknowledgments

- Medical professionals who provided guidance on breathing pattern analysis
- Open source community for excellent libraries (PyTorch, Flutter, librosa)
- Contributors who helped with testing and validation

## 📧 Contact

For questions, issues, or collaboration opportunities:
- **GitHub Issues**: [Create an issue](https://github.com/peterprospl12/breathing-classification-v2/issues)
- **Project Repository**: [breathing-classification-v2](https://github.com/peterprospl12/breathing-classification-v2)

---

*This project aims to advance breathing pattern analysis through machine learning, contributing to health monitoring and medical research applications.*