import numpy as np
import pyaudio
import matplotlib.pyplot as plt
import time
import torch
import torchaudio
import math
import requests
from enum import Enum


# from torch.distributed.rpc.internal import serialize


#############################################
# Model – CNN + Transformer
#############################################

class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class BreathPhaseTransformerSeq(torch.nn.Module):
    def __init__(self, n_mels=40, num_classes=3, d_model=128, nhead=4, num_transformer_layers=2):
        super(BreathPhaseTransformerSeq, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=(3, 3), stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))

        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=(3, 3), stride=1, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(64)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))

        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=(3, 3), stride=1, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(128)
        self.pool3 = torch.nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))

        self.out_freq = n_mels // 8
        cnn_feature_dim = 128 * self.out_freq

        self.fc_proj = torch.nn.Linear(cnn_feature_dim, d_model)
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=0.1, batch_first=True)
        self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)
        self.pos_encoder = PositionalEncoding(d_model=d_model, dropout=0.1)
        self.dropout = torch.nn.Dropout(0.3)
        self.fc_out = torch.nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x: (batch, 1, n_mels, time_steps)
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = torch.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        # shape: (batch, 128, out_freq, time_steps)
        x = x.permute(0, 3, 1, 2)  # (batch, time_steps, channels, out_freq)
        batch_size, time_steps, channels, freq = x.size()
        x = x.contiguous().view(batch_size, time_steps, channels * freq)  # (batch, time_steps, cnn_feature_dim)
        x = self.fc_proj(x)  # (batch, time_steps, d_model)
        x = self.pos_encoder(x)
        x = self.transformer(x)  # (batch, time_steps, d_model)
        x = self.dropout(x)
        logits = self.fc_out(x)  # (batch, time_steps, num_classes)
        return logits


#############################################
# Settings and constants
#############################################
MODEL_PATH = 'best_breath_seq_transformer_model_CURR_BEST.pth'  # Path to the trained model

REFRESH_TIME = 0.3  # time in seconds to read audio
FORMAT = pyaudio.paInt16
CHANNELS = 1  # 1 mono | 2 stereo
RATE = 44100  # sampling rate
DEVICE_INDEX = 1  # microphone device index (listed in the console output)
CHUNK_SIZE = int(RATE * REFRESH_TIME)

INHALE_COUNTER = 0
EXHALE_COUNTER = 0

running = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#############################################
# Audio handling class
#############################################
class SharedAudioResource:
    def __init__(self):
        self.p = pyaudio.PyAudio()
        self.buffer_size = CHUNK_SIZE
        # Print available devices
        for i in range(self.p.get_device_count()):
            print(self.p.get_device_info_by_index(i))
        self.stream = self.p.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                                  input=True, frames_per_buffer=self.buffer_size,
                                  input_device_index=DEVICE_INDEX)

    def read(self):
        data = self.stream.read(self.buffer_size, exception_on_overflow=False)
        return np.frombuffer(data, dtype=np.int16)

    def close(self):
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()


class MelTransformer:
    def __init__(self):
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=RATE,
            n_fft=1024,
            hop_length=512,
            n_mels=40
        )

    def get_mel_transform(self, y, sr=RATE):
        # y: int16 signal; convert to float32 in the range [-1, 1]
        y = y.astype(np.float32) / 32768.0
        # Ensure the signal is mono
        if y.ndim != 1:
            raise Exception("Otrzymano sygnał nie-mono!")
        # Convert to tensor (shape: [1, num_samples])
        waveform = torch.tensor(y, dtype=torch.float32).unsqueeze(0)
        # Compute Mel spectrogram – result: [1, n_mels, time_steps]
        mel_spec = self.mel_transform(waveform)
        mel_spec = torch.log(mel_spec + 1e-9)
        # Add channel dimension – expected shape: (batch, 1, n_mels, time_steps)d
        mel_spec = mel_spec.unsqueeze(0)
        return mel_spec


class PredictionModes(Enum):
    LOCAL = 1
    HTTP_SERVER = 2
    PRE_CALC_MEL_SOCKET = 3
    SOCKET = 4


#############################################
# Prediction class
#############################################
class RealTimeAudioClassifier:
    def __init__(self, model_path, mode: PredictionModes, http_url=None, socket_server_port=None):
        self.model = BreathPhaseTransformerSeq(n_mels=40, num_classes=3, d_model=128, nhead=4,
                                               num_transformer_layers=2).to(device)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()
        self.mel_transformer = MelTransformer()
        self.server_url = http_url
        self.socket_port = socket_server_port
        self.mode = mode
        self.socket_connection = None

        # Initialize socket connection if using socket mode
        if self.mode in [PredictionModes.PRE_CALC_MEL_SOCKET, PredictionModes.SOCKET]:
            self._connect_socket()

    def _connect_socket(self):
        if self.socket_connection is not None:
            try:
                self.socket_connection.close()
            except:
                pass

        import socket
        self.socket_connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self.socket_connection.connect(('localhost', self.socket_port))
            print(f"Connected to socket server on port {self.socket_port}")
        except Exception as e:
            print(f"Failed to connect to socket server: {e}")
            self.socket_connection = None

    def send_to_server(self, y):
        if self.mode is PredictionModes.HTTP_SERVER:
            if self.server_url is None:
                raise Exception("Server URL is not provided")

            mel_spec = self.mel_transformer.get_mel_transform(y)
            mel_np = mel_spec.cpu().numpy()
            response = requests.post(self.server_url, json={'mel_data': mel_np.tolist()})
            return response.json()['prediction']

        if self.mode in [PredictionModes.PRE_CALC_MEL_SOCKET, PredictionModes.SOCKET]:
            import pickle

            # Ensure we have a socket connection
            if self.socket_connection is None:
                self._connect_socket()
                if self.socket_connection is None:
                    # Fall back to local prediction if connection failed
                    return self._local_predict(y)

            # Process data based on mode
            if self.mode is PredictionModes.PRE_CALC_MEL_SOCKET:
                mel_spec = self.mel_transformer.get_mel_transform(y)
                data_to_send = mel_spec.cpu().numpy()
            else:  # SOCKET mode - send raw audio
                data_to_send = y

            # Serialize and send data
            try:
                serialized = pickle.dumps(data_to_send)
                data_size = len(serialized)

                self.socket_connection.sendall(data_size.to_bytes(4, byteorder='big'))
                self.socket_connection.sendall(serialized)

                result_bytes = self.socket_connection.recv(4)
                if result_bytes:
                    prediction = int.from_bytes(result_bytes, byteorder='big')
                    return prediction
                else:
                    # Connection lost, try to reconnect
                    print("Connection lost, reconnecting...")
                    self._connect_socket()
                    return self._local_predict(y)

            except Exception as e:
                print(f"Socket error: {e}")
                # Reset socket connection and try to reconnect next time
                self.socket_connection = None
                return self._local_predict(y)

    def _local_predict(self, y):
        """Fallback method for local prediction when server connection fails"""
        with torch.no_grad():
            mel = self.mel_transformer.get_mel_transform(y)
            mel = mel.to(device)
            logits = self.model(mel)  # shape: (1, time_steps, num_classes)
            probabilities = torch.softmax(logits, dim=2)
            probs_np = probabilities.squeeze(0).cpu().numpy()  # (time_steps, num_classes)
            # Aggregate predictions by frames – choose the most frequent class
            preds = np.argmax(probs_np, axis=1)
            predicted_class = int(np.bincount(preds).argmax())
        return predicted_class

    def predict(self, y, sr=RATE):
        if self.mode is not PredictionModes.LOCAL:
            try:
                predicted_class = self.send_to_server(y)
                if predicted_class is not None:
                    return predicted_class
            except Exception as e:
                print(f"Server prediction failed: {e}, falling back to local")
                # Fall back to local prediction in case of failure
                return self._local_predict(y)
        else:
            return self._local_predict(y)

    def __del__(self):
        """Clean up resources when the object is destroyed"""
        if self.socket_connection is not None:
            try:
                self.socket_connection.close()
            except:
                pass


#############################################
# Plot configuration
# #############################################
PLOT_TIME_HISTORY = 5  # seconds
PLOT_CHUNK_SIZE = CHUNK_SIZE
plot_data = np.zeros((RATE * PLOT_TIME_HISTORY, 1))
x_line_space = np.arange(0, RATE * PLOT_TIME_HISTORY, 1)
predictions = np.zeros((int(PLOT_TIME_HISTORY / REFRESH_TIME), 1))

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(plot_data, color='white')


def on_key(event):
    global running, INHALE_COUNTER, EXHALE_COUNTER
    if event.key == ' ':
        plt.close()
        running = False
    elif event.key == 'r':
        INHALE_COUNTER = 0
        EXHALE_COUNTER = 0


fig.canvas.manager.set_window_title('Realtime Breath Detector (Press [SPACE] to stop, [R] to reset counter)')
fig.suptitle(f'Inhales: {INHALE_COUNTER}  Exhales: {EXHALE_COUNTER}   (Red - Inhale, Green - Exhale, Blue - Silence)')
fig.canvas.mpl_connect('key_press_event', on_key)
y_lim = (-500, 500)
face_color = (0, 0, 0)
ax.set_facecolor(face_color)
ax.set_ylim(y_lim)


def update_plot(frames, current_prediction):
    global plot_data, predictions, ax, INHALE_COUNTER, EXHALE_COUNTER
    # Update plot buffer
    plot_data = np.roll(plot_data, -len(frames))
    plot_data[-len(frames):] = frames.reshape(-1, 1)
    predictions = np.roll(predictions, -1)
    predictions[-1] = current_prediction

    if current_prediction == 0:
        EXHALE_COUNTER += 1
    elif current_prediction == 1:
        INHALE_COUNTER += 1

    ax.clear()
    # For each segment (REFRESH_TIME window) plot the signal with color based on prediction
    for i in range(len(predictions)):
        if predictions[i] == 0:
            color = 'green'  # exhale
        elif predictions[i] == 1:
            color = 'red'  # inhale
        else:
            color = 'blue'  # silence
        start = i * PLOT_CHUNK_SIZE
        end = (i + 1) * PLOT_CHUNK_SIZE
        ax.plot(x_line_space[start:end], plot_data[start:end] / 4, color=color)
    ax.set_facecolor(face_color)
    ax.set_ylim(y_lim)
    fig.suptitle(
        f'Inhales: {INHALE_COUNTER}  Exhales: {EXHALE_COUNTER}   (Red - Inhale, Green - Exhale, Blue - Silence)')
    plt.draw()
    plt.pause(0.01)


if __name__ == '__main__':
    audio = SharedAudioResource()
    classifier = RealTimeAudioClassifier(MODEL_PATH, PredictionModes.SOCKET, socket_server_port=50000)

    while running:
        start_time = time.time()

        # Read CHUNK_SIZE samples from the microphone
        buffer = audio.read()
        if buffer is None:
            continue
        print(buffer.shape)
        prediction = classifier.predict(buffer)

        print("Prediction:", prediction)

        update_plot(buffer, prediction)
        print("Iteration time:", time.time() - start_time)
    audio.close()