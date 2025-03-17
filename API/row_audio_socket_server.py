# socket_server.py
import socket
import torch
import pickle
import numpy as np
from server_dependencies.server_dependecies import RealTimeAudioClassifier, MelTransformer, device

MODEL_PATH = 'server_dependencies/best_breath_seq_transformer_model_CURR_BEST.pth'
classifier = RealTimeAudioClassifier(MODEL_PATH)
mel_transformer = MelTransformer()


def start_socket_server():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(('localhost', 50000))
    sock.listen(1)

    print("Socket server running on port 50000")
    try:
        while True:
            conn, addr = sock.accept()
            print(f"Connection from {addr}")
            try:
                # Read data size (4 bytes)
                size_bytes = conn.recv(4)
                if not size_bytes:
                    continue

                data_size = int.from_bytes(size_bytes, byteorder='big')
                print(f"Receiving {data_size} bytes")

                # Receive raw audio data
                data = b''
                while len(data) < data_size:
                    packet = conn.recv(min(4096, data_size - len(data)))
                    if not packet:
                        break
                    data += packet

                if len(data) == data_size:
                    # Deserialize raw audio data
                    audio_data = pickle.loads(data)

                    print(f"Received audio data shape: {audio_data.shape}")

                    # Calculate mel spectrogram
                    mel_spec = mel_transformer.get_mel_transform(audio_data)

                    # Predict using the model
                    prediction, prediction_name, confidence = classifier.predict(
                        mel_spec, dont_calc_mel=True
                    )

                    print(f"Prediction: {prediction} ({prediction_name}) - Confidence: {confidence:.4f}")

                    # Send back prediction (4 bytes)
                    conn.sendall(prediction.to_bytes(4, byteorder='big'))
            except Exception as e:
                print(f"Error processing request: {e}")
            finally:
                conn.close()
    finally:
        sock.close()


if __name__ == "__main__":
    start_socket_server()