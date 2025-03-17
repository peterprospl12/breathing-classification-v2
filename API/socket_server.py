# socket_server.py
import socket
import torch
import pickle
import numpy as np
from server_dependencies.server_dependecies import RealTimeAudioClassifier, device

MODEL_PATH = 'server_dependencies/best_breath_seq_transformer_model_CURR_BEST.pth'
classifier = RealTimeAudioClassifier(MODEL_PATH)


def start_socket_server():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(('localhost', 50000))
    sock.listen(1)

    print("Socket server running on port 8000")
    try:
        while True:
            conn, addr = sock.accept()
            try:
                data_size = int.from_bytes(conn.recv(4), byteorder='big')
                data = b''
                while len(data) < data_size:
                    packet = conn.recv(min(4096, data_size - len(data)))
                    if not packet:
                        break
                    data += packet

                if len(data) == data_size:
                    mel_np = pickle.loads(data)
                    mel_tensor = torch.tensor(mel_np).to(device)
                    prediction, _, _ = classifier.predict(mel_tensor, dont_calc_mel=False)
                    conn.sendall(prediction.to_bytes(4, byteorder='big'))
            finally:
                conn.close()
    finally:
        sock.close()


if __name__ == "__main__":
    start_socket_server()