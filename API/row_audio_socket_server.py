import socket
import numpy as np
import traceback
from server_dependencies.server_dependecies import RealTimeAudioClassifier, MelTransformer, device

MODEL_PATH = 'server_dependencies/best_breath_seq_transformer_model_CURR_BEST.pth'
classifier = RealTimeAudioClassifier(MODEL_PATH)
mel_transformer = MelTransformer()

def start_socket_server():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(('localhost', 50000))
    sock.listen(5)  # Allow up to 5 pending connections

    print("Socket server running on port 50000")
    try:
        while True:
            conn, addr = sock.accept()
            print(f"Connection established with {addr}")

            # Set timeout to detect disconnected clients
            conn.settimeout(60.0)  # 60 second timeout

            handle_client(conn, addr)
    finally:
        sock.close()

def handle_client(conn, addr):
    try:
        while True:
            try:
                # Read data size (4 bytes)
                size_bytes = conn.recv(4)
                if not size_bytes or len(size_bytes) < 4:
                    print(f"Client {addr} disconnected")
                    break

                data_size = int.from_bytes(size_bytes, byteorder='big')
                print(f"Receiving {data_size} bytes from {addr}")

                # Receive raw audio data
                data = b''
                bytes_received = 0

                while bytes_received < data_size:
                    chunk_size = min(4096, data_size - bytes_received)
                    packet = conn.recv(chunk_size)

                    if not packet:
                        print(f"Connection with {addr} lost during data transfer")
                        return

                    data += packet
                    bytes_received += len(packet)

                # Process the received data
                audio_data = np.frombuffer(data, dtype=np.int16)
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

            except socket.timeout:
                print(f"Connection with {addr} timed out")
                break


                if len(data) == data_size:
                    try:
                        # Convert raw binary data to numpy array
                        # Assuming 16-bit PCM audio format (int16)
                        audio_data = np.frombuffer(data, dtype=np.int16)
                        
                        # If needed, reshape the array based on expected dimensions
                        # If stereo, you might need to reshape: audio_data = audio_data.reshape(-1, 2)
                        
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
                        print(f"Error processing audio data: {e}")
                        # First few bytes might help diagnose the issue
                        print(f"First 20 bytes of data: {data[:20]}")
                        conn.sendall((0).to_bytes(4, byteorder='big'))  # Send error code
                else:
                    print(f"Incomplete data received: got {len(data)} bytes, expected {data_size}")
                    conn.sendall((0).to_bytes(4, byteorder='big'))  # Send error code
            except Exception as e:
                print(f"Error processing request from {addr}: {e}")
                traceback.print_exc()
                break

    finally:
        conn.close()
        print(f"Connection with {addr} closed")

if __name__ == "__main__":
    start_socket_server()
