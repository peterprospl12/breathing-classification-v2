import numpy as np
import pyaudio
import matplotlib.pyplot as plt
import time
from model_classes import AudioClassifierLSTM as AudioClassifier
import torch
import librosa

"""

INSTRUCTION

Firstly start the program. In console it will write all possible input devices. Change
DEVICE_INDEX constant to the microphone's index that you want to use. You can also change sample rate constant,
but it is advisable to leave it as it is (44100 Hz). Then close program (press space bar) and run it again
with changed constants.

If you don't want to calibrate microphone, you want to do this manually or you are
using microphone plugged to USB port you can set CALIBRATE_MICROPHONE to False.

Calibration works only for input devices connected to mini jack port or built-in in laptop.
Program will calibrate device that is set as sys default'

Before starting program, please set your microphone volume to max manually and don't breathe
until message on program window stop showing 'Dont breathe! Calibrating microphone...'.

If program will classify silence as other classes it is probably because microphone sensitivity is not
set correctly. Try running program again to calibrate it again or try to adjust sensitivity manually.

You can press 'r' to reset inhale and exhale counters.

Have fun!

"""

# Model path

CLASSIFIER_MODEL_PATH = 'model_lstm.pth'

# Constants

REFRESH_TIME = 0.25

INHALE_COUNTER = 0
EXHALE_COUNTER = 0

CHANNELS = 1
RATE = 44100
DEVICE_INDEX = 4
CHUNK_SIZE = int(RATE * REFRESH_TIME)

running = True

state = torch.load(CLASSIFIER_MODEL_PATH, map_location=torch.device('cpu'))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Audio resource class


class SharedAudioResource:

    def __init__(self):
        self.p = pyaudio.PyAudio()
        self.buffer_size = int(RATE * REFRESH_TIME)
        self.buffer = np.zeros(self.buffer_size, dtype=np.int16)
        for i in range(self.p.get_device_count()):
            print(self.p.get_device_info_by_index(i))
        self.stream = self.p.open(format=pyaudio.paInt16, channels=CHANNELS, rate=RATE, input=True,
                                  frames_per_buffer=self.buffer_size, input_device_index=DEVICE_INDEX)

    def read(self):
        self.buffer = self.stream.read(
            self.buffer_size, exception_on_overflow=False)
        audio_data = np.frombuffer(self.buffer, dtype=np.int16)
        if CHANNELS == 2:
            audio_data = audio_data.reshape(-1, 2)
            audio_data = audio_data.mean(axis=1).astype(np.int16)
        return audio_data

    def close(self):
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()


# Class for prediction
class RealTimeAudioClassifier:
    def __init__(self, model_path):
        self.model = AudioClassifier()
        self.model.load_state_dict(torch.load(
            model_path, map_location=torch.device('cpu')))
        self.model = self.model.to(device)
        self.model.eval()
        self.hidden = None

    def predict(self, y):
        # Make sure that frame is mono, 44100 Hz and in int16 format
        if y.dtype != np.int16:
            raise Exception("Data type is not int16.")
        if y.ndim > 1 and y.shape[1] > 1:
            raise Exception("Audio is not mono.")

        # Conversion to float32 from int16
        frames_float32 = y.astype(np.float32) / np.iinfo(np.int16).max

        # Make sure that frame is mono, 44100 Hz and converted to float32
        if frames_float32.ndim > 1 and frames_float32.shape[1] > 1:
            raise Exception("Audio is not mono.")
        if frames_float32.dtype != np.float32:
            raise Exception("Data type is not float32.")

        # # Perform Short-Time Fourier Transform (STFT) to get spectrogram
        # stft = librosa.stft(frames_float32, n_fft=512, hop_length=256)
        # spectrogram = np.abs(stft)
        #
        # # Convert spectrogram to decibels
        # log_spectrogram = librosa.amplitude_to_db(spectrogram, ref=np.max)
        #
        # # Convert amplitude to dB
        # features = log_spectrogram.mean(axis=1)

        # Calculate mel-spectrogram with larger n_fft (e.g., 1024) and specify the number of mel bands (e.g., 128)
        mel_spec = librosa.feature.melspectrogram(
            y=frames_float32,
            sr=44100,
            n_fft=1024,  # Larger FFT window
            hop_length=512,
            n_mels=40  # Number of mel bands
        )

        # Convert amplitude to decibel scale
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

        # Extract features as the mean value for each mel band
        features = log_mel_spec.mean(axis=1)

        single_data = torch.tensor(features, dtype=torch.float32).unsqueeze(
            0).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs, self.hidden = self.model(single_data, self.hidden)
            predicted = outputs[0, 0].detach().cpu().numpy()

        return predicted


# Plot variables

PLOT_TIME_HISTORY = 5
PLOT_CHUNK_SIZE = int(RATE * REFRESH_TIME)

plot_data = np.zeros((RATE * PLOT_TIME_HISTORY, 1))
x_line_space = np.arange(0, RATE * PLOT_TIME_HISTORY, 1)
predictions = np.zeros((int(PLOT_TIME_HISTORY / REFRESH_TIME), 1))

fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(plot_data, color='white')


# Key handler for plot window

def on_key(event):
    global running
    if event.key == ' ':
        plt.close()
        running = False
    elif event.key == 'r':
        global INHALE_COUNTER, EXHALE_COUNTER
        INHALE_COUNTER = 0
        EXHALE_COUNTER = 0


# Configuration of plot properties and other elements

fig.canvas.manager.set_window_title(
    # Title
    'Realtime Breath Detector ( Press [SPACE] to stop, [R] to reset counter )')
# Instruction
fig.suptitle(
    f'Inhales: {INHALE_COUNTER}  Exhales: {EXHALE_COUNTER}        Colours meaning: Red - Inhale, Green - Exhale, Blue - Silence')
fig.canvas.mpl_connect('key_press_event', on_key)  # Key handler

y_lim = (-500, 500)
face_color = (0, 0, 0)

ax.set_facecolor(face_color)
ax.set_ylim(y_lim)


# Plot update function

def update_plot(frames, current_prediction):
    global plot_data, predictions, ax

    # Roll signals and predictions vectors and insert new value at the end

    plot_data = np.roll(plot_data, -len(frames))
    plot_data[-len(frames):] = frames.reshape(-1, 1)

    predictions = np.roll(predictions, -1)
    predictions[-1] = current_prediction

    # Clean the plot and plot the new data-raw

    ax.clear()

    for i in range(0, len(predictions)):
        if predictions[i] == 0:  # Exhale
            color = 'green'
        elif predictions[i] == 1:  # Inhale
            color = 'red'
        else:  # Silence
            color = 'blue'
        ax.plot(x_line_space[PLOT_CHUNK_SIZE * i:PLOT_CHUNK_SIZE * (i + 1)],
                plot_data[PLOT_CHUNK_SIZE * i:PLOT_CHUNK_SIZE * (i + 1)] / 4, color=color)

    # Set plot properties and show it

    ax.set_facecolor(face_color)
    ax.set_ylim(y_lim)

    # Instruction
    fig.suptitle(
        f'Inhales: {INHALE_COUNTER}  Exhales: {EXHALE_COUNTER}        Colours meaning: Red - Inhale, Green - Exhale, Blue - Silence')

    plt.draw()
    plt.pause(0.01)


# Main function

if __name__ == "__main__":

    # Initialize microphone and classifier

    audio = SharedAudioResource()

    classifier = RealTimeAudioClassifier(CLASSIFIER_MODEL_PATH)

    # Main loop

    last_prediction = 2
    while running:

        # Set timer to check how long each prediction takes

        start_time = time.time()

        # Collect samples

        buffer = audio.read()

        if buffer is None:
            continue

        # Make prediction
        prediction = classifier.predict(buffer)

        # Determine the class of the prediction
        print(prediction)
        print(np.argmax(prediction))

        # Update plot
        update_plot(buffer, np.argmax(prediction))

        # Print time needed for this loop iteration

        print(time.time() - start_time)
    # Close audio

    audio.close()
