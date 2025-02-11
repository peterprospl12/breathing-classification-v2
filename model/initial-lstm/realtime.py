import wave

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

# Constants

REFRESH_TIME = 0.25

FORMAT = pyaudio.paInt16

INHALE_COUNTER = 0
EXHALE_COUNTER = 0

CHANNELS = 1
RATE = 44100
DEVICE_INDEX = 4
CHUNK_SIZE = int(RATE * REFRESH_TIME)

running = True

# Model path

CLASSIFIER_MODEL_PATH = 'audio_lstm_classifier_test.pth'

state = torch.load(CLASSIFIER_MODEL_PATH, map_location=torch.device('cpu'))
print("Klucze w zapisanym state_dict:", list(state.keys()))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Audio resource class

class SharedAudioResource:
    buffer = None

    def __init__(self):
        self.p = pyaudio.PyAudio()
        self.buffer_size = int(RATE * REFRESH_TIME)
        self.buffer = np.zeros(self.buffer_size, dtype=np.int16)
        for i in range(self.p.get_device_count()):
            print(self.p.get_device_info_by_index(i))
        self.stream = self.p.open(format=pyaudio.paInt16, channels=CHANNELS, rate=RATE, input=True,
                                  frames_per_buffer=self.buffer_size, input_device_index=DEVICE_INDEX)

    def read(self):
        self.buffer = self.stream.read(self.buffer_size, exception_on_overflow=False)
        return np.frombuffer(self.buffer, dtype=np.int16)

    def close(self):
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()


# Class for prediction

class RealTimeAudioClassifier:
    def __init__(self, model_path):
        self.model = AudioClassifier()
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.model = self.model.to(device)
        self.model.eval()
        self.hidden = None

    def predict(self, y, sr=RATE):
        frames = y  # mono
        frames = frames[:CHUNK_SIZE]

        frames = frames.astype(np.float32)
        frames /= np.iinfo(np.int16).max
        mfcc = librosa.feature.mfcc(y=frames, sr=sr)

        mfcc_mean = np.mean(mfcc, axis=1)

        single_data = torch.tensor(mfcc_mean, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

        outputs, self.hidden = self.model(single_data, None)

        predicted = outputs[0, 0].detach().cpu().numpy()
        print(predicted)
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

fig.canvas.manager.set_window_title('Realtime Breath Detector ( Press [SPACE] to stop, [R] to reset counter )')  # Title
fig.suptitle(f'Inhales: {INHALE_COUNTER}  Exhales: {EXHALE_COUNTER}        Colours meaning: Red - Inhale, Green - Exhale, Blue - Silence')  # Instruction
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

    # Clean the plot and plot the new data

    ax.clear()

    for i in range(0, len(predictions)):
        if predictions[i] == 0:  # Exhale
            color = 'green'
        elif predictions[i] == 1:  # Inhale
            color = 'red'
        else:  # Silence
            color = 'blue'
        ax.plot(x_line_space[PLOT_CHUNK_SIZE * i:PLOT_CHUNK_SIZE * (i + 1)],
                plot_data[PLOT_CHUNK_SIZE * i:PLOT_CHUNK_SIZE * (i + 1)]/ 4, color=color)

    # Set plot properties and show it

    ax.set_facecolor(face_color)
    ax.set_ylim(y_lim)

    fig.suptitle(f'Inhales: {INHALE_COUNTER}  Exhales: {EXHALE_COUNTER}        Colours meaning: Red - Inhale, Green - Exhale, Blue - Silence')  # Instruction

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
        print(np.argmax(prediction))

        # Update plot
        update_plot(buffer, np.argmax(prediction))

        # Print time needed for this loop iteration

        print(time.time() - start_time)
    # Close audio

    audio.close()
