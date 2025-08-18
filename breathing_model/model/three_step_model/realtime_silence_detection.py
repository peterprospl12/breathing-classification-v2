import numpy as np
import pyaudio
import matplotlib.pyplot as plt
import time
from model.transformer_model_ref.inference.audio import SharedAudioResource

#############################################
# Settings and constants
#############################################
REFRESH_TIME = 0.2
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
DEVICE_INDEX = 1
CHUNK_SIZE = int(RATE * REFRESH_TIME)

# Volume threshold for silence detection
VOLUME_THRESHOLD = 300

running = True

#############################################
# Volume-based classifier
#############################################
class VolumeBasedClassifier:
    def __init__(self, threshold=None):
        if threshold is None:
            self.threshold = VOLUME_THRESHOLD
        else:
            self.threshold = threshold

    def set_threshold(self, new_threshold):
        self.threshold = new_threshold
        print(f"Volume threshold updated to: {self.threshold}")

    def classify(self, audio_data):
        vol = np.sqrt(np.mean(np.square(audio_data.astype(np.float32))))

        return (1 if vol > self.threshold else 0), vol

#############################################
# Threshold calibration function
#############################################
def calibrate_threshold():
    global VOLUME_THRESHOLD, classifier

    print("Calibration. Keep silence for 3 seconds...")
    silence_volumes = []
    start_time = time.time()

    while time.time() - start_time < 3:
        buffer = audio.read()
        if buffer is None:
            continue
        _, vol = classifier.classify(buffer)
        silence_volumes.append(vol)

    mean_silence = np.mean(silence_volumes)
    std_silence = np.std(silence_volumes)

    VOLUME_THRESHOLD = int(mean_silence + 2.0 * std_silence)
    classifier.set_threshold(VOLUME_THRESHOLD)

    print(f"New threshold: {VOLUME_THRESHOLD}")


#############################################
# Plot configuration
#############################################

PLOT_TIME_HISTORY = 5  # seconds
PLOT_CHUNK_SIZE = CHUNK_SIZE
plot_data = np.zeros((RATE * PLOT_TIME_HISTORY, 1))
x_line_space = np.arange(0, RATE * PLOT_TIME_HISTORY, 1)
predictions = np.zeros((int(PLOT_TIME_HISTORY / REFRESH_TIME), 1))

fig, ax = plt.subplots(1, 1, figsize=(10, 6))
ax.plot(plot_data, color='white')


def on_key(event):
    global running, VOLUME_THRESHOLD
    match event.key:
        case ' ':
            plt.close()
            running = False
        case 'c':
            calibrate_threshold()
        case 'up':
            VOLUME_THRESHOLD += 10
            classifier.set_threshold(VOLUME_THRESHOLD)
        case 'down':
            VOLUME_THRESHOLD = max(10, VOLUME_THRESHOLD - 10)
            classifier.set_threshold(VOLUME_THRESHOLD)


fig.canvas.manager.set_window_title('Realtime Breath Detector ([SPACE] to stop, [C] to calibrate, [UP]/[DOWN] to adjust threshold)')
fig.suptitle(f'Volume Threshold: {VOLUME_THRESHOLD} (Blue - Silence, Red - Breath Detected)')
fig.canvas.mpl_connect('key_press_event', on_key)
y_lim = (-500, 500)
face_color = (0, 0, 0)
ax.set_facecolor(face_color)
ax.set_ylim(y_lim)


def update_plot(frames, current_prediction):
    global plot_data, predictions, ax
    # Update plot buffer
    plot_data = np.roll(plot_data, -len(frames))
    plot_data[-len(frames):] = frames.reshape(-1, 1)
    predictions = np.roll(predictions, -1)
    predictions[-1] = current_prediction

    ax.clear()
    # For each segment (REFRESH_TIME window) plot the signal with color based on prediction
    for i in range(len(predictions)):
        if predictions[i] == 0:
            color = 'blue'  # silence
        elif predictions[i] == 1:
            color = 'red'  # breath detected
        start = i * PLOT_CHUNK_SIZE
        end = (i + 1) * PLOT_CHUNK_SIZE
        ax.plot(x_line_space[start:end], plot_data[start:end], color=color)

    threshold_line = VOLUME_THRESHOLD
    ax.axhline(y=threshold_line, color='yellow', linestyle='--', linewidth=1.5, label='Threshold')
    ax.axhline(y=-threshold_line, color='yellow', linestyle='--', linewidth=1.5, label='Threshold')

    ax.set_facecolor(face_color)
    ax.set_ylim(y_lim)
    fig.suptitle(f'Volume Threshold: {VOLUME_THRESHOLD} (Blue - Silence, Red - Breath Detected)')
    plt.draw()
    plt.pause(0.01)


if __name__ == '__main__':
    # Initialize components
    audio = SharedAudioResource(chunk_size=CHUNK_SIZE, format=FORMAT, channels=CHANNELS,
                                rate=RATE, device_index=DEVICE_INDEX)
    classifier = VolumeBasedClassifier(VOLUME_THRESHOLD)

    try:
        while running:
            start_time = time.time()

            # Read audio buffer
            buffer = audio.read()
            if buffer is None:
                print("Buffer error - skipping iteration")
                time.sleep(REFRESH_TIME)
                continue

            # Classify
            prediction, volume = classifier.classify(buffer)

            # Update plot
            update_plot(buffer, prediction)

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        running = False
        audio.close()
        plt.close('all')
        print("Audio stream closed.")
