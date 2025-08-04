import numpy as np
import pyaudio
import matplotlib.pyplot as plt
import time
from model.lib import SharedAudioResource

#############################################
# Settings and constants
#############################################
REFRESH_TIME = 0.25  # time in seconds to read audio
FORMAT = pyaudio.paInt16
CHANNELS = 1  # 1 mono | 2 stereo
RATE = 44100  # sampling rate
DEVICE_INDEX = 1  # microphone device index (listed in the console output)
CHUNK_SIZE = int(RATE * REFRESH_TIME)

# Volume threshold for silence detection
VOLUME_THRESHOLD = 500  # Initial threshold value

SOUND_COUNTER = 0
SILENCE_COUNTER = 0

running = True


#############################################
# Volume-based classifier
#############################################
class VolumeBasedClassifier:
    def __init__(self, threshold=VOLUME_THRESHOLD):
        self.threshold = threshold

    def set_threshold(self, new_threshold):
        """Update the volume threshold"""
        self.threshold = new_threshold
        print(f"Volume threshold updated to: {self.threshold}")

    def classify(self, audio_data):
        """
        Classify audio data based on volume threshold
        Returns: 0 = silence (blue), 1 = sound (red)
        """
        # Calculate RMS (Root Mean Square) as volume measure
        rms_volume = np.sqrt(np.mean(audio_data.astype(np.float32) ** 2))

        # Alternative: use absolute average
        # avg_volume = np.mean(np.abs(audio_data))

        if rms_volume > self.threshold:
            return 1, rms_volume  # Sound detected
        else:
            return 0, rms_volume  # Silence


#############################################
# Plot configuration
#############################################
PLOT_TIME_HISTORY = 5  # seconds
PLOT_CHUNK_SIZE = CHUNK_SIZE
plot_data = np.zeros((RATE * PLOT_TIME_HISTORY, 1))
x_line_space = np.arange(0, RATE * PLOT_TIME_HISTORY, 1)
predictions = np.zeros((int(PLOT_TIME_HISTORY / REFRESH_TIME), 1))

fig, ax = plt.subplots(1, 1, figsize=(12, 8))
ax.plot(plot_data, color='white')


def on_key(event):
    global running, SOUND_COUNTER, SILENCE_COUNTER, VOLUME_THRESHOLD, classifier

    if event.key == ' ':
        plt.close()
        running = False
    elif event.key == 'r':
        SOUND_COUNTER = 0
        SILENCE_COUNTER = 0
        print("Counters reset")
    elif event.key == 'up':
        # Increase threshold
        VOLUME_THRESHOLD += 50
        classifier.set_threshold(VOLUME_THRESHOLD)
    elif event.key == 'down':
        # Decrease threshold
        VOLUME_THRESHOLD = max(10, VOLUME_THRESHOLD - 50)  # Minimum threshold of 10
        classifier.set_threshold(VOLUME_THRESHOLD)
    elif event.key == '+' or event.key == '=':
        # Fine increase
        VOLUME_THRESHOLD += 10
        classifier.set_threshold(VOLUME_THRESHOLD)
    elif event.key == '-':
        # Fine decrease
        VOLUME_THRESHOLD = max(10, VOLUME_THRESHOLD - 10)
        classifier.set_threshold(VOLUME_THRESHOLD)


# Initialize plot
fig.canvas.manager.set_window_title('Volume-based Audio Detector')
fig.canvas.mpl_connect('key_press_event', on_key)
y_lim = (-1000, 1000)
face_color = (0, 0, 0)
ax.set_facecolor(face_color)
ax.set_ylim(y_lim)


def update_plot(frames, current_prediction, current_volume):
    global plot_data, predictions, ax, SOUND_COUNTER, SILENCE_COUNTER, VOLUME_THRESHOLD

    # Update plot buffer
    plot_data = np.roll(plot_data, -len(frames))
    plot_data[-len(frames):] = frames.reshape(-1, 1)
    predictions = np.roll(predictions, -1)
    predictions[-1] = current_prediction

    # Update counters
    if current_prediction == 1:  # Sound detected
        SOUND_COUNTER += 1
    else:  # Silence
        SILENCE_COUNTER += 1

    ax.clear()

    # Plot the signal with color based on prediction
    for i in range(len(predictions)):
        if predictions[i] == 1:
            color = 'red'  # Sound
        else:
            color = 'blue'  # Silence

        start = i * PLOT_CHUNK_SIZE
        end = (i + 1) * PLOT_CHUNK_SIZE
        ax.plot(x_line_space[start:end], plot_data[start:end], color=color, alpha=0.8)

    # Add threshold line
    ax.axhline(y=VOLUME_THRESHOLD, color='yellow', linestyle='--', alpha=0.7, label=f'Threshold: {VOLUME_THRESHOLD}')
    ax.axhline(y=-VOLUME_THRESHOLD, color='yellow', linestyle='--', alpha=0.7)

    ax.set_facecolor(face_color)
    ax.set_ylim(y_lim)
    ax.set_xlabel('Samples')
    ax.set_ylabel('Amplitude')
    ax.legend()

    # Update title with current stats
    fig.suptitle(
        f'Sound Segments: {SOUND_COUNTER}  Silence Segments: {SILENCE_COUNTER}  Current Volume: {current_volume:.1f}\n'
        f'Threshold: {VOLUME_THRESHOLD} (↑↓ to adjust by 50, +/- to adjust by 10)\n'
        f'[SPACE] to quit, [R] to reset counters  (Red = Sound, Blue = Silence)',
        fontsize=10
    )

    plt.draw()
    plt.pause(0.01)


if __name__ == '__main__':
    print("Starting volume-based audio detector...")
    print("Controls:")
    print("  ↑/↓ arrows: Adjust threshold by ±50")
    print("  +/-: Fine adjust threshold by ±10")
    print("  R: Reset counters")
    print("  SPACE: Quit")
    print(f"Initial threshold: {VOLUME_THRESHOLD}")

    audio = SharedAudioResource(chunk_size=CHUNK_SIZE, format=FORMAT, channels=CHANNELS,
                                rate=RATE, device_index=DEVICE_INDEX)
    classifier = VolumeBasedClassifier(VOLUME_THRESHOLD)

    try:
        while running:
            start_time = time.time()

            # Read audio buffer
            buffer = audio.read()
            if buffer is None:
                print("Iteration error: buffer is None")
                time.sleep(REFRESH_TIME)
                continue

            # Classify based on volume
            prediction, volume = classifier.classify(buffer)

            # Print current status
            status = "SOUND" if prediction == 1 else "SILENCE"
            print(f"Status: {status:7} | Volume: {volume:6.1f} | Threshold: {VOLUME_THRESHOLD}")

            # Update plot
            update_plot(buffer, prediction, volume)

            # Print iteration time for debugging
            iteration_time = time.time() - start_time
            if iteration_time > REFRESH_TIME * 1.5:  # Only print if significantly slower
                print(f"Iteration time: {iteration_time:.3f}s")

    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        # Clean up
        audio.close()
        plt.close(fig)
        print("Audio stream closed.")
        print(f"Final stats - Sound segments: {SOUND_COUNTER}, Silence segments: {SILENCE_COUNTER}")
