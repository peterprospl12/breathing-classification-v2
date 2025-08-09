import numpy as np
import pyaudio
import matplotlib.pyplot as plt
import time
from collections import deque
from model.lib import SharedAudioResource

#############################################
# Settings and constants
#############################################
REFRESH_TIME = 0.2
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
DEVICE_INDEX = 0
CHUNK_SIZE = int(RATE * REFRESH_TIME)

# Volume threshold for silence detection
VOLUME_THRESHOLD = 500

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
        self.threshold = new_threshold
        print(f"Volume threshold updated to: {self.threshold}")

    def classify(self, audio_data):
        # Use pre-computed squares to avoid repeated calculations
        rms_volume = np.sqrt(np.mean(np.square(audio_data.astype(np.float32))))

        return (1 if rms_volume > self.threshold else 0), rms_volume

#############################################
# Plot configuration with segment coloring
#############################################
class OptimizedPlotter:
    def __init__(self):
        self.PLOT_TIME_HISTORY = 3  # seconds to show
        self.max_samples = RATE * self.PLOT_TIME_HISTORY
        self.max_predictions = int(self.PLOT_TIME_HISTORY / REFRESH_TIME)

        # Use deques for efficient rolling
        self.plot_data = deque([0] * self.max_samples, maxlen=self.max_samples)
        self.predictions = deque([0] * self.max_predictions, maxlen=self.max_predictions)

        # Setup plot
        plt.ion()  # Interactive mode
        self.fig, self.ax = plt.subplots(1, 1, figsize=(12, 6))

        # Store line objects for each segment
        self.line_objects = []

        # Create x-axis data
        self.x_data = np.arange(self.max_samples)

        # Create threshold lines
        self.threshold_line_pos = self.ax.axhline(y=VOLUME_THRESHOLD, color='yellow', linestyle='--', alpha=0.7)
        self.threshold_line_neg = self.ax.axhline(y=-VOLUME_THRESHOLD, color='yellow', linestyle='--', alpha=0.7)

        # Set initial properties
        self.ax.set_facecolor((0, 0, 0))
        self.ax.set_ylim(-1500, 1500)
        self.ax.set_xlim(0, self.max_samples)
        self.ax.set_xlabel('Samples')
        self.ax.set_ylabel('Amplitude')

        # Connect keyboard events
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

        # Title text object for efficient updates
        self.title_text = self.fig.suptitle('', fontsize=10)

    def on_key(self, event):
        global running, SOUND_COUNTER, SILENCE_COUNTER, VOLUME_THRESHOLD, classifier

        if event.key == ' ':
            running = False
        elif event.key == 'r':
            SOUND_COUNTER = 0
            SILENCE_COUNTER = 0
            print("Counters reset")
        elif event.key == 'up':
            VOLUME_THRESHOLD += 50
            classifier.set_threshold(VOLUME_THRESHOLD)
            self.update_threshold_lines()
        elif event.key == 'down':
            VOLUME_THRESHOLD = max(50, VOLUME_THRESHOLD - 50)
            classifier.set_threshold(VOLUME_THRESHOLD)
            self.update_threshold_lines()

    def update_threshold_lines(self):
        self.threshold_line_pos.set_ydata([VOLUME_THRESHOLD, VOLUME_THRESHOLD])
        self.threshold_line_neg.set_ydata([-VOLUME_THRESHOLD, -VOLUME_THRESHOLD])

    def update(self, frames, prediction, volume):
        global SOUND_COUNTER, SILENCE_COUNTER

        # Update data efficiently
        self.plot_data.extend(frames)
        self.predictions.append(prediction)

        # Update counters
        if prediction == 1:
            SOUND_COUNTER += 1
        else:
            SILENCE_COUNTER += 1

        # Clear previous plot
        self.ax.clear()

        # Recreate threshold lines after clear
        self.threshold_line_pos = self.ax.axhline(y=VOLUME_THRESHOLD, color='yellow', linestyle='--', alpha=0.7, label=f'Threshold: {VOLUME_THRESHOLD}')
        self.threshold_line_neg = self.ax.axhline(y=-VOLUME_THRESHOLD, color='yellow', linestyle='--', alpha=0.7)

        # Convert data for plotting
        plot_array = np.array(self.plot_data)
        predictions_array = np.array(self.predictions)

        # Plot segments with appropriate colors
        samples_per_prediction = CHUNK_SIZE
        for i, pred in enumerate(predictions_array):
            start_idx = i * samples_per_prediction
            end_idx = min((i + 1) * samples_per_prediction, len(plot_array))

            if start_idx < len(plot_array):
                color = 'red' if pred == 1 else 'blue'
                x_segment = self.x_data[start_idx:end_idx]
                y_segment = plot_array[start_idx:end_idx]

                if len(x_segment) > 0 and len(y_segment) > 0:
                    self.ax.plot(x_segment, y_segment, color=color, alpha=0.8, linewidth=1)

        # Set properties again after clear
        self.ax.set_facecolor((0, 0, 0))
        self.ax.set_ylim(-1500, 1500)
        self.ax.set_xlim(0, self.max_samples)
        self.ax.set_xlabel('Samples')
        self.ax.set_ylabel('Amplitude')
        self.ax.legend()

        # Update title
        title_text = (f'Sound: {SOUND_COUNTER} | Silence: {SILENCE_COUNTER} | '
                     f'Volume: {volume:.1f} | Threshold: {VOLUME_THRESHOLD}\n'
                     f'[SPACE] quit | [R] reset | [↑↓] adjust threshold | Red=Sound Blue=Silence')
        self.fig.suptitle(title_text, fontsize=10)

        # Draw efficiently
        plt.draw()
        plt.pause(0.01)

if __name__ == '__main__':
    # Initialize components
    audio = SharedAudioResource(chunk_size=CHUNK_SIZE, format=FORMAT, channels=CHANNELS,
                                rate=RATE, device_index=DEVICE_INDEX)
    classifier = VolumeBasedClassifier(VOLUME_THRESHOLD)
    plotter = OptimizedPlotter()

    try:
        # Main processing loop - single thread
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
            plotter.update(buffer, prediction, volume)

            # Control timing to maintain consistent refresh rate
            elapsed = time.time() - start_time
            sleep_time = max(0, REFRESH_TIME - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        running = False
        audio.close()
        plt.close('all')
        print("Audio stream closed.")
        print(f"Final stats - Sound: {SOUND_COUNTER}, Silence: {SILENCE_COUNTER}")
