import numpy as np
import matplotlib.pyplot as plt
import wave
import csv
import onnxruntime as ort
from matplotlib.lines import Line2D
import os


def preprocess_audio(audio_data: np.ndarray, audio_length: int = 13230) -> np.ndarray:
    audio_float = audio_data.astype(np.float32) / 32768.0

    if len(audio_float) > audio_length:
        audio_float = audio_float[:audio_length]
    elif len(audio_float) < audio_length:
        padding = np.zeros(audio_length - len(audio_float), dtype=np.float32)
        audio_float = np.concatenate([audio_float, padding])

    return np.expand_dims(audio_float, axis=0)  # Add batch dimension


class AudioClassifier:
    def __init__(self, model_path):
        self.ort_session = ort.InferenceSession(model_path)
        self.input_name = self.ort_session.get_inputs()[0].name

    def predict(self, audio_data: np.ndarray) -> int:
        try:
            processed_audio = preprocess_audio(audio_data)
            ort_inputs = {self.input_name: processed_audio}
            ort_outputs = self.ort_session.run(None, ort_inputs)

            logits = ort_outputs[0]
            exp_logits = np.exp(logits)
            probs = exp_logits / np.sum(exp_logits, axis=2, keepdims=True)

            preds = np.argmax(probs[0], axis=1)
            predicted_class = int(np.bincount(preds).argmax())

            return predicted_class
        except Exception as e:
            print(f"Error while predicting: {e}")
            return 2


def read_audio_file(wav_path: str) -> tuple[np.ndarray, int]:
    with wave.open(wav_path, 'rb') as wf:
        n_frames = wf.getnframes()
        audio_data = np.frombuffer(wf.readframes(n_frames), dtype=np.int16)
        frame_rate = wf.getframerate()
    return audio_data, frame_rate


def read_labels(csv_path: str) -> list[dict[str, int | str]]:
    labels = []
    with open(csv_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip csv header
        for row in reader:
            class_name, start_sample, end_sample = row
            labels.append({
                'class': class_name,
                'start': int(start_sample),
                'end': int(end_sample)
            })
    return labels


def get_color_for_class(class_name: str | int) -> str:
    if class_name in ('inhale', 1):
        return 'red'
    elif class_name in ('exhale', 0):
        return 'green'
    return 'blue'


def generate_comparison_plots(
    wav_path: str,
    csv_path: str,
    model_path: str,
    output_path: str,
    plot_name: str,
    chunk_size: int = 13230
) -> None:
    # Read an audio file and labels
    audio_data, sample_rate = read_audio_file(wav_path)
    labels = read_labels(csv_path)
    classifier = AudioClassifier(model_path)

    print(f"Read audio: {len(audio_data)} samples, {len(audio_data) / sample_rate:.2f} seconds")
    print(f"Read {len(labels)} labels from CSV file")

    # Prepare plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle(plot_name)

    time = np.arange(len(audio_data)) / sample_rate

    # Plot 1: CSV labels
    ax1.set_title('Train data')
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Amplitude')

    for label in labels:
        start_idx = label['start']
        end_idx = label['end']
        color = get_color_for_class(label['class'])

        if start_idx < len(audio_data) and end_idx <= len(audio_data):
            segment_time = time[start_idx:end_idx]
            segment_data = audio_data[start_idx:end_idx]
            ax1.plot(segment_time, segment_data, color=color)

    # Plot 2: classificator predictions
    ax2.set_title('Classification result')
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Amplitude')

    for i in range(0, len(audio_data), chunk_size):
        end = min(i + chunk_size, len(audio_data))
        chunk = audio_data[i:end]

        # Skip too short fragments
        if len(chunk) < 1000:
            continue

        prediction = classifier.predict(chunk)

        color = get_color_for_class(prediction)
        segment_time = time[i:end]
        ax2.plot(segment_time, chunk, color=color)

    # Legend
    custom_lines = [
        Line2D([0], [0], color='red', lw=2),
        Line2D([0], [0], color='green', lw=2),
        Line2D([0], [0], color='blue', lw=2)
    ]
    fig.legend(custom_lines, ['Inhale', 'Exhale', 'Silence'],
               loc='lower center', ncol=3)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.savefig(output_path)
    print(f"Plots saved to file: {output_path}")


if __name__ == "__main__":
    raw_folder = "raw"
    label_folder = "label"
    model_path = "breath_classifier_model_audio_input.onnx"
    output_folder = "plots"

    os.makedirs(output_folder, exist_ok=True)

    wav_files = [f for f in os.listdir(raw_folder) if f.endswith('.wav')]

    print(f"Found {len(wav_files)} WAV files")

    for wav_file in wav_files:
        base_name = os.path.splitext(wav_file)[0]  # Name without extension
        csv_file = f"{base_name}.csv"
        csv_path = os.path.join(label_folder, csv_file)

        # Check if the corresponding CSV file exists
        if not os.path.exists(csv_path):
            print(f"No labels for {wav_file}, skipping")
            continue

        wav_path = os.path.join(raw_folder, wav_file)
        output_path = os.path.join(output_folder, f"{base_name}.png")

        # Create plot title according to filename
        # Example: "Piotr_nose_medium_2025-06-11_23-01-01" -> "Piotr, nos, medium"
        parts = base_name.split('_')
        if len(parts) >= 3:
            plot_name = f"Test data vs model: {parts[0]}, {parts[1]}, {parts[2]}"
        else:
            plot_name = f"Test data vs model: {base_name}"

        print(f"Process: {base_name}")
        generate_comparison_plots(wav_path, csv_path, model_path, output_path, plot_name)

    print(f"Ended plot generating. Plots in: {output_folder}")