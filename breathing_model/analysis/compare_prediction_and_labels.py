import numpy as np
import matplotlib.pyplot as plt
import wave
import csv
import torch
import librosa
import os
from matplotlib.lines import Line2D

from breathing_model.archive.lstm.model_classes import AudioClassifierLSTM


class LSTMClassifierWrapper:
    def __init__(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = AudioClassifierLSTM()
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        self.hidden = None

    def reset_hidden(self):
        self.hidden = None

    def predict(self, audio_chunk: np.ndarray, sample_rate: int = 44100) -> int:
        if audio_chunk.dtype == np.int16:
            audio_float = audio_chunk.astype(np.float32) / 32768.0
        else:
            audio_float = audio_chunk.astype(np.float32)

        mfcc = librosa.feature.mfcc(
            y=audio_float,
            sr=sample_rate,
            n_mfcc=13
        )

        features = mfcc.mean(axis=1)

        single_data = torch.tensor(features, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs, self.hidden = self.model(single_data, self.hidden)

            logits = outputs[0, 0]
            prediction_idx = torch.argmax(logits).item()

        return prediction_idx


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
            if len(row) < 3: continue
            class_name, start_sample, end_sample = row
            labels.append({
                'class': class_name,
                'start': int(start_sample),
                'end': int(end_sample)
            })
    return labels


def get_color_for_class(class_name: str | int) -> str:
    if class_name == 'inhale' or class_name == 1:
        return 'red'
    elif class_name == 'exhale' or class_name == 0:
        return 'green'
    return 'blue'


def generate_comparison_plots(
        wav_path: str,
        csv_path: str,
        model_path: str,
        output_path: str,
        plot_name: str,
        chunk_duration: float = 0.25
) -> None:
    audio_data, sample_rate = read_audio_file(wav_path)
    labels = read_labels(csv_path)

    classifier = LSTMClassifierWrapper(model_path)

    classifier.reset_hidden()

    chunk_size = int(sample_rate * chunk_duration)

    print(f"Read audio: {len(audio_data)} samples, {len(audio_data) / sample_rate:.2f} seconds")
    print(f"Read {len(labels)} labels from CSV file")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle(plot_name)

    time_axis = np.arange(len(audio_data)) / sample_rate

    ax1.set_title('Ground Truth (CSV Labels)')
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Amplitude')

    ax1.plot(time_axis, audio_data, color='lightgray', alpha=0.5)

    for label in labels:
        start_idx = label['start']
        end_idx = label['end']
        color = get_color_for_class(label['class'])

        if start_idx < len(audio_data) and end_idx <= len(audio_data):
            segment_time = time_axis[start_idx:end_idx]
            segment_data = audio_data[start_idx:end_idx]
            ax1.plot(segment_time, segment_data, color=color)

    ax2.set_title('LSTM Model Prediction')
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Amplitude')

    for i in range(0, len(audio_data), chunk_size):
        end = min(i + chunk_size, len(audio_data))
        chunk = audio_data[i:end]

        if len(chunk) < chunk_size:
            continue

        prediction = classifier.predict(chunk, sample_rate)

        color = get_color_for_class(prediction)
        segment_time = time_axis[i:end]

        ax2.plot(segment_time, chunk, color=color)

    custom_lines = [
        Line2D([0], [0], color='red', lw=2),
        Line2D([0], [0], color='green', lw=2),
        Line2D([0], [0], color='blue', lw=2)
    ]
    fig.legend(custom_lines, ['Inhale (1)', 'Exhale (0)', 'Silence (2)'],
               loc='lower center', ncol=3)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.savefig(output_path)
    plt.close(fig)
    print(f"Plots saved to file: {output_path}")


if __name__ == "__main__":
    raw_folder = "../data/eval/raw"
    label_folder = "../data/eval/label"
    model_path = "../archive/lstm/model_lstm.pth"
    output_folder = "plots_lstm"

    os.makedirs(output_folder, exist_ok=True)

    wav_files = [f for f in os.listdir(raw_folder) if f.endswith('.wav')]

    print(f"Found {len(wav_files)} WAV files")

    for wav_file in wav_files:
        base_name = os.path.splitext(wav_file)[0]
        csv_file = f"{base_name}.csv"
        csv_path = os.path.join(label_folder, csv_file)

        if not os.path.exists(csv_path):
            print(f"No labels for {wav_file}, skipping")
            continue

        wav_path = os.path.join(raw_folder, wav_file)
        output_path = os.path.join(output_folder, f"{base_name}.png")

        parts = base_name.split('_')
        if len(parts) >= 3:
            plot_name = f"LSTM Test: {parts[0]}, {parts[1]}, {parts[2]}"
        else:
            plot_name = f"LSTM Test: {base_name}"

        print(f"Processing: {base_name}")

        try:
            generate_comparison_plots(
                wav_path,
                csv_path,
                model_path,
                output_path,
                plot_name,
                chunk_duration=0.25
            )
        except Exception as e:
            print(f"Error processing {base_name}: {e}")

    print(f"Done. Check {output_folder}")