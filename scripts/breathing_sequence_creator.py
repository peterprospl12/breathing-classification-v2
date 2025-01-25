import os
import random
from pydub import AudioSegment
import csv
from collections import deque

NUM_SEQUENCES = 10 # Num of sequences to generate
NUM_SEGMENTS = 12  # Num of PHASES in each sequence

# Silence duration range
MIN_SILENCE = 300  # 0.3 sec (in ms)
MAX_SILENCE = 1500  # 1.5 sec (in ms)
PHASES = ['inhale', 'exhale', 'silence']

# Path to recordings
exhale_folder = '../data/raw/person1/manual/nose/exhale'
inhale_folder = '../data/raw/person1/manual/nose/inhale'
silence_folder = '../data/raw/person1/manual/nose/silence'


def load_recordings(folder, min_duration=0):
    """
    Load recordings from folder
    Args:
        folder:
        min_duration:

    Returns:

    """
    recordings = []
    for filename in os.listdir(folder):
        if filename.endswith('.wav'):
            file_path = os.path.join(folder, filename)
            recording = AudioSegment.from_wav(file_path)
            if len(recording) >= min_duration:
                recordings.append(recording)
    return recordings


# Load recordings
exhale_recordings = load_recordings(exhale_folder)
inhale_recordings = load_recordings(inhale_folder)
silence_recordings = load_recordings(silence_folder, MIN_SILENCE)


def create_sequence_with_rules(num_segments):
    """
    Create sequence with rules
    Args:
        num_segments:

    Returns:
        sequence, labels

    """
    sequence = AudioSegment.silent(duration=0)
    labels = []

    last_phases = deque(maxlen=3)  # Keep track of last 3 phases
    consecutive_count = 0
    prev_phase = None

    for _ in range(num_segments):
        # Define missing phases in last 3 segments
        missing_phases = []
        if len(last_phases) >= 3:
            missing_phases = [p for p in PHASES if p not in last_phases]

        # Available phases
        available = []
        if inhale_recordings: available.append('inhale')
        if exhale_recordings: available.append('exhale')
        if silence_recordings: available.append('silence')

        # Force missing phase if exists
        forced_phases = [p for p in missing_phases if p in available]

        # Limitation: max 2 consecutive same phases
        if consecutive_count >= 2 and prev_phase in available:
            available.remove(prev_phase)

        # Choose phase with priority for forced
        phase = None
        if forced_phases:
            phase = random.choice(forced_phases)
        elif available:
            phase = random.choice(available)

        if not phase:
            continue

        # Add clip
        if phase == 'inhale':
            clip = random.choice(inhale_recordings)
        elif phase == 'exhale':
            clip = random.choice(exhale_recordings)
        elif phase == 'silence':
            clip = random.choice(silence_recordings)[:random.randint(MIN_SILENCE, MAX_SILENCE)]

        sequence += clip
        labels.append((phase, len(clip)))

        # Update consecutive count
        if phase == prev_phase:
            consecutive_count += 1
        else:
            consecutive_count = 1
            prev_phase = phase

        last_phases.append(phase)

    return sequence, labels


# Save sequence and labels
def save_sequence_and_labels(sequence, labels, sequence_id, output_folder="output"):
    """
    Save sequence and labels to output folder
    Args:
        sequence:
        labels:
        sequence_id:
        output_folder:

    Returns:

    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Save sequence
    audio_path = os.path.join(output_folder, f"seq_{sequence_id}.wav")
    sequence.export(audio_path, format="wav")

    # Save labels
    csv_path = os.path.join(output_folder, f"seq_{sequence_id}_labels.csv")
    with open(csv_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Phase", "Duration (ms)"])
        for phase, duration in labels:
            writer.writerow([phase, duration])

def main():
    for i in range(NUM_SEQUENCES):
        sequence, labels = create_sequence_with_rules(NUM_SEGMENTS)
        save_sequence_and_labels(sequence, labels, i)

if __name__ == "__main__":
    main()
