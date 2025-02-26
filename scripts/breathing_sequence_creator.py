import os
import random
from pydub import AudioSegment
import csv
from collections import deque

NUM_SEQUENCES = 300  # Number of sequences to generate
NUM_SEGMENTS = 20  # Number of phases in each sequence

# Silence duration range (in ms)
MIN_SILENCE = 300  # 0.3 sec
MAX_SILENCE = 1500  # 1.5 sec
PHASES = ['inhale', 'exhale', 'silence']

# Desired final sequence duration in milliseconds and calculated sample count
FINAL_DURATION_MS = 30000  # 30 seconds
TARGET_FRAME_RATE = 44100
FINAL_SAMPLES = int((FINAL_DURATION_MS / 1000.0) * TARGET_FRAME_RATE)

# Paths to recordings
exhale_folder = '../data/raw/person1/manual/nose/exhale'
inhale_folder = '../data/raw/person1/manual/nose/inhale'
silence_folder = '../data-ours/silence'


def load_recordings(folder, min_duration=0):
    """
    Loads audio recordings from the specified folder.
    Converts each recording to mono and sets the frame rate to 44.1 kHz.

    Args:
        folder: path to the folder containing recordings.
        min_duration: minimum duration (in ms) for the recording to be loaded.

    Returns:
        List of AudioSegment objects.
    """
    recordings = []
    for filename in os.listdir(folder):
        if filename.endswith('.wav'):
            file_path = os.path.join(folder, filename)
            recording = AudioSegment.from_wav(file_path)
            # Convert to mono
            recording = recording.set_channels(1)
            # Set frame rate to 44.1 kHz
            recording = recording.set_frame_rate(TARGET_FRAME_RATE)
            if len(recording) >= min_duration:
                recordings.append(recording)
    return recordings


# Load recordings from each folder
exhale_recordings = load_recordings(exhale_folder)
inhale_recordings = load_recordings(inhale_folder)
silence_recordings = load_recordings(silence_folder, MIN_SILENCE)


def create_sequence_with_rules(num_segments):
    """
    Creates an audio sequence by concatenating clips following these rules:
      - Enforce that, in the last 3 segments, all phases occur at least once.
      - Allow a maximum of 2 consecutive segments of the same phase.

    Each phase is labeled with its phase code (0: exhale, 1: inhale, 2: silence) along
    with the starting and ending sample indices.

    Args:
        num_segments: number of segments (phases) to generate.

    Returns:
        sequence: An AudioSegment containing the concatenated clips.
        labels: A list of tuples (phase_code, start_sample, end_sample).
    """
    sequence = AudioSegment.silent(duration=0, frame_rate=TARGET_FRAME_RATE)
    labels = []

    last_phases = deque(maxlen=3)  # Track the last 3 phases
    consecutive_count = 0
    prev_phase = None

    # Track the current sample index in the sequence
    current_sample = 0

    for _ in range(num_segments):
        # Determine missing phases among the last 3 segments
        missing_phases = []
        if len(last_phases) >= 3:
            missing_phases = [p for p in PHASES if p not in last_phases]

        # List of available phases (only if recordings are available)
        available = []
        if inhale_recordings:
            available.append('inhale')
        if exhale_recordings:
            available.append('exhale')
        if silence_recordings:
            available.append('silence')

        # Force the missing phase if possible
        forced_phases = [p for p in missing_phases if p in available]

        # Limit: maximum 2 consecutive segments of the same phase
        if consecutive_count >= 2 and prev_phase in available:
            available.remove(prev_phase)

        # Choose phase, prioritizing forced phases
        phase = None
        if forced_phases:
            phase = random.choice(forced_phases)
        elif available:
            phase = random.choice(available)

        if not phase:
            continue

        # Select an audio clip for the chosen phase
        if phase == 'inhale':
            clip = random.choice(inhale_recordings)
        elif phase == 'exhale':
            clip = random.choice(exhale_recordings)
        elif phase == 'silence':
            # For silence, take a random segment from a silence clip
            base_clip = random.choice(silence_recordings)
            clip_duration = random.randint(MIN_SILENCE, MAX_SILENCE)
            clip = base_clip[:clip_duration]

        # Append the selected clip to the sequence
        sequence += clip

        # Calculate the number of samples in the clip
        # clip duration (in seconds) = len(clip) / 1000.0
        # number of samples = duration * frame_rate
        num_samples = int((len(clip) / 1000.0) * clip.frame_rate)

        # Determine the phase code: 0 for exhale, 1 for inhale, 2 for silence
        if phase == 'exhale':
            phase_code = 0
        elif phase == 'inhale':
            phase_code = 1
        else:
            phase_code = 2

        start_sample = current_sample
        end_sample = current_sample + num_samples - 1
        labels.append((phase_code, start_sample, end_sample))

        # Update the current sample index
        current_sample += num_samples

        # Update consecutive phase counter
        if phase == prev_phase:
            consecutive_count += 1
        else:
            consecutive_count = 1
            prev_phase = phase

        last_phases.append(phase)

    return sequence, labels


def adjust_labels_for_final_duration(labels):
    """
    Adjusts the labels based on the final sequence sample length.
    If a label extends beyond the final sample count, it is trimmed.
    Labels starting at or after the final sample count are discarded.

    Args:
        labels: list of tuples (phase_code, start_sample, end_sample).

    Returns:
        adjusted_labels: list of adjusted label tuples.
    """
    adjusted_labels = []
    for phase_code, start_sample, end_sample in labels:
        if start_sample >= FINAL_SAMPLES:
            # This label starts after the final duration, so skip it.
            continue
        if end_sample >= FINAL_SAMPLES:
            # Adjust the end_sample to the final sample.
            end_sample = FINAL_SAMPLES - 1
        adjusted_labels.append((phase_code, start_sample, end_sample))
    return adjusted_labels


def finalize_sequence(sequence, labels):
    """
    Adjusts the final sequence to be exactly 30 seconds long.
    If the sequence is longer than 30 seconds, it is trimmed.
    If it is shorter, silence is appended.
    The labels are updated accordingly.

    Args:
        sequence: AudioSegment representing the sequence.
        labels: list of tuples (phase_code, start_sample, end_sample).

    Returns:
        final_sequence: AudioSegment of exactly 30 seconds.
        final_labels: Adjusted labels corresponding to the final sequence.
    """
    if len(sequence) > FINAL_DURATION_MS:
        # Trim the sequence to 30 seconds
        final_sequence = sequence[:FINAL_DURATION_MS]
    elif len(sequence) < FINAL_DURATION_MS:
        # Pad with silence if the sequence is shorter than 30 seconds
        pad_duration = FINAL_DURATION_MS - len(sequence)
        pad = AudioSegment.silent(duration=pad_duration, frame_rate=TARGET_FRAME_RATE)
        final_sequence = sequence + pad
    else:
        final_sequence = sequence

    # Adjust labels based on the final sample count
    final_labels = adjust_labels_for_final_duration(labels)
    return final_sequence, final_labels


def save_sequence_and_labels(sequence, labels, sequence_id, output_folder="data-seq"):
    """
    Saves the audio sequence and corresponding labels into the specified folder.

    Args:
        sequence: AudioSegment to be saved.
        labels: List of label tuples.
        sequence_id: Identifier for the sequence (used in filenames).
        output_folder: Destination folder.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Save the audio sequence
    audio_path = os.path.join(output_folder, f"ours{sequence_id}.wav")
    sequence.export(audio_path, format="wav")

    # Save the labels to a CSV file
    csv_path = os.path.join(output_folder, f"ours{sequence_id}.csv")
    with open(csv_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["phase_code", "start_sample", "end_sample"])
        for phase_code, start_sample, end_sample in labels:
            writer.writerow([phase_code, start_sample, end_sample])


def main():
    for i in range(NUM_SEQUENCES):
        # Create the initial sequence and labels based on the segments
        sequence, labels = create_sequence_with_rules(NUM_SEGMENTS)
        # Adjust the sequence so that it is exactly 30 seconds long, and update labels accordingly
        final_sequence, final_labels = finalize_sequence(sequence, labels)
        save_sequence_and_labels(final_sequence, final_labels, i)


if __name__ == "__main__":
    main()
