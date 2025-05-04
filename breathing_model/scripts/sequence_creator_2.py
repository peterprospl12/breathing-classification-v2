import os
import random
import csv
from collections import deque, defaultdict
import math
import statistics 

from pydub import AudioSegment
# pylint: disable=R0914,R0912,R0915

NUM_SEQUENCES = 100
NUM_SEGMENTS = 6 

MIN_SILENCE = 300
MAX_SILENCE = 1500 
PHASES = ['inhale', 'exhale', 'silence']
PHASE_CODES = {'exhale': 0, 'inhale': 1, 'silence': 2}
CODE_PHASES = {v: k for k, v in PHASE_CODES.items()}

FINAL_DURATION_MS = 10000
TARGET_FRAME_RATE = 44100
FINAL_SAMPLES = int((FINAL_DURATION_MS / 1000.0) * TARGET_FRAME_RATE)

exhale_folder = '../data-raw/exhale/'
inhale_folder = '../data-raw/inhale/'
silence_folder = '../data-raw/silence/'

def load_recordings(folder, min_duration=0):
    """
    Loads audio recordings from a specified folder.
    Converts each recording to mono and sets the sampling rate to 44.1 kHz.

    Args:
        folder: Path to the folder containing the recordings.
        min_duration: Minimum duration (in ms) of the recording to be loaded.

    Returns:
        List of AudioSegment objects.
    """
    recordings = []
    print(f"Loading recordings from: {folder}")
    if not os.path.isdir(folder):
        print(f"Warning: Folder does not exist: {folder}")
        return recordings
    for filename in os.listdir(folder):
        if filename.endswith('.wav'):
            file_path = os.path.join(folder, filename)
            try:
                recording = AudioSegment.from_wav(file_path)
                recording = recording.set_channels(1)
                recording = recording.set_frame_rate(TARGET_FRAME_RATE)
                if len(recording) >= min_duration:
                    recordings.append(recording)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
    print(f"Loaded {len(recordings)} recordings.")
    return recordings

def calculate_average_duration(recordings):
    """Calculates the average duration of recordings in the list (in ms)."""
    if not recordings:
        return 0
    durations = [len(rec) for rec in recordings]
    return statistics.mean(durations) if durations else 0

def choose_phase_with_balancing(available_phases, total_durations_ms, current_target_duration_ms):
    """
    Selects a phase from the available ones, favoring those with a smaller total duration.

    Args:
        available_phases: List of available phases (strings: 'inhale', 'exhale', 'silence').
        total_durations_ms: Dictionary storing the total duration (in ms) for each phase
                            generated so far across all sequences.
        current_target_duration_ms: Total duration generated so far across all phases.

    Returns:
        Selected phase (string).
    """
    if not available_phases:
        return None

    if current_target_duration_ms == 0 or all(total_durations_ms.get(p, 0) == 0 for p in available_phases):
        return random.choice(available_phases)

    weights = []
    target_proportion = 1.0 / len(PHASES)
    epsilon = 1e-6 

    # Debug print
    # print("Balancing Info:")
    # print(f"  Total Duration So Far: {current_target_duration_ms:.2f} ms")
    # for p in PHASES:
    #     dur = total_durations_ms.get(p, 0)
    #     prop = dur / current_target_duration_ms if current_target_duration_ms > 0 else 0
    #     print(f"  Phase: {p}, Duration: {dur:.2f} ms, Proportion: {prop:.3f}")

    for phase in available_phases:
        current_duration = total_durations_ms.get(phase, 0)
        current_proportion = current_duration / (current_target_duration_ms + epsilon)

        # Enhanced weight - inverse square of the proportion to strongly favor underrepresented phases
        # Adding a small value to current_proportion prevents division by zero and gives a chance
        # even to phases that have temporarily exceeded the target.
        weight = 1.0 / (current_proportion + 0.05)**2 # Use power of 2 for a stronger effect

        # Alternative: Linear deficit (may be less aggressive)
        # weight = max(0.01, target_proportion - current_proportion) + 0.1

        weights.append(weight)
        # print(f"    Available: {phase}, Proportion: {current_proportion:.3f}, Weight: {weight:.3f}")


    total_weight = sum(weights)
    if total_weight == 0 or math.isnan(total_weight) or math.isinf(total_weight):
         # print(f"    Fallback: Zero, NaN or Inf total weight ({total_weight}). Choosing uniformly.")
         return random.choice(available_phases)

    try:
        chosen_phase = random.choices(available_phases, weights=weights, k=1)[0]
        # print(f"    Chosen phase by weight: {chosen_phase}")
        return chosen_phase
    except ValueError as e:
        print(f"    Fallback: Error in random.choices ({e}). Choosing uniformly.")
        return random.choice(available_phases)


def adjust_labels_for_final_duration(labels):
    """
    Adjusts labels based on the final number of samples in the sequence.
    If a label exceeds the final number of samples, it is trimmed.
    Labels starting at or after the final number of samples are discarded.

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
        # Trim the end of the label if it exceeds FINAL_SAMPLES
        adjusted_end_sample = min(end_sample, FINAL_SAMPLES - 1)

        # Ensure the start is less than or equal to the end after trimming
        if start_sample <= adjusted_end_sample:
             adjusted_labels.append((phase_code, start_sample, adjusted_end_sample))
    return adjusted_labels

def finalize_sequence(sequence, labels):
    """
    Adjusts the final sequence to have exactly 30 seconds duration.
    If the sequence is longer than 30 seconds, it is trimmed.
    If it is shorter, silence is added.
    Labels are updated accordingly.

    Args:
        sequence: AudioSegment representing the sequence.
        labels: list of tuples (phase_code, start_sample, end_sample).

    Returns:
        final_sequence: AudioSegment with a duration of exactly 30 seconds.
        final_labels: Adjusted labels corresponding to the final sequence.
    """
    final_sequence = sequence
    padding_duration_ms = 0

    if len(sequence) > FINAL_DURATION_MS:
        # Trim the sequence to 30 seconds
        final_sequence = sequence[:FINAL_DURATION_MS]
    elif len(sequence) < FINAL_DURATION_MS:
        # Add silence if the sequence is shorter than 30 seconds
        padding_duration_ms = FINAL_DURATION_MS - len(sequence)
        pad = AudioSegment.silent(
            duration=padding_duration_ms, frame_rate=TARGET_FRAME_RATE)
        final_sequence = sequence + pad
    # else: # If it is exactly 30 seconds, do nothing

    # Adjust labels based on the final number of samples
    final_labels = adjust_labels_for_final_duration(labels)

    # If silence was added, add a label for this silence at the end
    if padding_duration_ms > 0:
        start_sample_padding = int((len(sequence) / 1000.0) * TARGET_FRAME_RATE)  # Start of silence = end of the original sequence
        end_sample_padding = FINAL_SAMPLES - 1  # End of silence = end of the final sequence
        # Ensure we only add if there is space and duration > 0
        if start_sample_padding <= end_sample_padding:
             final_labels.append((PHASE_CODES['silence'], start_sample_padding, end_sample_padding))

    # Sort labels in case the added silence disrupted the order
    final_labels.sort(key=lambda x: x[1])

    return final_sequence, final_labels


def save_sequence_and_labels(sequence, labels, sequence_id, output_folder="data-train"):
    """
    Saves the audio sequence and its corresponding labels in the specified folder.

    Args:
        sequence: AudioSegment to save.
        labels: List of label tuples.
        sequence_id: Sequence identifier (used in file names).
        output_folder: Target folder.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: {output_folder}")

    audio_path = os.path.join(output_folder, f"ours{sequence_id}.wav")
    try:
        sequence.export(audio_path, format="wav")
        except Exception as e:
        print(f"Error while saving audio file {audio_path}: {e}")
        return False 

        csv_path = os.path.join(output_folder, f"ours{sequence_id}.csv")
        try:
        with open(csv_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["phase_code", "start_sample", "end_sample"])
            labels.sort(key=lambda x: x[1])
            for phase_code, start_sample, end_sample in labels:
             if start_sample <= end_sample:
                 writer.writerow([phase_code, start_sample, end_sample])
             else:
                 print(f"Warning in sequence {sequence_id}: Empty or reversed label ({phase_code}, {start_sample}, {end_sample}) after finalization, skipping save.")

        except Exception as e:
        print(f"Error while saving CSV file {csv_path}: {e}")
        return False

        return True 

    def calculate_duration_ms_from_samples(start_sample, end_sample, frame_rate):
        """Calculates duration in milliseconds based on samples."""
    num_samples = end_sample - start_sample + 1
    if num_samples < 0:
        return 0
    duration_sec = num_samples / frame_rate
    return duration_sec * 1000.0



print("--- Loading recordings ---")
exhale_recordings = load_recordings(exhale_folder)
inhale_recordings = load_recordings(inhale_folder)
silence_recordings = load_recordings(silence_folder, MIN_SILENCE)
print("--- Loading completed ---")

if not exhale_recordings and not inhale_recordings and not silence_recordings:
    raise ValueError("No recordings to process. Check folder paths.")
if not silence_recordings:
     print("Warning: No silence recordings! Silence phase will not be generated.")
     PHASES.remove('silence')
     del PHASE_CODES['silence']


avg_inhale_ms = calculate_average_duration(inhale_recordings)
avg_exhale_ms = calculate_average_duration(exhale_recordings)

if avg_inhale_ms > 0 and avg_exhale_ms > 0:
     avg_breath_ms = (avg_inhale_ms + avg_exhale_ms) / 2.0
elif avg_inhale_ms > 0:
     avg_breath_ms = avg_inhale_ms
elif avg_exhale_ms > 0:
     avg_breath_ms = avg_exhale_ms
else:
     avg_breath_ms = 1000.0 
    print("Warning: No inhale/exhale recordings available to calculate average duration. Using default value of 1000ms.")

print(f"\nAverage duration of Inhale recordings: {avg_inhale_ms:.2f} ms")
print(f"Average duration of Exhale recordings: {avg_exhale_ms:.2f} ms")
print(f"==> Averaged duration of a breathing segment (used as target for silence): {avg_breath_ms:.2f} ms\n")


def create_sequence_with_rules(num_segments, total_durations_ms, current_target_duration_ms, target_silence_duration_ms):
    """
    Creates an audio sequence using rules, balancing,
    and *adjusted silence duration*.

    Args:
       num_segments: Number of segments (phases) to generate.
       total_durations_ms: Global dictionary with total duration for each phase.
       current_target_duration_ms: Global total duration so far.
       target_silence_duration_ms: Target average duration of silence segments (in ms).

    Returns:
       sequence: AudioSegment.
       labels: List of tuples (phase_code, start_sample, end_sample).
    """
    sequence = AudioSegment.silent(duration=0, frame_rate=TARGET_FRAME_RATE)
    labels = []
    last_phases = deque(maxlen=3)
    consecutive_count = 0
    prev_phase = None
    current_sample = 0

    available_recordings = {
        'inhale': inhale_recordings,
        'exhale': exhale_recordings,
        'silence': silence_recordings
    }
    active_phases = [p for p, recs in available_recordings.items() if recs]
    if not active_phases:
        print("Critical error: No active phases (recordings) available.")
        return sequence, labels

    for i in range(num_segments):
        missing_phases = []
        if len(last_phases) >= 3:
            missing_phases = [p for p in active_phases if p not in last_phases]

        possible_phases = list(active_phases)  # Start with all active phases

        # Enforce missing phase
        forced_phases = [p for p in missing_phases if p in possible_phases]

        # Restriction: max 2 consecutive identical phases
        if consecutive_count >= 2 and prev_phase in possible_phases:
            possible_phases.remove(prev_phase)

        phase = None
        if forced_phases:
            # If a forced phase is possible, choose it randomly from the forced ones
            phase = random.choice(forced_phases)
        elif possible_phases:
            # Choose from possible phases using balancing
            phase = choose_phase_with_balancing(possible_phases, total_durations_ms, current_target_duration_ms)
        else:
            # If no phase is possible (e.g., due to rules), try to choose any active phase
            if active_phases:
                phase = random.choice(active_phases)
                # print(f"  Warning: Rules prevented selection. Randomly chosen: {phase}")
            else:
                print("  Error: No possible phases to select.")
                continue  # Skip this segment

        if not phase:
            print("  Warning: Failed to select a phase, skipping segment.")
            continue

        # --- Select clip for the phase ---
        recordings_for_phase = available_recordings[phase]
        if not recordings_for_phase:
            print(f"  Critical error: No recordings available for the selected active phase {phase}!")
            continue

        clip = None
        clip_duration_ms = 0

        if phase == 'inhale' or phase == 'exhale':
            clip = random.choice(recordings_for_phase)
            clip_duration_ms = len(clip)
        elif phase == 'silence':
            base_clip = random.choice(recordings_for_phase)
            base_clip_len = len(base_clip)

            # Add randomness around the target (e.g., +/- 20%)
            target_duration = random.uniform(0.8 * target_silence_duration_ms,
                                             1.2 * target_silence_duration_ms)

            # Limit by MIN, MAX, and the length of the base clip
            effective_max_silence = min(MAX_SILENCE, base_clip_len)
            # First, limit the target by MAX and the clip length
            target_duration = min(target_duration, effective_max_silence)
            # Then ensure it is at least MIN_SILENCE
            clip_duration_ms = max(MIN_SILENCE, target_duration)

            # Ensure we do not exceed the length of base_clip after rounding/limiting
            clip_duration_ms = int(min(clip_duration_ms, base_clip_len))

            if clip_duration_ms > 0:
                clip = base_clip[:clip_duration_ms]
            else:
                # Emergency situation: if the calculated duration is <= 0
                print(f"  Warning: Calculated silence duration <= 0 ({clip_duration_ms}ms).")
                # Try using MIN_SILENCE if the clip is long enough
                clip_duration_ms = int(min(max(MIN_SILENCE, 1), base_clip_len))
                if clip_duration_ms > 0:
                     clip = base_clip[:clip_duration_ms]
                else:
                     print(f"  Error: Unable to create a silence segment from base_clip of length {base_clip_len}ms.")
                     clip = None 

        if clip is None or len(clip) == 0:
            print(f"  Warning: Generated an empty clip for phase {phase}, skipping segment {i+1}.")
            continue

        sequence += clip
        num_samples = int((len(clip) / 1000.0) * clip.frame_rate)
        phase_code = PHASE_CODES[phase]
        start_sample = current_sample
        end_sample = current_sample + num_samples - 1
        labels.append((phase_code, start_sample, end_sample))
        current_sample += num_samples

        if phase == prev_phase:
            consecutive_count += 1
        else:
            consecutive_count = 1
            prev_phase = phase
        last_phases.append(phase)

    return sequence, labels


def main():
    total_durations_ms = defaultdict(float)
    sequences_generated = 0
    output_dir = "data-raw-val"

    target_silence_duration = avg_breath_ms
    if target_silence_duration <= 0:
        print("Critical error: Target silence duration is zero or negative.")
        return

    for i in range(NUM_SEQUENCES):
        print(f"\nGenerating sequence {i+1}/{NUM_SEQUENCES}...")

        current_total_duration = sum(total_durations_ms.values())

        sequence, labels = create_sequence_with_rules(
            NUM_SEGMENTS,
            total_durations_ms,
            current_total_duration,
            target_silence_duration
        )

        if len(sequence) == 0 or not labels:
            print(f"Warning: Sequence {i} is empty after the creation phase. Skipping.")
            continue

        final_sequence, final_labels = finalize_sequence(sequence, labels)

        if len(final_sequence) != FINAL_DURATION_MS:
            print(f"Critical error: Finalized sequence {i} has a length of {len(final_sequence)}ms instead of {FINAL_DURATION_MS}ms. Skipping.")
            continue

        if save_sequence_and_labels(final_sequence, final_labels, i, output_folder=output_dir):
            sequences_generated += 1
            current_seq_durations = defaultdict(float) 
            for phase_code, start_sample, end_sample in final_labels:
                 duration_ms = calculate_duration_ms_from_samples(start_sample, end_sample, TARGET_FRAME_RATE)
                 phase_name = CODE_PHASES.get(phase_code)
                 if phase_name:
                     total_durations_ms[phase_name] += duration_ms
                     current_seq_durations[phase_name] += duration_ms
                 else:
                      print(f"Warning: Unknown phase code {phase_code} in sequence {i}.")

            print(f"  Sequence {i+1} saved. Phase durations in this sequence:")
            for phase in PHASES: 
                 print(f"    {phase:<8}: {current_seq_durations.get(phase, 0):>8.2f} ms")

            if (i + 1) % 10 == 0 or (i + 1) == NUM_SEQUENCES : 
                 print(f"  --- Global progress after {i+1} sequences ---")
                 current_total = sum(total_durations_ms.values())
                 if current_total > 0:
                      for phase in PHASES:
                            dur = total_durations_ms.get(phase, 0)
                            print(f"    {phase:<8}: {dur:>10.2f} ms ({dur/current_total*100:>5.1f}%)")
                 print(f"  ---------------------------------------")


    print("\n--- Generation completed ---")
    print(f"Successfully generated and saved {sequences_generated} out of {NUM_SEQUENCES} sequences in the folder '{output_dir}'.")

    print("\n--- Final summary of total phase durations ---")
    grand_total_duration_ms = sum(total_durations_ms.values())

    if grand_total_duration_ms == 0:
        print("No data was generated (total duration = 0).")
    else:
        print(f"Total duration of all sequences: {grand_total_duration_ms / 1000.0:.2f} seconds ({grand_total_duration_ms:.2f} ms)")
        for phase in PHASES:
             dur = total_durations_ms.get(phase, 0)
             print(f"  - {phase:<8}: {dur:>10.2f} ms ({dur / grand_total_duration_ms * 100:.2f} %)")

        expected_total_duration = sequences_generated * FINAL_DURATION_MS
        print(f"\nExpected total duration ({sequences_generated} sequences * {FINAL_DURATION_MS} ms): {expected_total_duration:.2f} ms")
        discrepancy = abs(grand_total_duration_ms - expected_total_duration)
        print(f"Difference between calculated and expected duration: {discrepancy:.2f} ms")
        if discrepancy > sequences_generated * 2: 
            print("Warning: Significant difference between calculated and expected total duration!")

if __name__ == "__main__":
    required_folders = [exhale_folder, inhale_folder, silence_folder]
    abort = False
    for folder in required_folders:
        if not os.path.isdir(folder):
            print(f"ERROR: Data folder '{folder}' does not exist!")
            abort = True
    if abort:
        print("Script execution aborted due to missing data folders.")
    else:
        main()