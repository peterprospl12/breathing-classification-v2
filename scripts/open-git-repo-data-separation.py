import os
from pydub import AudioSegment

# Define paths
labels_dir = '../labels'
audio_dir = '../audio'
output_dir = '../data-2'
categories = {0: 'silence', 1: 'inhale', 2: 'exhale', 3: 'inhale', 4: 'exhale'}

# Create output directories
os.makedirs(output_dir, exist_ok=True)
for category in {'silence', 'inhale', 'exhale'}:
    os.makedirs(os.path.join(output_dir, category), exist_ok=True)

# Initialize counters for each category
counters = {'silence': 1, 'inhale': 1, 'exhale': 1}

# Process each label file
for label_file in os.listdir(labels_dir):
    if label_file.endswith('.txt'):
        audio_file = os.path.join(audio_dir, label_file.replace('.txt', '.wav'))
        if not os.path.exists(audio_file):
            continue

        # Load audio file
        audio = AudioSegment.from_wav(audio_file)

        # Read label file
        with open(os.path.join(labels_dir, label_file), 'r') as f:
            for line in f:
                start, end, label = line.strip().split()
                start = float(start) * 1000  # Convert to milliseconds
                end = float(end) * 1000  # Convert to milliseconds
                label = int(label)

                if label in categories:
                    category = categories[label]
                    segment = audio[start:end]
                    output_path = os.path.join(output_dir, category, f"{category}{counters[category]}.wav")
                    segment.export(output_path, format='wav')
                    counters[category] += 1