import os
import shutil
import pandas as pd
from pydub import AudioSegment
import warnings

# Suppress pydub warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Update paths to match your workspace
base_dir = os.path.expanduser("~/Downloads/ASR")
hf_clips = os.path.join(base_dir, "Datasets/clips")
hf_transcriptions = os.path.join(base_dir, "Datasets/transcriptions.csv")

# New directory structure
new_dataset = os.path.join(base_dir, "Trimmed_Dataset")
new_clips = os.path.join(new_dataset, "clips")
new_transcriptions_path = os.path.join(new_dataset, "transcriptions.csv")

# Create directories if they don't exist
os.makedirs(new_clips, exist_ok=True)
os.makedirs(new_dataset, exist_ok=True)

def get_audio_duration(file_path):
    try:
        audio = AudioSegment.from_file(file_path)
        duration = len(audio) / 1000.0  # Convert to seconds
        # Skip very short clips (less than 0.5 seconds)
        if duration < 0.5:
            print(f"  Skipping short clip: {os.path.basename(file_path)} ({duration:.2f}s)")
            return None
        return duration
    except Exception as e:
        print(f"  Error processing {os.path.basename(file_path)}: {str(e)}")
        return None

# ======== PROCESS DATASET ========
def process_dataset(clips_folder, csv_path, dataset_name):
    """Process dataset and return valid entries"""
    print(f"\nProcessing {dataset_name} dataset...")
    
    # Read transcriptions
    df = pd.read_csv(csv_path)
    
    valid_entries = []
    total, valid = 0, 0
    
    for _, row in df.iterrows():
        total += 1
        filename = row['filename']
        transcription = row['transcription']
        
        audio_path = os.path.join(clips_folder, filename)
        
        # Check if file exists
        if not os.path.exists(audio_path):
            print(f"  File not found: {filename}")
            continue
        
        # Check duration (≤8 seconds)
        duration = get_audio_duration(audio_path)
        if duration is None or duration > 8.0:
            continue
        
        valid += 1
        valid_entries.append({
            'filename': filename,
            'transcription': transcription,
            'source_path': audio_path,
            'duration': duration
        })
    
    print(f"  Found {valid}/{total} valid clips (≤8s)")
    return valid_entries

# Process dataset
all_entries = process_dataset(hf_clips, hf_transcriptions, "Telugu ASR")

# ======== COPY FILES & HANDLE DUPLICATES ========
print("\nCopying files to new dataset...")
new_records = []
filename_counter = {}
copied_count = 0

for entry in all_entries:
    src = entry['source_path']
    orig_name = entry['filename']
    base, ext = os.path.splitext(orig_name)
    
    # Handle duplicate filenames
    if orig_name in filename_counter:
        filename_counter[orig_name] += 1
        new_name = f"{base}_{filename_counter[orig_name]}{ext}"
    else:
        filename_counter[orig_name] = 0
        new_name = orig_name
    
    # Copy file to new location
    dst = os.path.join(new_clips, new_name)
    shutil.copy2(src, dst)
    copied_count += 1
    
    # Add to records
    new_records.append({
        'filename': new_name,
        'transcription': entry['transcription'],
        'duration': entry['duration']
    })

# ======== SAVE TRANSCRIPTIONS ========
df_new = pd.DataFrame(new_records)
df_new.to_csv(new_transcriptions_path, index=False)

# ======== FINAL REPORT ========
print("\nOperation completed successfully!")
print(f"Total clips processed: {len(all_entries)}")
print(f"Clips copied: {copied_count}")
print(f"New dataset location: {new_dataset}")
print(f"  - Audio clips: {new_clips}")
print(f"  - Transcriptions: {new_transcriptions_path}")

# Print duration statistics
if new_records:
    durations = [r['duration'] for r in new_records]
    print("\nDuration Statistics:")
    print(f"  Average duration: {sum(durations)/len(durations):.2f}s")
    print(f"  Min duration: {min(durations):.2f}s")
    print(f"  Max duration: {max(durations):.2f}s")