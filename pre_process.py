import os
import pandas as pd
import soundfile as sf
from datasets import load_dataset, Audio

# ===== Configuration =====
DATASET_NAME = "zsy12345/telugu-asr"
OUTPUT_PATH = os.path.expanduser("~/Downloads/ASR/Datasets")
CLIPS_DIR = os.path.join(OUTPUT_PATH, "clips")
MAX_DURATION = 40 * 3600  # 40 hours in seconds

# Create necessary directories
os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(CLIPS_DIR, exist_ok=True)

print("Analyzing dataset structure...")
try:
    test_sample = next(iter(
        load_dataset(DATASET_NAME, split="train", streaming=True)
    ))
    print("Dataset structure verified:")
    print("Sample keys:", test_sample.keys())
    print("Audio features:", list(test_sample['audio'].keys()))
except Exception as e:
    print(f"Dataset analysis failed: {str(e)}")
    exit()

print("\nLoading dataset...")
dataset = load_dataset(
    DATASET_NAME,
    split="train",
    streaming=True,
    trust_remote_code=True
).cast_column("audio", Audio(sampling_rate=16000))

metadata = []
total_duration = 0.0
file_count = 0
error_count = 0

print("\nStarting processing...")
try:
    for sample in dataset:
        if total_duration >= MAX_DURATION:
            print("Reached duration limit, stopping...")
            break
            
        try:
            audio = sample["audio"]
            array = audio["array"]
            sr = audio["sampling_rate"]
            
            duration = len(array) / sr
            
            if (total_duration + duration) > MAX_DURATION:
                continue
                
            filename = f"telugu_{file_count:06d}.wav"
            filepath = os.path.join(CLIPS_DIR, filename)
            sf.write(filepath, array, sr, subtype='PCM_16')
            
            if not os.path.exists(filepath):
                raise IOError("File not created")
                
            metadata.append({
                "filename": filename,
                "transcription": sample["sentence"].strip(),
                "duration_seconds": round(duration, 2)
            })
            
            total_duration += duration
            file_count += 1
            
            if file_count % 100 == 0:
                print(f"Processed {file_count} files ({total_duration/3600:.2f} hrs)")
                
        except Exception as e:
            error_count += 1
            print(f"Error processing file {file_count}: {str(e)}")
            continue
            
finally:
    # Save metadata
    csv_path = os.path.join(OUTPUT_PATH, "transcriptions.csv")
    pd.DataFrame(metadata).to_csv(csv_path, index=False)
    
    print("\nProcessing complete!")
    print(f"• Saved {file_count} WAV files to: {CLIPS_DIR}")
    print(f"• Total duration: {total_duration/3600:.2f} hours")
    print(f"• Errors encountered: {error_count}")
    print(f"• Metadata saved to: {csv_path}")