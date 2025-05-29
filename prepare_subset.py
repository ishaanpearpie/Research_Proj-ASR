import os
import pandas as pd
import shutil
from pathlib import Path

def prepare_one_hour_subset(input_dir, output_dir, max_duration=3600):  # 3600 seconds = 1 hour
    """
    Prepare a one-hour subset of the dataset for training.
    """
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "clips"), exist_ok=True)
    
    # Read the transcriptions
    df = pd.read_csv(os.path.join(input_dir, "transcriptions.csv"))
    
    # Sort by duration to get the most files possible within one hour
    df = df.sort_values('duration_seconds')
    
    # Select files until we reach one hour
    total_duration = 0
    selected_files = []
    
    for _, row in df.iterrows():
        if total_duration + row['duration_seconds'] <= max_duration:
            selected_files.append(row)
            total_duration += row['duration_seconds']
        else:
            break
    
    # Create subset dataframe
    subset_df = pd.DataFrame(selected_files)
    
    # Copy selected files
    for _, row in subset_df.iterrows():
        src = os.path.join(input_dir, "clips", row['filename'])
        dst = os.path.join(output_dir, "clips", row['filename'])
        shutil.copy2(src, dst)
    
    # Save subset transcriptions
    subset_df.to_csv(os.path.join(output_dir, "transcriptions.csv"), index=False)
    
    print(f"\nSubset preparation complete:")
    print(f"• Selected {len(selected_files)} files")
    print(f"• Total duration: {total_duration/3600:.2f} hours")
    print(f"• Files saved to: {output_dir}")

if __name__ == "__main__":
    input_dir = os.path.expanduser("~/Downloads/ASR/Datasets")
    output_dir = os.path.expanduser("~/Downloads/ASR/Datasets/one_hour_subset")
    
    prepare_one_hour_subset(input_dir, output_dir) 