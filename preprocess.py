import os
import librosa
import soundfile as sf
import pandas as pd
from tqdm import tqdm
import numpy as np

def preprocess_audio(input_dir, output_dir, target_sr=16000):
    """
    Preprocess audio files:
    1. Convert to 16kHz sampling rate
    2. Convert to mono if stereo
    3. Normalize audio
    4. Save as wav format
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of audio files
    audio_files = [f for f in os.listdir(input_dir) if f.endswith('.mp3')]
    
    for audio_file in tqdm(audio_files, desc="Processing audio files"):
        # Load audio file
        audio_path = os.path.join(input_dir, audio_file)
        y, sr = librosa.load(audio_path, sr=target_sr)
        
        # Convert to mono if stereo
        if len(y.shape) > 1:
            y = librosa.to_mono(y)
        
        # Normalize audio
        y = librosa.util.normalize(y)
        
        # Save as wav
        output_path = os.path.join(output_dir, audio_file.replace('.mp3', '.wav'))
        sf.write(output_path, y, target_sr)

def prepare_transcription_file(input_file, output_file):
    """
    Prepare the transcription file:
    1. Remove any special characters (if needed)
    2. Convert to proper format
    """
    df = pd.read_csv(input_file, sep='\t', header=None, names=['file_id', 'text'])
    
    # Clean text if needed (add any specific cleaning steps here)
    df['text'] = df['text'].str.strip()
    
    # Save processed transcriptions
    df.to_csv(output_file, sep='\t', index=False, header=False)

if __name__ == "__main__":
    # Directory structure
    raw_data_dir = "raw_data"
    processed_data_dir = "data"
    
    # Create processed data directory
    os.makedirs(processed_data_dir, exist_ok=True)
    
    # Process audio files
    print("Processing audio files...")
    preprocess_audio(raw_data_dir, processed_data_dir)
    
    # Process transcription file
    print("Processing transcription file...")
    prepare_transcription_file(
        os.path.join(raw_data_dir, "transcriptions.txt"),
        os.path.join(processed_data_dir, "transcriptions.txt")
    )
    
    print("Preprocessing completed!") 