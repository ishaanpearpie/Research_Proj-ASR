import os
import torch
import torchaudio
import numpy as np
from datasets import Dataset
from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor
)
from dataclasses import dataclass
from typing import Dict, List, Union
import pandas as pd
from jiwer import wer, cer
from tqdm import tqdm

# Configuration
class Config:
    model_dir = os.path.expanduser("~/Downloads/ASR/output/final_model")
    data_dir = os.path.expanduser("~/Downloads/ASR/Datasets/one_hour_subset")
    sampling_rate = 16000
    batch_size = 8

config = Config()

def prepare_dataset(data_dir: str):
    """Load and prepare the dataset."""
    # Read the transcription file
    df = pd.read_csv(os.path.join(data_dir, "transcriptions.csv"))
    # Drop rows with missing, empty, or NaN transcriptions
    df = df[df["transcription"].notna() & (df["transcription"].astype(str).str.strip() != "")]
    print(f"Filtered dataset size: {len(df)} rows")
    
    def load_audio(filename):
        audio_path = os.path.join(data_dir, "clips", filename)
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample if necessary
        if sample_rate != config.sampling_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, config.sampling_rate)
            waveform = resampler(waveform)
        
        return waveform.squeeze().numpy()

    # Create dataset
    dataset_dict = {
        "file_id": df["filename"].tolist(),
        "text": df["transcription"].tolist(),
        "audio": [load_audio(filename) for filename in tqdm(df["filename"])]
    }
    
    return Dataset.from_dict(dataset_dict)

def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)
    
    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
    
    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(pred.label_ids, group_tokens=True)
    
    cer_metric = cer(label_str, pred_str)
    wer_metric = wer(label_str, pred_str)
    
    return {"cer": cer_metric, "wer": wer_metric}

if __name__ == "__main__":
    print("Loading model and processor...")
    processor = Wav2Vec2Processor.from_pretrained(config.model_dir)
    model = Wav2Vec2ForCTC.from_pretrained(config.model_dir)
    model.eval()
    
    print("Loading and preparing dataset...")
    dataset = prepare_dataset(config.data_dir)
    test_dataset = dataset.train_test_split(test_size=0.1)["test"]
    
    print("\nEvaluating model...")
    all_predictions = []
    all_labels = []
    
    # Process in batches
    for i in tqdm(range(0, len(test_dataset), config.batch_size)):
        batch = test_dataset[i:i + config.batch_size]
        
        # Prepare inputs
        input_values = processor(
            batch["audio"],
            sampling_rate=config.sampling_rate,
            return_tensors="pt",
            padding=True
        ).input_values
        
        # Get predictions
        with torch.no_grad():
            logits = model(input_values).logits
            predicted_ids = torch.argmax(logits, dim=-1)
        
        # Decode predictions
        transcription = processor.batch_decode(predicted_ids)
        all_predictions.extend(transcription)
        all_labels.extend(batch["text"])
    
    # Calculate metrics
    cer_score = cer(all_labels, all_predictions)
    wer_score = wer(all_labels, all_predictions)
    
    print("\nEvaluation Results:")
    print(f"Character Error Rate (CER): {cer_score:.4f}")
    print(f"Word Error Rate (WER): {wer_score:.4f}")
    
    # Print some example predictions
    print("\nExample Predictions:")
    for i in range(min(5, len(all_predictions))):
        print(f"\nReference: {all_labels[i]}")
        print(f"Prediction: {all_predictions[i]}") 