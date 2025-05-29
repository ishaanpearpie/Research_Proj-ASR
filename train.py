import os
import torch
import torchaudio
import numpy as np
from datasets import Dataset
from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    TrainingArguments,
    Trainer,
    Wav2Vec2Config
)
from torch.optim import AdamW
from dataclasses import dataclass
from typing import Dict, List, Union
import pandas as pd
from jiwer import wer, cer
import wandb
from tqdm import tqdm
import json
import psutil
import platform
from torch.nn.utils.rnn import pad_sequence
import datetime

# Set environment variable for MPS fallback (only affects this script)
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

def check_system_resources():
    """Check system resources and provide recommendations."""
    print("\n=== System Resources Check ===")
    
    # CPU Info
    cpu_count = psutil.cpu_count(logical=False)
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    
    print(f"CPU Cores: {cpu_count}")
    print(f"CPU Usage: {cpu_percent}%")
    print(f"Available Memory: {memory.available / (1024**3):.2f} GB")
    
    # GPU Info
    if torch.cuda.is_available():
        print("\nGPU Information:")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(i).total_memory / (1024**3):.2f} GB")
    elif torch.backends.mps.is_available():
        print("\nUsing Apple Silicon GPU (MPS)")
    else:
        print("\nNo GPU detected - Training will be slow!")
        print("\n=== Alternative Computing Resources ===")
        print("For faster training, consider using:")
        print("1. Kaggle Notebooks (Free GPU, 30 hours/week)")
        print("   - Visit: https://www.kaggle.com/notebooks")
        print("2. Google Colab (Free GPU, limited hours)")
        print("   - Visit: https://colab.research.google.com")
        print("3. RunPod (Pay-as-you-go GPU instances)")
        print("   - Visit: https://www.runpod.io")
        print("4. Vast.ai (Marketplace for GPU rentals)")
        print("   - Visit: https://vast.ai")
        print("\nRecommended minimum specs for this project:")
        print("- 16GB RAM")
        print("- NVIDIA GPU with 8GB+ VRAM")
        print("- 4+ CPU cores")

def cleanup_wandb_runs():
    """Clean up old wandb runs, keeping only the latest one."""
    api = wandb.Api()
    runs = api.runs("telugu-asr")
    if len(runs) > 1:
        # Sort runs by creation time, newest first
        runs = sorted(runs, key=lambda x: x.created_at, reverse=True)
        # Delete all runs except the latest one
        for run in runs[1:]:
            run.delete()
        print(f"Cleaned up {len(runs)-1} old wandb runs")

def cleanup_old_datasets():
    """Clean up old processed datasets, keeping only the latest one."""
    cache_dir = config.output_dir
    if os.path.exists(cache_dir):
        # Find all processed dataset directories
        dataset_dirs = [d for d in os.listdir(cache_dir) if d.startswith("processed_dataset")]
        if len(dataset_dirs) > 1:
            # Sort by modification time, newest first
            dataset_dirs.sort(key=lambda x: os.path.getmtime(os.path.join(cache_dir, x)), reverse=True)
            # Delete all but the newest one
            for old_dir in dataset_dirs[1:]:
                old_path = os.path.join(cache_dir, old_dir)
                print(f"Removing old processed dataset: {old_path}")
                import shutil
                shutil.rmtree(old_path)

# Configuration
class Config:
    model_name = "facebook/wav2vec2-base-960h"
    learning_rate = 1e-4
    epochs = 15
    hidden_dropout = 0.1
    batch_size = 6
    sampling_rate = 16000
    data_dir = "/kaggle/input/combined-dataset"
    output_dir = "/kaggle/working/output"
    device = "cuda"

config = Config()

# Initialize wandb with Kaggle-specific settings
os.environ["WANDB_API_KEY"] = "3ab5e0653513d2ec4abd11776b47d3cb515acf63"
os.environ["WANDB_MODE"] = "online"

wandb.init(
    project="telugu-asr",
    config=vars(config),
    settings=wandb.Settings(start_method="thread")
)

def prepare_dataset(data_dir: str):
    """Load and prepare the dataset."""
    # Clean up old datasets first
    cleanup_old_datasets()
    
    # Define cache path with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    cache_path = os.path.join(config.output_dir, f"processed_dataset_{timestamp}")
    
    # Check if any processed dataset exists
    existing_datasets = [d for d in os.listdir(config.output_dir) if d.startswith("processed_dataset")]
    if existing_datasets:
        # Use the most recent dataset
        latest_dataset = max(existing_datasets, key=lambda x: os.path.getmtime(os.path.join(config.output_dir, x)))
        cache_path = os.path.join(config.output_dir, latest_dataset)
        print(f"Loading cached processed dataset: {latest_dataset}")
        return Dataset.load_from_disk(cache_path)
    
    print("Processing dataset for the first time...")
    # Using the exact paths provided by the user
    csv_path = "/kaggle/input/combined-dataset/combined_dataset.csv"
    audio_clips_dir = "/kaggle/input/combined-dataset/combined_dataset/combined_dataset"

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Transcription file not found at {csv_path}. Please verify the dataset structure.")
    if not os.path.exists(audio_clips_dir) or not os.path.isdir(audio_clips_dir):
         raise FileNotFoundError(f"Audio clips directory not found at {audio_clips_dir}. Please verify the dataset structure.")

    # Read the CSV file without headers and assign column names
    df = pd.read_csv(csv_path, header=None, names=['filename', 'transcription'])
    
    # Drop rows with missing, empty, or NaN transcriptions
    df = df[df["transcription"].notna() & (df["transcription"].astype(str).str.strip() != "")]
    print(f"Filtered dataset size: {len(df)} rows")
    
    def load_audio(filename):
        # Audio files are in the specified audio_clips_dir
        audio_path = os.path.join(audio_clips_dir, filename)
        if not os.path.exists(audio_path):
             raise FileNotFoundError(f"Audio file {filename} not found in {audio_clips_dir}.")

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
    
    dataset = Dataset.from_dict(dataset_dict)
    
    # Save processed dataset
    print(f"Saving processed dataset to {cache_path}...")
    dataset.save_to_disk(cache_path)
    
    return dataset

@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        filtered_features = []
        for f in features:
            text = f.get("text")
            if text is None or (isinstance(text, float) and pd.isna(text)) or str(text).strip() == "":
                continue
            filtered_features.append(f)
        features = filtered_features
        if len(features) == 0: # Handle case where all features are filtered out
             # This might happen on the last batch if it's all invalid, which is fine.
             # For an empty batch, return empty tensors or handle gracefully.
             # Depending on the Trainer's expectation, an empty dict might work,
             # or returning tensors of size 0.
             # Let's return tensors with correct shapes but size 0.
             print("Warning: All features in a batch were filtered out.")
             return {
                 "input_values": torch.empty(0, dtype=torch.float32),
                 "attention_mask": torch.empty(0, dtype=torch.long),
                 "labels": torch.empty(0, dtype=torch.long)
             }
        
        # Prepare input features for processor.pad
        input_features = [{"input_values": feature["audio"]} for feature in features]
        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )

        # Prepare labels (do NOT use processor.pad for labels)
        label_ids = [torch.tensor(self.processor.tokenizer.encode(feature["text"]), dtype=torch.long) for feature in features]
        labels = pad_sequence(label_ids, batch_first=True, padding_value=self.processor.tokenizer.pad_token_id)
        # Replace padding with -100 for CTC loss
        labels = labels.masked_fill(labels == self.processor.tokenizer.pad_token_id, -100)
        batch["labels"] = labels

        return batch

def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)
    
    # pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id # This line causes issues, will remove
    # Decode predicted ids, ignoring pad tokens
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    # Decode label ids, grouping consecutive tokens
    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id # Replace -100 back to pad token id for decoding
    label_str = processor.batch_decode(pred.label_ids, group_tokens=True)
    
    # Calculate CER and WER
    cer_metric = cer(label_str, pred_str)
    wer_metric = wer(label_str, pred_str)
    
    return {"cer": cer_metric, "wer": wer_metric}

def evaluate_model(model, processor, dataset):
    """Evaluate the model and return CER and WER."""
    print("\nEvaluating model on the test set...")
    eval_results = trainer.evaluate(eval_dataset=dataset)
    print("\nEvaluation Results:")
    print(f"Character Error Rate (CER): {eval_results['eval_cer']:.4f}")
    print(f"Word Error Rate (WER): {eval_results['eval_wer']:.4f}")
    return eval_results

if __name__ == "__main__":
    # Check system resources before starting
    check_system_resources()
    
    # Clean up old wandb runs
    cleanup_wandb_runs()
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Initialize wandb
    wandb.init(project="telugu-asr", config=vars(config))
    
    # Create vocabulary for Telugu
    # Re-including the custom vocabulary
    vocab_dict = {
        "<pad>": 0,
        "": 1,
        "|": 2,
        "a": 3,
        "b": 4,
        "c": 5,
        "d": 6,
        "e": 7,
        "f": 8,
        "g": 9,
        "h": 10,
        "i": 11,
        "j": 12,
        "k": 13,
        "l": 14,
        "m": 15,
        "n": 16,
        "o": 17,
        "p": 18,
        "q": 19,
        "r": 20,
        "s": 21,
        "t": 22,
        "u": 23,
        "v": 24,
        "w": 25,
        "x": 26,
        "y": 27,
        "z": 28,
        " ": 29,
        "అ": 30,
        "ఆ": 31,
        "ఇ": 32,
        "ఈ": 33,
        "ఉ": 34,
        "ఊ": 35,
        "ఋ": 36,
        "ౠ": 37,
        "ఎ": 38,
        "ఏ": 39,
        "ఐ": 40,
        "ఒ": 41,
        "ఓ": 42,
        "ఔ": 43,
        "క": 44,
        "ఖ": 45,
        "గ": 46,
        "ఘ": 47,
        "ఙ": 48,
        "చ": 49,
        "ఛ": 50,
        "జ": 51,
        "ఝ": 52,
        "ఞ": 53,
        "ట": 54,
        "ఠ": 55,
        "డ": 56,
        "ఢ": 57,
        "ణ": 58,
        "త": 59,
        "థ": 60,
        "ద": 61,
        "ధ": 62,
        "న": 63,
        "ప": 64,
        "ఫ": 65,
        "బ": 66,
        "భ": 67,
        "మ": 68,
        "య": 69,
        "ర": 70,
        "ఱ": 71,
        "ల": 72,
        "ళ": 73,
        "వ": 74,
        "శ": 75,
        "ష": 76,
        "స": 77,
        "హ": 78,
        "ఽ": 79,
        "ా": 80,
        "ి": 81,
        "ీ": 82,
        "ు": 83,
        "ూ": 84,
        "ృ": 85,
        "ౄ": 86,
        "ె": 87,
        "ే": 88,
        "ై": 89,
        "ొ": 90,
        "ో": 91,
        "ౌ": 92,
        "్": 93,
        "౦": 94,
        "౧": 95,
        "౨": 96,
        "౩": 97,
        "౪": 98,
        "౫": 99,
        "౬": 100,
        "౭": 101,
        "౮": 102,
        "౯": 103,
    }
    
    # Save vocabulary (optional, but good practice)
    vocab_path = os.path.join(config.output_dir, "vocab.json")
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab_dict, f, ensure_ascii=False)
    
    # Create tokenizer using custom vocabulary
    tokenizer = Wav2Vec2CTCTokenizer(
        vocab_path,
        unk_token="",
        pad_token="<pad>",
        word_delimiter_token="|"
    )
    
    # Load feature extractor from pre-trained model
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(config.model_name)
    
    # Create processor using the loaded feature extractor and custom tokenizer
    processor = Wav2Vec2Processor(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer
    )
    
    # Load and prepare dataset
    print("Loading and preparing dataset...")
    dataset = prepare_dataset(config.data_dir)
    train_dataset = dataset.train_test_split(test_size=0.1)
    
    # Load pre-trained model and configure for the new vocabulary size
    print(f"Loading pre-trained model: {config.model_name} and adapting head for {len(vocab_dict)} labels...")
    
    # First load the model config
    model_config = Wav2Vec2Config.from_pretrained(
        config.model_name,
        num_labels=len(vocab_dict),
        vocab_size=len(vocab_dict),  # Set vocab size to match our vocabulary
        pad_token_id=processor.tokenizer.pad_token_id,
    )
    
    # Then load the model with the custom config
    model = Wav2Vec2ForCTC.from_pretrained(
        config.model_name,
        config=model_config,
        ctc_loss_reduction="mean",
    )
    
    # Ensure the model is moved to the correct device
    model.to(config.device)
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=2,
        learning_rate=config.learning_rate,
        num_train_epochs=config.epochs,
        save_strategy="epoch",
        save_steps=100,
        save_total_limit=1,  # Keep only the latest checkpoint
        eval_steps=100,
        logging_steps=10,
        report_to="wandb",
        optim="adamw_torch",
        remove_unused_columns=False,
        fp16=True,
        gradient_checkpointing=True,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        eval_strategy="steps",
        load_best_model_at_end=False,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset["train"],
        eval_dataset=train_dataset["test"],
        data_collator=DataCollatorCTCWithPadding(processor=processor, padding=True),
        compute_metrics=compute_metrics,
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Evaluate the model after training
    evaluate_model(model, processor, train_dataset["test"])
    
    # Save final model and processor
    print("\nSaving final model and processor...")
    trainer.save_model(os.path.join(config.output_dir, "final_model"))
    if not os.path.exists(os.path.join(config.output_dir, "final_model", "preprocessor_config.json")):
         processor.save_pretrained(os.path.join(config.output_dir, "final_model"))

    print("Process completed!")