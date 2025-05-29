# Telugu ASR using Wav2Vec2

This project implements an Automatic Speech Recognition (ASR) system for the Telugu language using Facebook's wav2vec2-small model. The system calculates Character Error Rate (CER) and Word Error Rate (WER) as performance metrics.

## Requirements

- Python 3.7+
- CUDA-compatible GPU (recommended)
- 10 hours of Telugu audio data in MP3 format (6-10 seconds per file)
- Corresponding transcriptions

## Setup

1. Install the required packages:
```bash
pip install -r requirements.txt
```

2. Prepare your data:
   - Place all MP3 audio files in the `raw_data` directory
   - Place the transcription file (`transcriptions.txt`) in the `raw_data` directory
   - The transcription file should be tab-separated with format:
     ```
     file_id    telugu_text
     ```

3. Run the preprocessing script:
```bash
python preprocess.py
```

4. Start the training:
```bash
python train.py
```

## Project Structure

```
.
├── README.md
├── requirements.txt
├── train.py
├── preprocess.py
├── raw_data/
│   ├── *.mp3
│   └── transcriptions.txt
├── data/
│   ├── *.wav
│   └── transcriptions.txt
└── output/
    └── final_model/
```

## Configuration

The main configuration parameters can be found in the `Config` class in `train.py`:

- `learning_rate`: 1e-4
- `epochs`: 30
- `hidden_dropout`: 0.1
- `batch_size`: 8
- `max_duration`: 10 seconds
- `sampling_rate`: 16000

## Training Monitoring

The training progress is monitored using Weights & Biases (wandb). You'll need to:
1. Create a wandb account
2. Run `wandb login` before training
3. Monitor your training at wandb.ai

## Performance Metrics

The system calculates two main metrics:
1. Character Error Rate (CER)
2. Word Error Rate (WER)

These metrics are logged to wandb during training and evaluation.

## Notes

- The audio preprocessing includes:
  - Converting to 16kHz sampling rate
  - Converting to mono if stereo
  - Normalizing audio
  - Converting to WAV format
- The model uses the wav2vec2-small architecture pretrained on unlabeled audio data
- Training includes automatic mixed precision for faster training
- The model and processor are saved after training in the `output/final_model` directory 