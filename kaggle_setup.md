# Setting up Telugu ASR Training on Kaggle (40-hour Dataset)

This guide provides steps to run the `train.py` script on Kaggle Notebooks using the 40-hour Telugu ASR dataset, `wav2vec2-small` model, custom vocabulary, and checkpointing.

## 1. Create a New Kaggle Notebook
1. Go to [Kaggle Notebooks](https://www.kaggle.com/notebooks)
2. Click "New Notebook"
3. Enable GPU accelerator:
   - Click on "Accelerator" in the right sidebar
   - Select "GPU T4 ×2" (or the best available GPU option, T4 is recommended for FP16 support)
   - Click "Save"

## 2. Add the 40-hour Dataset
1. In the right sidebar, click on "Add Data"
2. Search for your 40-hour Telugu ASR dataset that you have uploaded to Kaggle Datasets
3. Click "Add" to add it to your notebook
4. The dataset will typically be mounted at `/kaggle/input/your-dataset-name`. **Verify the exact path** by looking at the input paths in the right sidebar after adding the dataset
   ```python
   class Config:
       # ... other configs
       data_dir = "/kaggle/input/your-actual-dataset-name" # Update if needed
       # ... other configs
   ```

## 3. Setup Code
1. Open a terminal in your Kaggle notebook (usually under the `...` menu -> "Open Terminal")
2. Clone your repository containing `train.py` and `requirements.txt`:
   ```bash
   !git clone your_repository_url
   %cd your_repository_directory
   ```
   *(Remember to replace `your_repository_url` and `your_repository_directory` with your actual repository details.)*
3. Install required packages:
   ```python
   !pip install -r requirements.txt
   ```

## 4. Run Training
1. Execute the training script in a notebook code cell:
   ```python
   !python train.py
   ```
2. The script will automatically check the `output` directory (`/kaggle/working/output`) for the latest saved checkpoint and resume training from there if one is found. This is crucial for continuing training if your Kaggle session is interrupted (e.g., after 9 hours). If no checkpoint is found, training will start from the beginning.

## 5. Monitor Training
- **Console Output:** The notebook's output will show training progress (logging steps every 10 steps) and the final CER/WER after each epoch and at the very end.
- **Weights & Biases (wandb):** Open the wandb link printed in the console output to view detailed training metrics, loss curves, system usage, and more. The evaluation metrics (CER/WER) are also logged here per epoch.
- **Kaggle Metrics:** Monitor GPU usage and other system metrics in the "Accelerator" tab in the right sidebar.

## 6. Resume Training After Interruption
If your Kaggle session stops before training completes (e.g., 9-hour limit):
1. Start a new session for the same notebook.
2. Re-enable the GPU accelerator.
3. Rerun steps 3 and 4 (cloning and installing dependencies). The cloning should be fast if the directory already exists, or you can adjust the commands.
4. Run step 5 (`!python train.py`). The script will detect the checkpoints saved in `/kaggle/working/output` and automatically resume training from the last saved epoch.

## 7. Download Model and Checkpoints
- The trained model, processor, and checkpoints are saved in `/kaggle/working/output`.
- You can download this directory from the "Output" section in the right sidebar of your notebook after training is complete or at any point to save intermediate checkpoints.

**Important Notes:**
- Kaggle's free tier has limitations on GPU hours per week and session duration.
- Training on 40 hours of data, even with a GPU, can take significant time.
- The `wav2vec2-small` model might have limitations in achieving very low CER/WER on a large dataset of a complex language like Telugu compared to larger models or models specifically pre-trained on Telugu. 