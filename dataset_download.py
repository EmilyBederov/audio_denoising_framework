import os
import glob
import random
import torchaudio
import pandas as pd
from tqdm import tqdm
from kagglehub import dataset_download

# Config
random.seed(42)
TARGET_SR = 16000
SPLIT_RATIO = 0.9
DATA_ROOT = "data"
CSV_TRAIN = os.path.join(DATA_ROOT, "training_pairs_16khz.csv")
CSV_EVAL = os.path.join(DATA_ROOT, "evaluation_pairs_16khz.csv")

# Output directories
dirs = {
    "train_clean": os.path.join(DATA_ROOT, "training/clean"),
    "train_noisy": os.path.join(DATA_ROOT, "training/noisy"),
    "eval_clean": os.path.join(DATA_ROOT, "evaluation/clean"),
    "eval_noisy": os.path.join(DATA_ROOT, "evaluation/noisy"),
}

for d in dirs.values():
    os.makedirs(d, exist_ok=True)

# Step 1: Download
print("Downloading Valentini dataset...")
dataset_path = dataset_download("muhmagdy/valentini-noisy")
print("Downloaded to:", dataset_path)

# Step 2: Get the 56-speaker sets only
clean_files = sorted(glob.glob(os.path.join(dataset_path, "clean_trainset_56spk_wav", "*.wav")))
noisy_files = sorted(glob.glob(os.path.join(dataset_path, "noisy_trainset_56spk_wav", "*.wav")))

assert len(clean_files) == len(noisy_files), "Mismatch in clean/noisy file counts"

# Step 3: Shuffle and split
paired = list(zip(clean_files, noisy_files))
random.shuffle(paired)
split_idx = int(len(paired) * SPLIT_RATIO)
train_pairs = paired[:split_idx]
eval_pairs = paired[split_idx:]

resampler = torchaudio.transforms.Resample(orig_freq=48000, new_freq=TARGET_SR)

def process_pairs(pairs, clean_dir, noisy_dir, csv_path):
    rows = []
    for i, (clean_path, noisy_path) in enumerate(tqdm(pairs)):
        sample_name = f"sample_{i+1:04d}.wav"
        clean_out = os.path.join(clean_dir, sample_name)
        noisy_out = os.path.join(noisy_dir, sample_name)

        # Load and resample
        clean_wav, _ = torchaudio.load(clean_path)
        noisy_wav, _ = torchaudio.load(noisy_path)
        clean_wav = resampler(clean_wav)
        noisy_wav = resampler(noisy_wav)

        torchaudio.save(clean_out, clean_wav, TARGET_SR)
        torchaudio.save(noisy_out, noisy_wav, TARGET_SR)

        rows.append({
            "clean_path": os.path.abspath(clean_out),
            "noisy_path": os.path.abspath(noisy_out),
        })

    pd.DataFrame(rows).to_csv(csv_path, index=False)

# Step 4: Run
print("Processing training set...")
process_pairs(train_pairs, dirs["train_clean"], dirs["train_noisy"], CSV_TRAIN)

print("Processing evaluation set...")
process_pairs(eval_pairs, dirs["eval_clean"], dirs["eval_noisy"], CSV_EVAL)

print(" All done! Files saved in 'data/'")
