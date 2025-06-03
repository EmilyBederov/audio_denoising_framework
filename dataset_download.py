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
TEST_DATA_ROOT = "test_data"
CSV_TRAIN = os.path.join(DATA_ROOT, "training_pairs_16khz.csv")
CSV_EVAL = os.path.join(DATA_ROOT, "evaluation_pairs_16khz.csv")
CSV_TEST = os.path.join(TEST_DATA_ROOT, "test_pairs_16khz.csv")

# Output directories
dirs = {
    "train_clean": os.path.join(DATA_ROOT, "training/clean"),
    "train_noisy": os.path.join(DATA_ROOT, "training/noisy"),
    "eval_clean": os.path.join(DATA_ROOT, "evaluation/clean"),
    "eval_noisy": os.path.join(DATA_ROOT, "evaluation/noisy"),
    "test_clean": os.path.join(TEST_DATA_ROOT, "test/clean"),
    "test_noisy": os.path.join(TEST_DATA_ROOT, "test/noisy"),
}

for d in dirs.values():
    os.makedirs(d, exist_ok=True)

# Step 1: Download
print("Downloading Valentini dataset...")
dataset_path = dataset_download("muhmagdy/valentini-noisy")
print("Downloaded to:", dataset_path)

# Step 2: Get the 56-speaker sets for train/eval
clean_files = sorted(glob.glob(os.path.join(dataset_path, "clean_trainset_56spk_wav", "*.wav")))
noisy_files = sorted(glob.glob(os.path.join(dataset_path, "noisy_trainset_56spk_wav", "*.wav")))

assert len(clean_files) == len(noisy_files), "Mismatch in clean/noisy file counts"

# Step 3: Get the 28-speaker test sets
test_clean_files = sorted(glob.glob(os.path.join(dataset_path, "clean_testset_wav", "*.wav")))
test_noisy_files = sorted(glob.glob(os.path.join(dataset_path, "noisy_testset_wav", "*.wav")))

assert len(test_clean_files) == len(test_noisy_files), "Mismatch in test clean/noisy file counts"

# Step 4: Shuffle and split train/eval
paired = list(zip(clean_files, noisy_files))
random.shuffle(paired)
split_idx = int(len(paired) * SPLIT_RATIO)
train_pairs = paired[:split_idx]
eval_pairs = paired[split_idx:]

# Test pairs (no shuffling needed)
test_pairs = list(zip(test_clean_files, test_noisy_files))

resampler = torchaudio.transforms.Resample(orig_freq=48000, new_freq=TARGET_SR)

def process_pairs(pairs, clean_dir, noisy_dir, csv_path, prefix="sample"):
    rows = []
    for i, (clean_path, noisy_path) in enumerate(tqdm(pairs)):
        sample_name = f"{prefix}_{i+1:04d}.wav"
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

# Step 5: Process all sets
print("Processing training set...")
process_pairs(train_pairs, dirs["train_clean"], dirs["train_noisy"], CSV_TRAIN, "train")

print("Processing evaluation set...")
process_pairs(eval_pairs, dirs["eval_clean"], dirs["eval_noisy"], CSV_EVAL, "eval")

print("Processing test set (28spk)...")
process_pairs(test_pairs, dirs["test_clean"], dirs["test_noisy"], CSV_TEST, "test")

print(f"All done!")
print(f"Training/Evaluation files saved in '{DATA_ROOT}/'")
print(f"Test files saved in '{TEST_DATA_ROOT}/'")
print(f"Datasets:")
print(f"  - Training pairs: {len(train_pairs)}")
print(f"  - Evaluation pairs: {len(eval_pairs)}")
print(f"  - Test pairs: {len(test_pairs)}")
