import os
import glob
import shutil
import csv
import random

def organize_audio_and_create_csv(clean_dir, noisy_dir, output_dir, split_ratio=0.8):
    # Ensure clean and noisy lists match
    clean_files = sorted(glob.glob(os.path.join(clean_dir, "*.wav")))
    noisy_files = sorted(glob.glob(os.path.join(noisy_dir, "*.wav")))
    assert len(clean_files) == len(noisy_files), "Mismatch in clean and noisy files"

    filenames = [os.path.basename(f) for f in clean_files]
    combined = list(zip(filenames, clean_files, noisy_files))
    random.shuffle(combined)

    # Split into training and evaluation
    split_idx = int(len(combined) * split_ratio)
    train_set = combined[:split_idx]
    eval_set = combined[split_idx:]

    # Output paths
    paths = {
        "training_clean": os.path.join(output_dir, "training", "clean"),
        "training_noisy": os.path.join(output_dir, "training", "noisy"),
        "evaluation_clean": os.path.join(output_dir, "evaluation", "clean"),
        "evaluation_noisy": os.path.join(output_dir, "evaluation", "noisy"),
    }
    for path in paths.values():
        os.makedirs(path, exist_ok=True)

    def copy_and_log(set_data, subfolder, csv_path):
        rows = []
        for fname, c_path, n_path in set_data:
            clean_dst = os.path.join(paths[f"{subfolder}_clean"], fname)
            noisy_dst = os.path.join(paths[f"{subfolder}_noisy"], fname)
            shutil.copy2(c_path, clean_dst)
            shutil.copy2(n_path, noisy_dst)
            rows.append([clean_dst, noisy_dst])
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["clean_path", "noisy_path"])
            writer.writerows(rows)

    copy_and_log(train_set, "training", os.path.join(output_dir, "training_pairs.csv"))
    copy_and_log(eval_set, "evaluation", os.path.join(output_dir, "evaluation_pairs.csv"))

# Example usage
organize_audio_and_create_csv(
    clean_dir="./clean",
    noisy_dir="./noisy",
    output_dir=".",  # current directory
    split_ratio=0.9
)
