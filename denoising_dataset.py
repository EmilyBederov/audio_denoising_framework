import torch
from torch.utils.data import Dataset
import pandas as pd
import torchaudio

class DenoisingDataset(Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        noisy, _ = torchaudio.load(self.df.iloc[idx]['noisy_path'])
        clean, _ = torchaudio.load(self.df.iloc[idx]['clean_path'])
        return noisy, clean
