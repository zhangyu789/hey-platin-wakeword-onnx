# train.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import csv
import librosa

# ================== 1. 数据集 ==================
class WakeWordDataset(Dataset):
    def __init__(self, csv_file, sample_rate=16000, duration=1.5):
        self.data = []
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                self.data.append((row[0], int(row[1])))
        self.sr = sample_rate
        self.duration = int(duration * sample_rate)
        self.mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=40,
            melkwargs={'n_fft': 400, 'hop_length': 160, 'n_mels': 40}
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path, label = self.data[idx]
        wav, sr = torchaudio.load(path)
        if sr != self.sr:
            wav = torchaudio.transforms.Resample(sr, self.sr)(wav)
        wav = wav.mean(0)  # to mono

        # pad or truncate
        if wav.shape[0] > self.duration:
            wav = wav[:self.duration]
        else:
            pad = self.duration - wav.shape[0]
            wav = torch.nn.functional.pad(wav, (0, pad))

        # MFCC
        mfcc = self.mfcc_transform(wav)  # (40, T)
        mfcc = mfcc - mfcc.mean(dim=1, keepdim=True)  # normalize
        return mfcc, torch.tensor(label, dtype=torch.long)

# ================== 2. 模型：轻量 CNN + ArcFace ==================
class WakeWordCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(128, 128)  # embedding

    def forward(self, x):
        x = x.unsqueeze(1)  # (B,1,40,T)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x  # embedding

class ArcFace(nn.Module):
    def __init__(self, embedding_size=128, num_classes=2, s=30.0, m=0.3):
        super().__init__()
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_size))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embedding, label):
        cosine = nn.functional.linear(
            nn.functional.normalize(embedding),
            nn.functional.normalize(self.weight)
        )
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2) + 1e-6)
        phi = cosine * torch.cos(self.m) - sine * torch.sin(self.m)
        phi = torch.where(cosine > 0, phi, cosine)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output

# ================== 3. 训练主函数 ==================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 生成 train.csv
    print("Generating train.csv...")
    with open("data/train.csv", "w", newline="") as f:
        writer = csv.writer(f)
        for path in os.listdir("data/positive"):
            if path.endswith(".wav"):
                writer.writerow([f"data/positive/{path}", 1])
        for path in os.listdir("data/negative"):
            if path.endswith(".wav"):
                writer.writerow([f"data/negative/{path}", 0])

    # 数据加载
    dataset = WakeWordDataset("data/train.csv")
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

    # 模型
    model = WakeWordCNN().to(device)
    arcface = ArcFace().to(device)
    optimizer = optim.Adam(list(model.parameters()) + list(arcface.parameters()), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # 训练
    epochs = 25
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for mfcc, label in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            mfcc, label = mfcc.to(device), label.to(device)

            embedding = model(mfcc)
            output = arcface(embedding, label)
            loss = nn.CrossEntropyLoss()(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()
        print(f"Loss: {total_loss/len(dataloader):.4f}")

    # 保存 PyTorch 模型
    torch.save({
        'model': model.state_dict(),
        'arcface': arcface.state_dict()
    }, "model/checkpoint.pth")
    print("Training finished. Checkpoint saved.")

if __name__ == "__main__":
    main()