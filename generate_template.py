# generate_template.py
from train import WakeWordCNN, WakeWordDataset
from torch.utils.data import DataLoader
import torch
import numpy as np

model = WakeWordCNN()
checkpoint = torch.load("model/checkpoint.pth", map_location="cpu")
model.load_state_dict(checkpoint['model'])
model.eval()

dataset = WakeWordDataset("data/train.csv")
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

embeddings = []
for mfcc, label in dataloader:
    if label.item() == 1:  # 只取正样本
        with torch.no_grad():
            emb = model(mfcc).numpy()
        embeddings.append(emb)

template = np.mean(embeddings, axis=0)
np.save("model/platin_template.npy", template)
print("Template saved.")