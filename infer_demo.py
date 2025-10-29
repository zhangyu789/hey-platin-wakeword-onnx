# infer_demo.py
import onnxruntime as ort
import numpy as np
import torchaudio
import torch

session = ort.InferenceSession("model/hey_platin_wakeword.onnx")

# 加载模板（训练时正样本均值）
template = np.load("model/platin_template.npy")  # 后面教你生成

def cosine_score(emb1, emb2):
    return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

def predict_wakeword(audio_path, threshold=0.85):
    wav, sr = torchaudio.load(audio_path)
    if sr != 16000:
        wav = torchaudio.transforms.Resample(sr, 16000)(wav)
    wav = wav.mean(0)
    if wav.shape[0] > 16000:
        wav = wav[:16000]
    else:
        wav = torch.nn.functional.pad(wav, (0, 16000 - wav.shape[0]))
    wav = wav.numpy().reshape(1, -1)

    emb = session.run(None, {"audio": wav})[0]
    score = cosine_score(emb[0], template)
    return score > threshold, score

# 测试
test_files = [
    "data/positive/001_hey_platin.wav",
    "data/negative/001_hey_prada.wav",
    "data/negative/002_hey_brother.wav"
]

for f in test_files:
    wake, score = predict_wakeword(f)
    print(f"{f} -> {'WAKE' if wake else 'REJECT'} (score: {score:.3f})")