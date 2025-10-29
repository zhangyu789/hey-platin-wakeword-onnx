# export_onnx.py
import torch
from train import WakeWordCNN
import torchaudio

# 加载模型
device = torch.device("cpu")
model = WakeWordCNN()
checkpoint = torch.load("model/checkpoint.pth", map_location=device)
model.load_state_dict(checkpoint['model'])
model.eval()

# 虚拟输入：1秒音频
dummy_audio = torch.randn(1, 16000)  # (batch, time)
mfcc_transform = torchaudio.transforms.MFCC(
    sample_rate=16000, n_mfcc=40,
    melkwargs={'n_fft': 400, 'hop_length': 160, 'n_mels': 40}
)

# 包装前向（包含 MFCC）
class ONNXModel(torch.nn.Module):
    def __init__(self, cnn, mfcc):
        super().__init__()
        self.cnn = cnn
        self.mfcc = mfcc

    def forward(self, audio):
        mfcc = self.mfcc(audio)  # (B,40,T)
        mfcc = mfcc - mfcc.mean(dim=2, keepdim=True)
        embedding = self.cnn(mfcc)
        return embedding

onnx_model = ONNXModel(model, mfcc_transform)
onnx_model.eval()

# 导出
torch.onnx.export(
    onnx_model,
    dummy_audio,
    "model/hey_platin_wakeword.onnx",
    input_names=["audio"],
    output_names=["embedding"],
    dynamic_axes={"audio": {0: "batch", 1: "time"}},
    opset_version=15
)

print("ONNX model exported: model/hey_platin_wakeword.onnx")