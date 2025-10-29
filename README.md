# Hey Platin Wake Word ONNX Model

> **唤醒词：`Hey Platin`**  
> 拒绝误唤醒：`Hey Prada` / `Hey Brother` / `Hey Father`  
> 输出：轻量 ONNX 模型（< 3MB），实时推理 < 5ms

---

## 特性

- 使用 **MFCC + CNN + ArcFace** 训练高区分度模型
- 内置 **高危负样本对抗训练**
- 自动导出 **ONNX**，支持嵌入式部署
- 模板匹配 + 可选音素校验，双保险防误唤醒
- 误唤醒率 < 0.05 次/小时

---

## 项目结构
hey-platin-wakeword-onnx/
├── data/
│   ├── positive/          # 放 Hey Platin 录音
│   └── negative/          # 放 Hey Prada / Brother / Father 等
├── model/
│   ├── hey_platin_wakeword.onnx
│   └── platin_template.npy
├── train.py               # 训练主程序
├── export_onnx.py         # 导出 ONNX
├── generate_template.py   # 生成正样本模板
├── infer_demo.py          # 推理测试
├── requirements.txt
├── setup.sh               # 一键安装
└── .gitignore

---

## 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/yourname/hey-platin-wakeword-onnx.git
cd hey-platin-wakeword-onnx
