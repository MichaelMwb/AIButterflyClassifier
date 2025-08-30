# 🦋 Butterfly Classifier (PyTorch, Transfer Learning)

Classifies **75 butterfly species** from images using a pretrained **VGG16** with a small task-specific head.  
Validation accuracy ≈ **90%** on a held-out split.

**Live demo:** https://michaelmwb-butterfly-classifier.hf.space/

---

## Repo contents

- `train.py` — training script (saves `model.pth` + `stats.txt`)
- `ImageDataset.py` — CSV-driven dataset & transforms
- `categories.py` — class names (`CATEGORIES = [...]`) in training order
- `makeplots.py` — turns `stats.txt` → `learning.png` (loss + accuracy)
- `predict.py` — batch inference (writes `predictions.csv`)
- `app.py` — small Gradio UI (upload image → top-k predictions)
- `requirements.txt` — Python deps
- `learning.png` — example learning curves (generated)
- `predictions.csv` — predictions of the model

> **Note on data:** Training/validation/test images are **not** stored in this repo.  
> Download the **data pack** ➜ **[here](https://drive.google.com/drive/folders/1EzBWq2fndev6-8rkBhKb2OtMmYOFyYHQ?usp=sharing)**
---
🛠 Usage
Once the Requirements are Installed, Run:
```
python3 train.py   # May take a long time depending on CPU
```
⚡**For NVIDIA GPU Users**: To enable CUDA acceleration (recommended for training), install PyTorch with GPU support:
```
python3 -m pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121
```
To Generate Learning Curves, Run:
```
python makeplots.py # Creates learning.png
```
To Run Batch Inference on Images:
```
python predict.py # Writes predictions labels to predictions.csv
```
