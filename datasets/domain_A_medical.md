# Domain A: Medical & Healthcare — Dataset Guide

Welcome to the Clinical Artificial Intelligence Track. You will use these two datasets across all 12 experiments.

## 🖼️ Vision Dataset (Labs 2, 3, 4, 6)
We use **MedMNIST (ChestMNIST)**, which contains $28 \times 28$ greyscale X-ray images of the human chest cavity.

### How to load in PyTorch:
```python
# First run: !pip install medmnist
import medmnist
from medmnist import INFO
import torchvision.transforms as transforms

data_flag = 'chestmnist'
info = INFO[data_flag]
DataClass = getattr(medmnist, info['python_class'])

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5], std=[.5])
])

train_dataset = DataClass(split='train', transform=transform, download=True)

📝 NLP/Text Dataset (Labs 5, 8, 9, 11)

We use MedQA board exam subsets to train diagnostic text generation models, medical summarizers, and clinical RAG pipelines.
How to load via Hugging Face:
Python

from datasets import load_dataset

# Loading the verified, script-free pure parquet repository
dataset = load_dataset("fzkuji/MedQA", split="train")

# Verify the data structure by printing the total count and first row
print(f"🎉 Success! Loaded {len(dataset)} rows of medical clinical text data.")
print("Sample row layout:", dataset[0])
