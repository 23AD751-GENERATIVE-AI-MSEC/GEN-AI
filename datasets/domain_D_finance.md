# Domain D: Finance & Economics — Dataset Guide

Welcome to the Quantitative Finance Track. You will use these two datasets across all 12 experiments.

## 🖼️ Vision Dataset (Labs 2, 3, 4, 6)
We map stock metrics into structural $32 \times 32$ matrix heatmaps using synthetic imagery. This teaches models how to learn generative distributions from non-standard visual structures.

### How to load in PyTorch:
```python
import torchvision.datasets as datasets
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.ToTensor()
])

# Automated download and cache of baseline matrix footprints
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

📝 NLP/Text Dataset (Labs 5, 8, 9, 11)

We use the stable, script-free Financial Phrasebank repository to train financial sentiment classifiers, automated company disclosure generators, and compliance-checked RAG applications.
How to load via Hugging Face:
Python

from datasets import load_dataset

# Loading the verified, script-free parquet format repository
dataset = load_dataset("FinanceMTEB/financial_phrasebank", split="train")

# Verify the data structure by printing the total count and first row
print(f"🎉 Success! Loaded {len(dataset)} rows of financial data.")
print("Sample row layout:", dataset[0])

📊 Understanding the Data Schema

To view this data as a clean spreadsheet matrix inside Google Colab, convert it into a Pandas DataFrame using this block:
Python

import pandas as pd

df = pd.DataFrame(dataset)
df.head(10)
