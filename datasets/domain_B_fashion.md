# Domain B: Fashion & Retail — Dataset Guide

Welcome to the Fashion & Retail E-Commerce Track. You will use these two datasets across all 12 experiments.

## 🖼️ Vision Dataset (Labs 2, 3, 4, 6)
We use the **MNIST** shape registry as high-contrast structural blueprints. This teaches generative models how to synthesize clean edge contours, visual silhouettes, and item configurations from flat spatial inputs.

### How to load in PyTorch:
```python
import torchvision.datasets as datasets
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.ToTensor()
])

# Automated download and baseline caching of mathematical visual footprints
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

📝 NLP/Text Dataset (Labs 5, 8, 9, 11)

We use the stable, verified IMDB Sentiment Corpus as a benchmark proxy. Students will learn text preprocessing, classification, and language generation principles on large-scale consumer sentiment data.
How to load via Hugging Face:
Python

from datasets import load_dataset

# Loading the verified, script-free consumer sentiment repository via standfordnlp namespace
dataset = load_dataset("stanfordnlp/imdb", split="train")

# Verify the data structure by printing the total count and first row
print(f"🎉 Success! Loaded {len(dataset)} rows of retail customer review data.")
print("Sample row layout:", dataset[0])

📊 Understanding the Data Schema

To view this data as a clean spreadsheet matrix inside Google Colab, convert it into a Pandas DataFrame using this block:
Python

import pandas as pd

df = pd.DataFrame(dataset)
df.head(10)
