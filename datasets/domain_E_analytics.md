# Domain E: Education & Analytics — Dataset Guide

Welcome to the Cognitive Learning & Analytics Track. You will use these two datasets across all 12 experiments.

## 🖼️ Vision Dataset (Labs 2, 3, 4, 6)
We use the **MNIST** dataset as a structural baseline proxy for sketch pattern analytics. This teaches generative models how to process clean vector stroke positions, geometric boundaries, and mathematical pixel weights from human handwriting primitives.

### How to load in PyTorch:
```python
import torchvision.datasets as datasets
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.ToTensor()
])

# Automated download and baseline caching of mathematical vector footprints
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

📝 NLP/Text Dataset (Labs 5, 8, 9, 11)

We use the OpenBookQA scientific question-answering corpus to build academic reasoning systems, context retriever agents, and learning support frameworks.
How to load via Hugging Face:
Python

from datasets import load_dataset

# Loading the structured, script-free academic knowledge base via allenai namespace
dataset = load_dataset("allenai/openbookqa", "main", split="train")

# Verify the data structure by printing the total count and first row
print(f"🎉 Success! Loaded {len(dataset)} scientific reasoning instances.")
print("Sample row layout:", dataset[0])

📊 Visualizing the Educational Database

To view the question structures and multiple-choice formats as a clean matrix spreadsheet inside Google Colab, use this block:
Python

import pandas as pd

df = pd.DataFrame(dataset)
df[['question_stem', 'choices', 'answerKey']].head(10)
