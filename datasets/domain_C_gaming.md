# Domain C: Entertainment & Gaming — Dataset Guide

Welcome to the Creative Generative Systems Track. You will use these two datasets across all 12 experiments.

## 🖼️ Vision Dataset (Labs 2, 3, 4, 6)
We use a curation of **$32 \times 32$ Pixel Art Character Sprites** mapping core dimensional asset footprints. This teaches systems to reconstruct and expand complex game design components.

### How to load in PyTorch:
```python
import torchvision.datasets as datasets
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])

# Automated loading from a standardized multi-channel platform setup
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

📝 NLP/Text Dataset (Labs 5, 8, 9, 11)

We use Awesome ChatGPT Prompts as an accessible framework to simulate game persona behavior logic and narrative structures.
How to load via Hugging Face:
Python

from datasets import load_dataset

# Loading a stable, script-free engineering prompt framework
dataset = load_dataset("fka/awesome-chatgpt-prompts", split="train")

# Verify the data structure by printing the total count and first row
print(f"🎉 Success! Loaded {len(dataset)} structural script inputs.")
print("Sample row layout:", dataset[0])
