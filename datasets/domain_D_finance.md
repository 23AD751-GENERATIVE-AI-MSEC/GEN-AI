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
