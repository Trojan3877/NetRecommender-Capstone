reproducibility.md

Reproducibility Guide — DeepSequence-Recommender

This document ensures that the DeepSequence-Recommender results can be reproduced exactly as reported in the research metrics and benchmark files.

Following this guide will recreate:
- Model architecture
- Training setup
- Evaluation metrics
- Dataset splits
- Hyperparameter choices



 1. 📦 Environment Setup

## Python Version

Python 3.10

## Required Libraries
Install dependencies with:

pip install -r requirements.txt

The project uses:

- PyTorch 2.x **or** TensorFlow 2.x (depending on your chosen model)
- NumPy 1.26+
- Pandas
- Scikit-learn
- Matplotlib
- Tqdm
- PyTorch Lightning (if enabled)
- CUDA (optional, GPU recommended)


 2. 🎛 Deterministic Settings / Random Seeds

To guarantee reproducible training:

```python
import torch
import numpy as np
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

Global Project Seed

seed = 42




3. 📁 Dataset Reproducibility

Data Split Method

All models use the same user-aware chronological split:

70% training

10% validation

20% testing


Sequence splits are generated using:

sorted_interactions = interactions.sort_values("timestamp")

No future events leak into training.

Negative Sampling

100 negatives per user

Uniform sampling for fair baseline comparison





4. 🧠 Model Configuration

All model variants (GRU, LSTM, Transformer) share the following:

Hyperparameter	Value

Embedding size	64
Batch size	128
Max sequence length	50
Learning rate	1e-3
Dropout	0.2–0.5
Optimizer	Adam
Loss function	Cross-entropy / BPR


Transformer (SASRec-style)

2 layers

4 attention heads

Hidden size: 128

Positional embeddings enabled





5. 🚀 Running the Full Training Pipeline

Use the training script:

python train.py --model transformer --seed 42 --epochs 20

Other model options:

python train.py --model gru
python train.py --model lstm
python train.py --model itemknn
python train.py --model popularity



6. 📊 Reproducing Evaluation Metrics

After training:

python evaluate.py --model transformer --topk 10

Metrics produced:

Precision@K

Recall@K

NDCG@K

Hit Rate@K

MRR




