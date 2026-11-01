# BacLLM

**Language-guided protein representations for antimicrobial resistance analysis**

[Model weights on Hugging Face](https://huggingface.co/rongye1/BacLLM/tree/main/BacLLM_model)

BacLLM is a protein representation framework for antimicrobial resistance (AMR) analysis in WHO priority bacterial pathogens. It adapts an ESM-2 sequence encoder using functional knowledge from protein annotations. After training, BacLLM takes amino acid sequences as input and produces protein embeddings for downstream analysis.

This repository explains how to load the released sequence encoder and extract embeddings. Model weights are hosted on [Hugging Face](https://huggingface.co/rongye1/BacLLM/tree/main/BacLLM_model).


## Quick start with random protein sequences

Copy the following code into `quickstart.py` and run `python quickstart.py`. It generates three reproducible random amino acid sequences, loads BacLLM, extracts one embedding per sequence, and saves the results to an NPZ file.

The random sequences demonstrate the input and output format. They have no assigned biological function or resistance labels.

```python
import random

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

# The weights are in a subfolder of the Hugging Face repository.
REPO_ID = "rongye1/BacLLM"
SUBFOLDER = "BacLLM_model"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(REPO_ID, subfolder=SUBFOLDER)
model = AutoModel.from_pretrained(
    REPO_ID,
    subfolder=SUBFOLDER,
    add_pooling_layer=False,
).to(device).eval()

# Each input is a continuous amino acid string, without spaces.
rng = random.Random(42)
amino_acids = "ACDEFGHIKLMNPQRSTVWY"
protein_ids = ["random_protein_1", "random_protein_2", "random_protein_3"]
sequences = [
    "".join(rng.choices(amino_acids, k=length))
    for length in (80, 120, 160)
]

inputs = tokenizer(
    sequences,
    return_tensors="pt",
    add_special_tokens=True,
    padding=True,
    truncation=True,
    max_length=1024,
)
inputs = {key: value.to(device) for key, value in inputs.items()}

with torch.inference_mode():
    hidden_states = model(**inputs).last_hidden_state
    # Match the reference script: average all non-padding tokens.
    # The <cls> and <eos> tokens are included in this average.
    mask = inputs["attention_mask"].unsqueeze(-1).to(hidden_states.dtype)
    embeddings = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1)

embeddings = embeddings.float().cpu().numpy()
np.savez_compressed(
    "random_protein_embeddings.npz",
    embeddings=embeddings,
    protein_ids=np.asarray(protein_ids),
    sequences=np.asarray(sequences),
)

print("Device:", device)
print("Sequence lengths:", [len(sequence) for sequence in sequences])
print("Embedding shape:", embeddings.shape)
print("Saved to random_protein_embeddings.npz")
```

For the released checkpoint, the expected shapes are:

```text
Sequence lengths: [80, 120, 160]
Embedding shape: (3, 480)
Saved to random_protein_embeddings.npz
```

