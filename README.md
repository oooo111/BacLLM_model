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

The embedding dimension is given by `model.config.hidden_size`; this checkpoint uses 12 transformer layers and a hidden size of 480. Embedding values can differ slightly across devices and software versions.

## Batch extraction and FASTA input

The accompanying [`example_bacllm_embeddings.py`](example_bacllm_embeddings.py) provides the same extraction procedure with configurable batching, optional FASTA input, and local model loading.

Run the random-sequence example:

```bash
python example_bacllm_embeddings.py --output random_embeddings.npz
```

For FASTA input, install Biopython and run:

```bash
python -m pip install biopython
python example_bacllm_embeddings.py --fasta proteins.fasta --batch-size 8 --output protein_embeddings.npz
```

A protein FASTA file has this format:

```fasta
>protein_1
MKTAYIAKQRQISFVKSHFSRQDILDLWIYHTQGYFP
>protein_2
MNNIRRVAILLAVAVAFSSAQADKPEADKVKFNK
```

The script preserves FASTA record order and stores each `record.id` with its embedding. Inputs should be amino acid sequences; translate nucleotide sequences before extraction.

## Load a local model

Download the contents of the Hugging Face `BacLLM_model` folder into a local directory. The model directory should contain the configuration, weights, and tokenizer files, including:

```text
BacLLM_model/
├── config.json
├── model.safetensors
├── special_tokens_map.json
├── tokenizer_config.json
└── vocab.txt
```

Then use:

```bash
python example_bacllm_embeddings.py --model-dir ./BacLLM_model --output random_embeddings.npz
```

Or replace the loading code in the quick start with:

```python
tokenizer = AutoTokenizer.from_pretrained("./BacLLM_model")
model = AutoModel.from_pretrained(
    "./BacLLM_model", add_pooling_layer=False
).to(device).eval()
```

## Read the saved embeddings

```python
import numpy as np

with np.load("random_protein_embeddings.npz", allow_pickle=False) as data:
    embeddings = data["embeddings"]
    protein_ids = data["protein_ids"]
    sequences = data["sequences"]

print(embeddings.shape)  # (number_of_proteins, hidden_size)
print(protein_ids[0])
print(sequences[0])
```

| NPZ field       | Contents                                                            |
| --------------- | ------------------------------------------------------------------- |
| `embeddings`  | Float32 matrix of shape `(N, D)`; `D = 480` for this checkpoint |
| `protein_ids` | Protein identifiers in input order                                  |
| `sequences`   | Original amino acid sequences in input order, before truncation     |

The embeddings can be used as features for classifiers such as logistic regression or an SVM, trained on labeled examples for the target task.

## Sequence length and pooling

- The examples use a maximum length of **1,024 tokens**, including `<cls>` and `<eos>`. For standard amino acid inputs, this retains at most **1,022 residues** per protein. Longer sequences are truncated; their embeddings therefore represent the retained prefix. The saved `sequences` field still contains the original input.
- Mean pooling follows the reference extraction script: padding tokens are excluded, while `<cls>` and `<eos>` are included. Excluding special tokens would change the embeddings and can affect compatibility with an existing downstream classifier.
- The output is the mean-pooled final hidden state of the adapted ESM-2 encoder. The examples do not apply the training-time sequence projection module or L2 normalization.
- Reduce `--batch-size` if GPU memory is insufficient. The default script automatically uses the first CUDA GPU when available and otherwise uses the CPU. Use `--device cpu` to select the CPU explicitly.

For long proteins, any alternative handling, such as windowing, should be defined and evaluated for the downstream task.

## Manuscript

Rongye Ye, Lun Li, Zongyi Geng, and Shuhui Song. *Language-guided domain adaptation improves protein sequence representations for antimicrobial resistance analysis.*

## Resources

- [BacLLM model files](https://huggingface.co/rongye1/BacLLM/tree/main/BacLLM_model)
- [Hugging Face ESM documentation](https://huggingface.co/docs/transformers/model_doc/esm)
- [Hugging Face model loading documentation](https://huggingface.co/docs/transformers/models)
