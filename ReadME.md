# CP468 Final Project

## Requirements

Install the required dependencies from the project root:

```bash
python -m pip install -r requirements.txt
```

## Run Steps

Run the data preparation scripts in the order shown below.

### Dataset

This project uses the **English subset of the LR-Sum dataset**.

To download the dataset, run:

```bash
python data/download_dataset.py
```

The downloaded dataset will be saved to:

```text
data/raw/lr_sum_eng/
```

### Preprocess Dataset

The preprocessing script cleans the article and summary text while preserving the official training, validation, and test splits.

To preprocess the dataset, run:

```bash
python data/preprocess.py
```

The cleaned dataset will be saved to:

```text
data/processed/lr_sum_eng_clean/
```

### Build Vocabulary

The vocabulary is built using only the training split to prevent validation or test data leakage.

To build the vocabulary, run:

```bash
python data/build_vocabulary.py
```

The vocabulary files will be saved as:

```text
data/vocabulary/
├── token_to_id.json
└── id_to_token.json
```

The vocabulary contains the following special tokens:

```text
<pad> = 0
<unk> = 1
<sos> = 2
<eos> = 3
```

### Encode Dataset

The encoding script tokenizes the cleaned article and summary text and converts each token into its corresponding vocabulary ID.

The same vocabulary, built exclusively from the training split, is used to encode the training, validation, and test splits. Tokens that are not present in the vocabulary are mapped to `<unk>`.

To encode the dataset, run:

```bash
python data/encode_dataset.py
```

The encoded dataset will be saved to:

```text
data/encoded/lr_sum_eng/
```

Each encoded example contains:

```text
id
url
title
source_text
target_text
source_ids
target_ids
source_length
target_length
```

The source article is limited to 512 tokens, including `<eos>`.

The target summary is limited to 64 tokens, including `<sos>` and `<eos>`.

### Batch Padding and Masking

Padding is performed dynamically at model runtime using the helper functions in:

```text
data/collate.py
```

The encoded dataset is not permanently padded. Each batch is padded only to the length of the longest source and target sequence in that batch.

The collate function returns:

```text
source_ids
source_lengths
source_mask
target_ids
target_lengths
target_mask
```

The source mask allows the attention mechanism to ignore `<pad>` tokens.

`collate.py` is imported by the model training code and is not run directly.

## Complete Data Preparation Order

Run the following commands from the project root:

```bash
python data/download_dataset.py
python data/preprocess.py
python data/build_vocabulary.py
python data/encode_dataset.py
```

The raw, processed, and encoded dataset directories are excluded from Git. They can be recreated using the provided scripts.
