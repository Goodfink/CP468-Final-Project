# CP468 Final Project

## Requirements

```bash
python -m pip install -r requirements.txt
```

## Run Steps

### Dataset

This project uses the LR-SUM English dataset

TO download the dataset run

```bash
python data/download_dataset.py
```

data will be saved too:

```text
data/raw/lr_sum_eng_clean/
```

### Preprocess dataset

To preprocess the dataset run:

```bash
python data/preprocess.py
```

Cleaned dataset will be saved too:

```text
data/processed/lr_sum_eng_clean/
```

### Build Vocabulary

To build the vocab run:

```bash
python data/build_vocabulary.py
```

Vocab will be saved like:

```text
data/vocabulary/
├── token_to_id.json
└── id_to_token.json
```