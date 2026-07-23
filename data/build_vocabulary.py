import json
from collections import Counter
from pathlib import Path

from datasets import load_from_disk
from tokenizer import tokenize

DATA_DIRECTORY = Path(__file__).resolve().parent
PROCESSED_DATASET_PATH = Path(DATA_DIRECTORY / "processed" / "lr_sum_eng")
VOCABULARY_PATH = Path(DATA_DIRECTORY / "vocabulary")
TOKEN_TO_ID_PATH = VOCABULARY_PATH / "token_to_id.json"
ID_TO_TOKEN_PATH = VOCABULARY_PATH / "id_to_token.json"

MAX_VOCAB_SIZE = 30000
special_tokens = [
    "<pad>",
    "<unk>",
    "<sos>",
    "<eos>",
]

def main() -> None:
    token_to_id = {}
    id_to_token = {}
    token_counts = Counter()
    training_dataset = load_from_disk(str(PROCESSED_DATASET_PATH))["train"]

    for example in training_dataset:
        article_text = tokenize(example["source_text"])
        target_text = tokenize(example["target_text"])

        token_counts.update(article_text)
        token_counts.update(target_text)

    number_of_most_common = MAX_VOCAB_SIZE - len(special_tokens)
    most_common_tokens = token_counts.most_common(number_of_most_common)

    # IDs 0, 1, 2, and 3
    for i in range(len(special_tokens)):
        token = special_tokens[i]

        token_to_id[token] = i
        id_to_token[i] = token

    # Normal tokens begin at ID 4
    for i in range(len(most_common_tokens)):
        token = most_common_tokens[i][0]
        token_id = i + len(special_tokens)

        token_to_id[token] = token_id
        id_to_token[token_id] = token

    save_vocabulary(token_to_id, id_to_token)

def save_vocabulary(token_to_id, id_to_token) -> None:
    VOCABULARY_PATH.mkdir(parents=True, exist_ok=True)

    with TOKEN_TO_ID_PATH.open(mode="w", encoding="utf-8") as f:
        json.dump(token_to_id, f, ensure_ascii=False, indent=4)

    with ID_TO_TOKEN_PATH.open(mode="w", encoding="utf-8") as f:
        json.dump(id_to_token, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()