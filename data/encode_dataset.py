import json
from pathlib import Path

from datasets import load_from_disk, DatasetDict

from tokenizer import tokenize

DATA_DIRECTORY = Path(__file__).resolve().parent
PROCESSED_DATASET_PATH = Path(DATA_DIRECTORY / "processed" / "lr_sum_eng")
VOCABULARY_PATH = Path(DATA_DIRECTORY / "vocabulary")
TOKEN_TO_ID_PATH = VOCABULARY_PATH / "token_to_id.json"
UNKNOWN = "<unk>"
START_OF_SENTENCE = "<sos>"
END_OF_SENTENCE = "<eos>"

MAX_SOURCE_LENGTH = 512
MAX_TARGET_LENGTH = 64

ENCODED_DATASET_PATH = DATA_DIRECTORY / "encoded" / "lr_sum_eng"

def encode_text(tokens, token_to_id, maximum_length, add_start_token=False):
    if tokens is None:
        raise ValueError("Tokens cannot be None")

    unknown_id = token_to_id[UNKNOWN]
    end_id = token_to_id[END_OF_SENTENCE]

    encoded_text = []

    if add_start_token:
        encoded_text.append(token_to_id[START_OF_SENTENCE])

    available_token_positions = maximum_length - len(encoded_text) - 1

    for token in tokens[:available_token_positions]:
        token_id = token_to_id.get(token, unknown_id)
        encoded_text.append(token_id)

    encoded_text.append(end_id)

    return encoded_text

def encode_example(example, token_to_id):
    article_tokens = tokenize(example["source_text"])
    target_tokens = tokenize(example["target_text"])

    source_ids = encode_text(
        tokens=article_tokens,
        token_to_id=token_to_id,
        maximum_length=MAX_SOURCE_LENGTH,
        add_start_token=False,
    )

    target_ids = encode_text(
        tokens=target_tokens,
        token_to_id=token_to_id,
        maximum_length=MAX_TARGET_LENGTH,
        add_start_token=True,
    )

    return {
        "source_ids": source_ids,
        "target_ids": target_ids,
        "source_length": len(source_ids),
        "target_length": len(target_ids),
    }

def main():
    with TOKEN_TO_ID_PATH.open("r", encoding="utf-8") as file:
        token_to_id = json.load(file)

    dataset = load_from_disk(str(PROCESSED_DATASET_PATH))

    encoded_splits = {}

    for split_name, dataset_split in dataset.items():
        print(f"Encoding {split_name} split...")

        encoded_splits[split_name] = dataset_split.map(
            encode_example,
            fn_kwargs = {"token_to_id": token_to_id,},
            desc = f"Encoding {split_name}"
        )

    encoded_dataset = DatasetDict(encoded_splits)

    ENCODED_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)

    encoded_dataset.save_to_disk(str(ENCODED_DATASET_PATH))


if __name__ == "__main__":
    main()
