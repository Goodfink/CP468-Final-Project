from pathlib import Path
import re
import unicodedata

from datasets import Dataset, DatasetDict, load_from_disk

DATA_DIRECTORY = Path(__file__).resolve().parent
RAW_DATASET_PATH = Path(DATA_DIRECTORY / "raw" / "lr_sum_eng")
PROCESSED_DATASET_PATH = Path(DATA_DIRECTORY / "processed" / "lr_sum_eng")

MIN_ARTICLE_WORDS = 20
MIN_SUMMARY_WORDS = 3

def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()

def preprocess(example):
    return {
        "source_text": clean_text(example["text"]),
        "target_text": clean_text(example["summary"])
    }

def is_valid(example: dict) -> bool:
    source_length = len(example["source_text"].split())
    target_length = len(example["target_text"].split())

    return source_length > MIN_ARTICLE_WORDS and target_length > MIN_SUMMARY_WORDS

def build_split(dataset_split: Dataset, split_name: str) -> Dataset:
    processed_split = dataset_split.map(
        preprocess,
        desc = f"cleaning {split_name}"
    )

    precessed_split = processed_split.filter(
        is_valid,
        desc = f"filtering {split_name}"
    )

    desired_columns = [
        "id",
        "url",
        "title",
        "source_text",
        "target_text"
    ]

    existing_columns = [column for column in desired_columns if column in processed_split.column_names]
    processed_split = processed_split.select_columns(existing_columns)

    return processed_split

def build_processed_dataset(dataset: DatasetDict) -> DatasetDict:
    return DatasetDict(
        {
            "train": build_split(dataset["train"], "train"),
            "validation": build_split(dataset["validation"], "validation"),
            "test": build_split(dataset["test"], "test")
        }
    )


def main() -> None:
    if not RAW_DATASET_PATH.exists():
        raise FileNotFoundError(f"RAW_DATASET_PATH does not exist: {RAW_DATASET_PATH} please run data/download_dataset.py first")

    dataset = load_from_disk(str(RAW_DATASET_PATH))
    processed_dataset = build_processed_dataset(dataset)

    if PROCESSED_DATASET_PATH.exists():
        raise FileExistsError(f"{PROCESSED_DATASET_PATH} already exists, delete it before rebuilding the dataset")

    PROCESSED_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    processed_dataset.save_to_disk(str(PROCESSED_DATASET_PATH))

if __name__ == "__main__":
    main()