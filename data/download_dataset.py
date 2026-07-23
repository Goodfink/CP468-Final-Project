from pathlib import Path
from datasets import load_dataset

DATASET_NAME = ("bltlab/lr-sum")
LANGUAGE_CONFIG = "eng"
DATA_DIRECTORY = Path(__file__).resolve().parent
OUTPUT_DIRECTORY = DATA_DIRECTORY / "raw" / "lr_sum_eng"

def main() -> None:
    OUTPUT_DIRECTORY.parent.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(DATASET_NAME, LANGUAGE_CONFIG)

    dataset.save_to_disk(str(OUTPUT_DIRECTORY))

if __name__ == "__main__":
    main()

