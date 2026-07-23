from functools import partial
from typing import Any

import torch
from torch.nn.utils.rnn import pad_sequence


def collate_batch(batch: list[dict[str, Any]], pad_id: int) -> dict[str, Any]:
    if not batch:
        raise ValueError("Cannot collate an empty batch")

    source_sequences = [torch.tensor(example["source_ids"], dtype = torch.long) for example in batch]
    target_sequences = [torch.tensor(example["target_ids"], dtype = torch.long) for example in batch]
    source_lengths = torch.tensor([len(sequence) for sequence in source_sequences], dtype = torch.long,)
    target_lengths = torch.tensor([len(sequence) for sequence in target_sequences], dtype = torch.long,)

    padded_source_ids = pad_sequence(source_sequences, batch_first = True, padding_value = pad_id)

    padded_target_ids = pad_sequence(target_sequences, batch_first = True, padding_value = pad_id)

    source_mask = padded_source_ids != pad_id
    target_mask = padded_target_ids != pad_id

    return {
        "source_ids": padded_source_ids,
        "source_lengths": source_lengths,
        "source_mask": source_mask,
        "target_ids": padded_target_ids,
        "target_lengths": target_lengths,
        "target_mask": target_mask,

        "example_ids": [example["id"] for example in batch],
        "source_texts": [example["source_text"] for example in batch],
        "target_texts": [example["target_text"] for example in batch],
    }


def create_collate_fn(pad_id: int):
    return partial(collate_batch, pad_id=pad_id)