import logging
from functools import partial
from pathlib import Path

from datasets import load_dataset
from transformers import DistilBertTokenizerFast

from src.utils.config import load_config, setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def preprocess(examples, tokenizer, config):
    questions = [q.strip() for q in examples["question"]]
    contexts = examples["context"]
    answers = examples["answers"]

    tokenized = tokenizer(
        questions,
        contexts,
        truncation=config["tokenization"]["truncation"],
        max_length=config["tokenization"]["max_length"],
        stride=config["tokenization"]["stride"],
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding=config["tokenization"]["padding"],
    )

    sample_mapping = tokenized.pop("overflow_to_sample_mapping")
    offset_mappings = tokenized.pop("offset_mapping")

    start_positions = []
    end_positions = []

    for i, offsets in enumerate(offset_mappings):
        sample_idx = sample_mapping[i]
        answer = answers[sample_idx]

        # no answer case
        if len(answer["answer_start"]) == 0:
            start_positions.append(0)
            end_positions.append(0)
            continue

        start_char = answer["answer_start"][0]
        end_char = start_char + len(answer["text"][0])

        sequence_ids = tokenized.sequence_ids(i)

        try:
            context_start = next(j for j, s_id in enumerate(sequence_ids) if s_id == 1)
            context_end = len(sequence_ids) - 1 - next(
                j for j, s_id in enumerate(reversed(sequence_ids)) if s_id == 1
            )
        except StopIteration:
            start_positions.append(0)
            end_positions.append(0)
            continue

        if not (
            offsets[context_start][0] <= start_char
            and offsets[context_end][1] >= end_char
        ):
            start_positions.append(0)
            end_positions.append(0)
            continue

        token_start_index = context_start
        while (
            token_start_index <= context_end
            and offsets[token_start_index][0] < start_char
        ):
            token_start_index += 1

        token_end_index = context_end
        while (
            token_end_index >= context_start
            and offsets[token_end_index][1] > end_char
        ):
            token_end_index -= 1

        start_positions.append(token_start_index)
        end_positions.append(token_end_index)

    tokenized["start_positions"] = start_positions
    tokenized["end_positions"] = end_positions

    return tokenized


if __name__ == "__main__":
    
    config = load_config()

    processed_data_dir = Path(config["paths"]["processed_dir"])
    tokenizer_dir = Path(config["paths"]["tokenizer_dir"])

    tokenizer_name = config["models"]["tokenizer_name"]
    dataset_name = config["data"]["dataset_name"]

    logger.info("Loading dataset: %s", dataset_name)
    dataset = load_dataset(dataset_name)

    logger.info("Loading tokenizer: %s", tokenizer_name)
    tokenizer = DistilBertTokenizerFast.from_pretrained(tokenizer_name)

    logger.info("Saving tokenizer to: %s", tokenizer_dir)
    tokenizer.save_pretrained(tokenizer_dir)

    logger.info("Starting tokenization...")

    tokenized_dataset = dataset.map(
        partial(preprocess, tokenizer=tokenizer, config=config),
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    logger.info("Train columns: %s", tokenized_dataset["train"].column_names)

    logger.info("Saving processed dataset to: %s", processed_data_dir)
    tokenized_dataset.save_to_disk(processed_data_dir)

    logger.info("Preprocessing complete.")