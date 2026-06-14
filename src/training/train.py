import logging
from pathlib import Path

from datasets import load_from_disk
from transformers import (
    DistilBertForQuestionAnswering,
    DistilBertTokenizerFast,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)

from src.utils.config import load_config, setup_logging


setup_logging()
logger = logging.getLogger(__name__)


def train():
    
    config = load_config()

    logger.info("Loading dataset from: %s", config["paths"]["processed_dir"])
    dataset = load_from_disk(config["paths"]["processed_dir"])

    logger.info("Loading model: %s", config["models"]["pretrained_model"])
    model = DistilBertForQuestionAnswering.from_pretrained(
        config["models"]["pretrained_model"]
    )

    training_args = TrainingArguments(
        output_dir=config["paths"]["model_dir"],
        logging_steps=20,
        logging_strategy="steps",
        logging_dir=config["paths"]["logs_dir"],
        num_train_epochs=config["training"]["num_train_epochs"],
        per_device_train_batch_size=config["training"]["per_device_train_batch_size"],
        save_steps=config["training"]["save_steps"],
        save_total_limit=config["training"]["save_total_limit"],
        remove_unused_columns=config["training"]["remove_unused_columns"],
        dataloader_pin_memory=config["training"]["dataloader_pin_memory"],
    )

    logger.info("Dataset columns: %s", dataset["train"].column_names)
    logger.info("Sample example: %s", dataset["train"][0])

    dataset.set_format(
        type="torch",
        columns=[
            "input_ids",
            "attention_mask",
            "start_positions",
            "end_positions",
        ],
    )

    # Subsampling for faster experiments (should later go to config)
    train_size = 30000
    val_size = 3000

    logger.info("Using subset: train=%d, val=%d", train_size, val_size)

    small_train = dataset["train"].shuffle(seed=42).select(range(train_size))
    small_valid = dataset["validation"].shuffle(seed=42).select(range(val_size))

    tokenizer = DistilBertTokenizerFast.from_pretrained(
        config["models"]["pretrained_model"]
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=small_train,
        eval_dataset=small_valid,
        data_collator=data_collator,
    )

    logger.info("Starting training...")
    trainer.train()

    logger.info("Saving model to: %s", config["paths"]["model_dir"])
    model.save_pretrained(config["paths"]["model_dir"])

    logger.info("Training completed successfully.")


if __name__ == "__main__":
    train()