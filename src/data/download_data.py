import logging
from pathlib import Path

from datasets import load_dataset

from src.utils.config import load_config, setup_logging


setup_logging()
logger = logging.getLogger(__name__)


def download_squad():
    config = load_config()

    raw_data_dir = Path(config["paths"]["raw_data"])

    logger.info("Downloading SQuAD dataset...")
    dataset = load_dataset("squad")

    logger.info("Saving dataset to: %s", raw_data_dir)
    dataset.save_to_disk(raw_data_dir)

    logger.info("SQuAD dataset download and save completed.")

if __name__ == "__main__":
    download_squad()