import logging
import os

import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

from src.utils.config import load_config, setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def load_model(model_path: str):
    logger.info("Loading model from: %s", model_path)
    model = AutoModelForQuestionAnswering.from_pretrained(model_path)
    model.eval()
    return model


def quantize_model(model):
    logger.info("Applying dynamic quantization (INT8)...")

    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},
        dtype=torch.qint8,
    )

    return quantized_model


def get_model_size_mb(path: str):
    total_size = 0
    for root, _, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            total_size += os.path.getsize(fp)
    return total_size / (1024 * 1024)


def save_model(model, tokenizer, output_path: str):
    os.makedirs(output_path, exist_ok=True)

    logger.info("Saving quantized model to: %s", output_path)
    torch.save(model.state_dict(), os.path.join(output_path, "pytorch_model.bin"))
    tokenizer.save_pretrained(output_path)


def run_quantization():
    config = load_config()

    model_dir = config["paths"]["inference_model_dir"]
    output_dir = config["paths"].get(
        "quantized_model_dir",
        "checkpoints/quantized_model"
    )

    logger.info("Model directory: %s", model_dir)
    logger.info("Output directory: %s", output_dir)

    logger.info("Loading model...")
    model = load_model(model_dir)

    original_size = get_model_size_mb(model_dir)
    logger.info("Original model size: %.2f MB", original_size)

    logger.info("Quantizing model...")
    quantized_model = quantize_model(model)

    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    save_model(quantized_model, tokenizer, output_dir)

    quantized_size = get_model_size_mb(output_dir)
    logger.info("Quantized model size: %.2f MB", quantized_size)

    if quantized_size > 0:
        logger.info(
            "Compression ratio: %.2f x",
            original_size / quantized_size
        )

    logger.info("Quantization completed successfully.")


if __name__ == "__main__":
    run_quantization()