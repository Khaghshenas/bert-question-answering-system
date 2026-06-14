import logging
import os
from pathlib import Path

import torch
from transformers import DistilBertForQuestionAnswering, DistilBertTokenizerFast

from src.utils.config import load_config, setup_logging


setup_logging()
logger = logging.getLogger(__name__)


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
IS_CI = os.getenv("CI") == "true"

_model = None
_tokenizer = None


def load_model_and_tokenizer():
    global _model, _tokenizer

    if _model is not None and _tokenizer is not None:
        return _model, _tokenizer

    # ---------------- CI Mode ----------------
    if IS_CI:

        logger.info("Running in CI mode with dummy model.")

        class DummyOutput:
            def __init__(self):
                self.start_logits = torch.tensor([[0, 0, 0, 0, 10]])
                self.end_logits = torch.tensor([[0, 0, 0, 0, 10]])

        class DummyModel:
            def eval(self):
                pass

            def __call__(self, **kwargs):
                return DummyOutput()

        class DummyTokenizer:
            def __call__(
                self,
                question,
                context,
                return_tensors="pt",
                truncation=True,
                max_length=384,
            ):
                return {
                    "input_ids": torch.tensor([[101, 102, 103, 104, 105]]),
                    "attention_mask": torch.tensor([[1, 1, 1, 1, 1]]),
                }

            def decode(self, ids, skip_special_tokens=True):
                return "Paris"

        _model = DummyModel()
        _tokenizer = DummyTokenizer()

        return _model, _tokenizer

    # ---------------- Normal Mode: Real Model ----------------
    config = load_config()
    model_dir = Path(config["paths"]["inference_model_dir"])

    logger.info("Loading model from: %s", model_dir)

    valid_files = [
        "pytorch_model.bin",
        "model.safetensors",
        "tf_model.h5",
        "flax_model.msgpack",
    ]

    if not model_dir.exists() or not any((model_dir / f).exists() for f in valid_files):
        raise FileNotFoundError(f"Model not found at {model_dir.resolve()}")

    _model = DistilBertForQuestionAnswering.from_pretrained(model_dir)
    _tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)

    _model.eval()

    return _model, _tokenizer


def predict_answer(question: str, context: str) -> str:
    model, tokenizer = load_model_and_tokenizer()

    inputs = tokenizer(
        question,
        context,
        return_tensors="pt",
        truncation=True,
        max_length=384,
    )

    with torch.no_grad():
        outputs = model(**inputs)

    start_idx = torch.argmax(outputs.start_logits, dim=-1).item()
    end_idx = torch.argmax(outputs.end_logits, dim=-1).item()

    if end_idx < start_idx:
        end_idx = start_idx

    answer = tokenizer.decode(
        inputs["input_ids"][0][start_idx : end_idx + 1],
        skip_special_tokens=True,
    )

    return answer

if __name__ == "__main__":

    answer = predict_answer("What does CPU stand for?", "The CPU (Central Processing Unit) is the brain of a computer.")

    logger.info("Answer:  %s", answer)
