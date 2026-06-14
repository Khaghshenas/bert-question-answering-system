import logging
import re
import string

import numpy as np
import torch
from collections import Counter
from datasets import load_from_disk
from transformers import DistilBertForQuestionAnswering, DistilBertTokenizerFast

from src.utils.config import load_config, setup_logging


setup_logging()
logger = logging.getLogger(__name__)


def normalize_answer(s):
    """Lower text, remove punctuation, articles, and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def remove_punc(text):
        return "".join(ch for ch in text if ch not in set(string.punctuation))

    def white_space_fix(text):
        return " ".join(text.split())

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()

    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0

    precision = num_same / len(prediction_tokens)
    recall = num_same / len(ground_truth_tokens)

    return 2 * precision * recall / (precision + recall)


def exact_match_score(prediction, ground_truth):
    return int(normalize_answer(prediction) == normalize_answer(ground_truth))


def evaluate():
    
    config = load_config()

    raw_dataset_path = config["paths"]["raw_data"]
    tokenized_dataset_path = config["paths"]["processed_dir"]
    model_dir = config["paths"]["model_dir"]

    logger.info("Loading raw dataset from: %s", raw_dataset_path)
    raw_dataset = load_from_disk(raw_dataset_path)

    logger.info("Loading tokenized dataset from: %s", tokenized_dataset_path)
    tokenized_dataset = load_from_disk(tokenized_dataset_path)

    raw_val_dataset = raw_dataset["validation"].shuffle(seed=42).select(range(500))
    val_dataset = tokenized_dataset["validation"].shuffle(seed=42).select(range(500))

    logger.info("Loading model from: %s", model_dir)
    model = DistilBertForQuestionAnswering.from_pretrained(model_dir)

    logger.info("Loading tokenizer from: %s", model_dir)
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)

    model.eval()

    em_scores = []
    f1_scores = []

    logger.info("Starting evaluation...")

    for tokenized_example, raw_example in zip(val_dataset, raw_val_dataset):
        context = raw_example["context"]
        question = raw_example["question"]
        answers = raw_example.get("answers", {"text": [""]})

        inputs = tokenizer(
            question,
            context,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )

        with torch.no_grad():
            outputs = model(**inputs)

        start_idx = torch.argmax(outputs.start_logits)
        end_idx = torch.argmax(outputs.end_logits)

        all_tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        answer_tokens = all_tokens[start_idx : end_idx + 1]
        prediction = tokenizer.convert_tokens_to_string(answer_tokens)

        em = max([exact_match_score(prediction, ans) for ans in answers["text"]])
        f1 = max([f1_score(prediction, ans) for ans in answers["text"]])

        em_scores.append(em)
        f1_scores.append(f1)

    avg_em = np.mean(em_scores) * 100
    avg_f1 = np.mean(f1_scores) * 100

    logger.info("Validation Exact Match (EM): %.2f%%", avg_em)
    logger.info("Validation F1 Score: %.2f%%", avg_f1)

    logger.info("Evaluation completed.")


if __name__ == "__main__":
    evaluate()