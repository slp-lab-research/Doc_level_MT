import json
import logging
from datetime import datetime

import evaluate
import torch
from pytz import timezone
from torch.optim import AdamW
from tqdm.auto import tqdm
from transformers import get_scheduler

KST = timezone('Asia/Seoul')


def get_logger(name: str) -> logging.Logger:
    """
    Returns a logger that prints '[YYYY/MM/DD HH:MM:SS] message' (KST) to stderr.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        formatter = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%Y/%m/%d %H:%M:%S')
        formatter.converter = lambda ts: datetime.fromtimestamp(ts, KST).timetuple()
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def load_data(data_path: str) -> list:
    """
    Loads data from a JSON file.

    Args:
        data_path (str): Path to the JSON file containing the data.
    Returns:
        data (list): A list of data pairs loaded from the JSON file.
    """
    with open(data_path, 'r', encoding='utf-8') as d:
        return json.load(d)


class Trainer():
    """
    Fine-tunes a (LoRA) causal language model for machine translation.

    Attributes:
        model (PreTrainedModel): The model to be fine-tuned.
        epochs (int): Number of epochs for training.
        dataloader (DataLoader): DataLoader for training data.
        device (torch.device): Device the batches are moved to.
        steps (int): Total number of training steps.
        optimizer (AdamW): Optimizer for updating model parameters.
        scheduler (LRScheduler): Linear schedule with 1% warm-up.
    """
    def __init__(self, model, epochs, dataloader, device, lr=5e-5):
        self.model = model
        self.epochs = epochs
        self.dataloader = dataloader
        self.device = device

        self.steps = epochs * len(dataloader)
        self.optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        self.scheduler = get_scheduler(
            'linear',
            optimizer=self.optimizer,
            num_warmup_steps=int(self.steps * 0.01),
            num_training_steps=self.steps,
        )

    def __call__(self):
        progress_bar = tqdm(range(self.steps))

        self.model.train()
        for _ in range(self.epochs):
            for batch in self.dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                loss = self.model(**batch).loss
                if not torch.isfinite(loss):
                    raise RuntimeError('Non-finite loss, ending training.')
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                progress_bar.set_description(f'loss: {loss.item():.4f}')
                progress_bar.update(1)


class Evaluator():
    """
    Generates translations for the evaluation set and computes sacreBLEU.

    Attributes:
        model (PreTrainedModel): The fine-tuned model to be evaluated.
        tokenizer (Tokenizer): Tokenizer used to process the text data.
        dataloader (DataLoader): DataLoader for evaluation data (left-padded).
        device (torch.device): Device the batches are moved to.
    """
    def __init__(self, model, tokenizer, dataloader, device):
        self.model = model
        self.tokenizer = tokenizer
        self.dataloader = dataloader
        self.device = device

    def __call__(self):
        """
        Returns:
            (bleu_score, result):
                bleu_score (dict): The computed sacreBLEU score.
                result (list): A list of dictionaries containing 'input', 'output', and 'label'.
        """
        bleu_metric = evaluate.load('sacrebleu')
        result = []

        self.model.eval()
        for batch in tqdm(self.dataloader):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    pad_token_id=self.tokenizer.pad_token_id,
                    max_new_tokens=8192,
                )

            # Inputs are left-padded, so every generated continuation starts at the same index.
            prompt_len = batch['input_ids'].shape[1]
            decoded_input = self.tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=True)
            decoded_input = [p.split('user\n\n', 1)[1].split('assistant\n\n', 1)[0] for p in decoded_input]
            decoded_pred = self.tokenizer.batch_decode(outputs[:, prompt_len:], skip_special_tokens=True)
            decoded_labels = self.tokenizer.batch_decode(batch['labels'], skip_special_tokens=True)

            bleu_metric.add_batch(predictions=decoded_pred, references=decoded_labels)

            for i in range(len(decoded_pred)):
                result.append({
                    'input': decoded_input[i],
                    'output': decoded_pred[i],
                    'label': decoded_labels[i],
                })

        bleu_score = bleu_metric.compute(tokenize='13a')

        return bleu_score, result
