#!/usr/bin/env python
# coding: utf-8
"""
Two-stage fine-tuning: the sentence-level LoRA model (sl_MT_finetuning.py) is further fine-tuned
on document-level data, followed by sacreBLEU evaluation on human-labeled document-level data.
"""

import json
import random
from datetime import datetime

import torch
from peft import PeftModel
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from dataset import TrainDataset, EvaluateDataset, Collator
from trainer import KST, Trainer, Evaluator, get_logger, load_data

_LOGGER = get_logger(__name__)


def main():
    random.seed(10)
    date = datetime.now(KST).strftime('%m%d%H%M')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_path = '../data/filtered_data/new/new_train_score_merged.json'    # Path to the document-level augmented/filtered training data
    eval_path = '../data/news_commentary_v16/doc-level/merged_news.json'    # Path to the document-level human-labeled evaluation data
    model_save_path = f'../models/new/dl_{date}_fine-tuned_llama3_1B_lora_epoch3'
    result_path = f'../result/new/dl_{date}_fine-tuned_epoch3_1B_lora_result.json'

    base_model_id = 'meta-llama/Llama-3.2-1B-Instruct'
    model_path = '../models/sl_04082238_fine-tuned_llama3_1B_lora_epoch3'   # Path to the LoRA model fine-tuned on sentence-level data

    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(base_model_id, device_map='auto')

    # (1) Load data
    _LOGGER.info('Starting Pre-Processing..')
    train_data_pairs = load_data(train_path)
    eval_data_pairs = load_data(eval_path)
    random.shuffle(train_data_pairs)
    random.shuffle(eval_data_pairs)
    train_dataset = TrainDataset(train_data_pairs[:2000], tokenizer)
    evaluation_dataset = EvaluateDataset(eval_data_pairs[9000:10500], tokenizer)

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, batch_size=1,
        collate_fn=Collator(tokenizer, padding_side='right'),
    )
    evaluation_dataloader = DataLoader(
        evaluation_dataset, batch_size=1,
        collate_fn=Collator(tokenizer, padding_side='left', label_pad_token_id=tokenizer.pad_token_id),
    )

    # (2) Load the LoRA model that was fine-tuned on sentence-level data
    lora_model = PeftModel.from_pretrained(base_model, model_path, is_trainable=True)
    lora_model.to(device)

    # (3) Train
    _LOGGER.info(f'Starting Fine-Tuning..\nmodel: {model_save_path}\nbase model: {model_path}')
    trainer = Trainer(model=lora_model, epochs=3, dataloader=train_dataloader, device=device)
    trainer()

    lora_model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)

    # (4) Evaluation
    _LOGGER.info('Starting Evaluation..')
    evaluator = Evaluator(model=lora_model, tokenizer=tokenizer, dataloader=evaluation_dataloader, device=device)
    bleu, result = evaluator()

    _LOGGER.info('Success!!')
    print('=' * 30)
    print(bleu)
    print('=' * 30)

    with open(result_path, 'wt', encoding='utf-8') as r:
        json.dump(result, r, indent=4)


if __name__ == '__main__':
    main()
