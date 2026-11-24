#!/usr/bin/env python
# coding: utf-8
"""
Document-level-only (DLO) LoRA fine-tuning of Llama-3.2-1B-Instruct for English -> German translation,
followed by sacreBLEU evaluation on human-labeled document-level data.
"""

import json
import random
from datetime import datetime

import torch
from peft import LoraConfig, get_peft_model
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
    model_save_path = f'../models/new/dlo_{date}_fine-tuned_llama3_1B_lora_epoch3_warmup200'
    result_path = f'../result/new/dlo_{date}_fine-tuned_epoch3_1B_lora_result.json'

    model_id = 'meta-llama/Llama-3.2-1B-Instruct'

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_id, device_map='auto')

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

    # (2) LoRA settings
    config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=['q_proj', 'v_proj'],
        lora_dropout=0.1,
        bias='none',
    )
    lora_model = get_peft_model(model, config)
    lora_model.to(device)

    # (3) Train
    _LOGGER.info(f'Starting Fine-Tuning..\nmodel: {model_save_path}\nbase model: {model_id}')
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
