#!/usr/bin/env python
# coding: utf-8
"""
Sentence-level LoRA fine-tuning of Llama-3.2-1B-Instruct for English -> German translation.
"""

import random
from datetime import datetime

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from dataset import TrainDataset, Collator
from trainer import KST, Trainer, get_logger, load_data

_LOGGER = get_logger(__name__)


def main():
    random.seed(10)
    date = datetime.now(KST).strftime('%m%d%H%M')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_path = '../data/news_commentary_v16/News-Commentary.json'    # Path to the sentence-level training data
    model_save_path = f'../models/new/sl_{date}_fine-tuned_llama3_1B_lora_epoch3'

    model_id = 'meta-llama/Llama-3.2-1B-Instruct'

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_id, device_map='auto')

    # (1) Load data
    _LOGGER.info('Starting Pre-Processing..')
    data_pairs = load_data(train_path)
    train_dataset = TrainDataset(random.sample(data_pairs, 20000), tokenizer)

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, batch_size=3,
        collate_fn=Collator(tokenizer, padding_side='right'),
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
    _LOGGER.info('Starting Fine-Tuning..')
    trainer = Trainer(model=lora_model, epochs=3, dataloader=train_dataloader, device=device)
    trainer()

    lora_model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)

    _LOGGER.info('Success!!')


if __name__ == '__main__':
    main()
