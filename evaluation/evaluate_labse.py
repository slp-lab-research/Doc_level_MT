#!/usr/bin/env python
# coding: utf-8
"""
Evaluates translation results (output of the fine-tuning scripts) with LaBSE cosine similarity (LaBSE-CosSim)
between the English input and the German output.
"""

import json
import os

import spacy
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer


def data_search(base_path: str, add_path: str = '') -> list:
    """
    Recursively searches for JSON files in the given directory.

    Args:
        base_path (str): The base directory to search.
        add_path (str): Path relative to the original base directory (used for recursion).
    Returns:
        data_loc (list): Paths of the found JSON files, relative to base_path.
    """
    data_loc = []
    for c in os.listdir(base_path):
        full_path = f'{base_path}/{c}'
        if os.path.isdir(full_path):
            data_loc.extend(data_search(full_path, f'{add_path}/{c}'))
        elif os.path.isfile(full_path) and os.path.splitext(c)[1] == '.json':
            data_loc.append(f'{add_path}/{c}')

    return data_loc


def emb_avg(embedding: torch.Tensor, sent_lens: list) -> torch.Tensor:
    """
    Averages consecutive sentence embeddings into one embedding per document.

    Args:
        embedding (torch.Tensor): Sentence embeddings of all documents, concatenated.
        sent_lens (list): Number of sentences in each document.
    Returns:
        torch.Tensor: One averaged embedding per document.
    """
    emb_avgs = []
    idx = 0
    for l in sent_lens:
        emb_avgs.append(embedding[idx:idx + l].mean(dim=0))
        idx += l
    return torch.stack(emb_avgs, dim=0)


def evaluate_labse(target_data: list, model, nlp_en, nlp_de) -> list:
    """
    Computes the document-level cosine similarity between 'input' (English) and 'output' (German)
    using averaged LaBSE sentence embeddings (LaBSE-CosSim).

    Args:
        target_data (list): A list of dictionaries containing 'input', 'output', and 'label'.
        model (SentenceTransformer): LaBSE model.
        nlp_en, nlp_de (spacy.Language): English / German sentence splitters.
    Returns:
        list: LaBSE-CosSim score of each item.
    """
    en_sents, de_sents = [], []
    en_lens, de_lens = [], []

    for d in target_data:
        en_sent = [sent.text for sent in nlp_en(d['input']).sents]
        de_sent = [sent.text for sent in nlp_de(d['output']).sents]

        en_sents.extend(en_sent)
        de_sents.extend(de_sent)
        en_lens.append(len(en_sent))
        de_lens.append(len(de_sent))

    en_emb_avgs = emb_avg(torch.from_numpy(model.encode(en_sents)), en_lens)
    de_emb_avgs = emb_avg(torch.from_numpy(model.encode(de_sents)), de_lens)

    return F.cosine_similarity(en_emb_avgs, de_emb_avgs, dim=1).tolist()


def main():
    result_path = '../result/new'

    model = SentenceTransformer('sentence-transformers/LaBSE')
    nlp_en = spacy.load('en_core_web_trf')  # English sentence splitter for document-level text
    nlp_de = spacy.load('de_dep_news_trf')  # German sentence splitter for document-level text

    for l in data_search(result_path):
        with open(result_path + l, 'rt', encoding='utf-8') as d:
            data = json.load(d)

        cs_list = evaluate_labse(data, model, nlp_en, nlp_de)

        cs_avg = sum(cs_list) / len(cs_list)
        print(l)
        print(f'{cs_avg} ({min(cs_list)} ~ {max(cs_list)})')
        print('-----------------------------------')


if __name__ == '__main__':
    main()
