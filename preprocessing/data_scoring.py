#!/usr/bin/env python
# coding: utf-8
"""
Scores augmented (source, label) pairs against the human reference with sacreBLEU, COMET,
and LaBSE cosine similarity (LaBSE-CosSim).
"""

import json
import logging
from datetime import datetime

import evaluate
import spacy
import torch
import torch.nn.functional as F
from comet import download_model, load_from_checkpoint
from pytz import timezone
from sentence_transformers import SentenceTransformer

KST = timezone('Asia/Seoul')


def get_logger(name: str) -> logging.Logger:
    formatter = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%Y/%m/%d %H:%M:%S')
    formatter.converter = lambda ts: datetime.fromtimestamp(ts, KST).timetuple()
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


_LOGGER = get_logger(__name__)


def bleu_scoring(target_data: list) -> list:
    """
    Computes a sentence-level sacreBLEU score of 'label' against 'ref' for each item.

    Args:
        target_data (list): List of dictionaries containing 'source', 'label', and 'ref'.
    Returns:
        list: sacreBLEU score of each item.
    """
    bleu_metric = evaluate.load('sacrebleu')

    return [
        bleu_metric.compute(predictions=[d['label']], references=[d['ref']], tokenize='13a')['score']
        for d in target_data
    ]


def comet_scoring(target_data: list) -> list:
    """
    Computes the COMET (wmt22-comet-da) score of each item.

    Args:
        target_data (list): List of dictionaries containing 'source', 'label', and 'ref'.
    Returns:
        list: COMET score of each item.
    """
    model = load_from_checkpoint(download_model('Unbabel/wmt22-comet-da'))
    model.enable_context()

    data = [{'src': d['source'], 'mt': d['label'], 'ref': d['ref']} for d in target_data]

    return model.predict(data, batch_size=1, gpus=1).scores


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


def cosine_similarity_labse(target_data: list) -> list:
    """
    Computes the document-level cosine similarity between 'source' (English) and 'label' (German)
    using averaged LaBSE sentence embeddings (LaBSE-CosSim).

    Args:
        target_data (list): List of dictionaries containing 'source' and 'label'.
    Returns:
        list: LaBSE-CosSim score of each item.
    """
    model = SentenceTransformer('sentence-transformers/LaBSE')

    nlp_en = spacy.load('en_core_web_trf')  # English sentence splitter for document-level text
    nlp_de = spacy.load('de_dep_news_trf')  # German sentence splitter for document-level text

    en_sents, de_sents = [], []
    en_lens, de_lens = [], []

    for d in target_data:
        en_sent = [sent.text for sent in nlp_en(d['source']).sents]
        de_sent = [sent.text for sent in nlp_de(d['label']).sents]

        en_sents.extend(en_sent)
        de_sents.extend(de_sent)
        en_lens.append(len(en_sent))
        de_lens.append(len(de_sent))

    en_emb_avgs = emb_avg(torch.from_numpy(model.encode(en_sents)), en_lens)
    de_emb_avgs = emb_avg(torch.from_numpy(model.encode(de_sents)), de_lens)

    return F.cosine_similarity(en_emb_avgs, de_emb_avgs, dim=1).tolist()


def main():
    data_path = '../data/merged_data/train_re_merged.json'
    save_path = '../data/filtered_data'

    with open(data_path, 'rt', encoding='utf-8') as d:
        data = json.load(d)

    # (1) Evaluate scores
    _LOGGER.info('Evaluating sacreBLEU score..')
    bleu_scores = bleu_scoring(data)

    _LOGGER.info('Evaluating COMET score..')
    comet_scores = comet_scoring(data)

    _LOGGER.info('Evaluating Cosine Similarity based on LaBSE..')
    cosine_similarity = cosine_similarity_labse(data)

    # (2) Save the scores
    result = [
        {
            'id': d['id'],
            'source': d['source'],
            'label': d['label'],
            'ref': d['ref'],
            'bleu': bleu_scores[i],
            'comet': comet_scores[i],
            'cos': cosine_similarity[i],
        } for i, d in enumerate(data)
    ]

    with open(f'{save_path}/new_train_score_merged.json', 'wt', encoding='utf-8') as f:
        json.dump(result, f, indent=4)
    _LOGGER.info(f'Saved {len(result)} scored items.')


if __name__ == '__main__':
    main()
