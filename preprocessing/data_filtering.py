#!/usr/bin/env python
# coding: utf-8
"""
Filters the scored augmented data (output of data_scoring.py) with every combination of
sacreBLEU, COMET, and LaBSE-CosSim thresholds.
"""

import json
from itertools import combinations

KEYS = ('id', 'source', 'label', 'ref', 'bleu', 'comet', 'cos')


def score_filtering(target_data: list, thresholds: dict) -> list:
    """
    Keeps the items whose scores are all greater than or equal to the given thresholds.

    Args:
        target_data (list): List of dictionaries containing 'id', 'source', 'label', 'ref', 'bleu', 'comet', and 'cos'.
        thresholds (dict): Score name ('bleu', 'comet', 'cos') -> threshold.
    Returns:
        list: The items that meet every threshold.
    """
    return [
        {k: d[k] for k in KEYS}
        for d in target_data
        if all(d[name] >= value for name, value in thresholds.items())
    ]


def main():
    data_path = '../data/filtered_data/new_train_score_merged.json'
    save_path = '../data/filtered_data/new'

    thresholds = {
        'bleu': 30,     # sacreBLEU
        'comet': 0.70,  # COMET
        'cos': 0.80,    # LaBSE-CosSim
    }

    with open(data_path, 'rt', encoding='utf-8') as d:
        data = json.load(d)

    # Every non-empty combination: bleu, comet, cos, bleu&comet, bleu&cos, comet&cos, bleu&comet&cos
    for r in range(1, len(thresholds) + 1):
        for names in combinations(thresholds, r):
            selected = {name: thresholds[name] for name in names}
            filtered_data = score_filtering(data, selected)

            print(f"{' & '.join(names)} : {len(filtered_data)}")

            prefix = 'train_both_' if len(names) > 1 else 'train_'
            file_name = prefix + '_'.join(f'{name}{value}' for name, value in selected.items()) + '.json'
            with open(f'{save_path}/{file_name}', 'wt', encoding='utf-8') as f:
                json.dump(filtered_data, f, indent=4)


if __name__ == '__main__':
    main()
