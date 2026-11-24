#!/usr/bin/env python
# coding: utf-8
"""
Evaluates translation results (output of the fine-tuning scripts) with COMET (wmt22-comet-da).
"""

import json
import os

from comet import download_model, load_from_checkpoint


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


def main():
    result_path = '../result/new'

    model = load_from_checkpoint(download_model('Unbabel/wmt22-comet-da'))
    model.enable_context()

    for l in data_search(result_path):
        with open(result_path + l, 'rt', encoding='utf-8') as f:
            result = json.load(f)
        data = [{'src': d['input'], 'mt': d['output'], 'ref': d['label']} for d in result]

        model_output = model.predict(data, batch_size=1, gpus=1)

        print(l)
        print(f'{min(model_output.scores)}~{max(model_output.scores)}')
        print(model_output.system_score)
        print('-----------------------')


if __name__ == '__main__':
    main()
