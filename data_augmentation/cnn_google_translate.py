#!/usr/bin/env python
# coding: utf-8
"""
Augments CNN/DailyMail with German translations from Google Translate (googletrans).
"""

import asyncio
import json
import logging
import os
from datetime import datetime

from googletrans import Translator
from pytz import timezone

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


def data_search(base_path: str, add_path: str = '') -> list:
    """
    Recursively searches for JSONL files in the given directory.

    Args:
        base_path (str): The base directory to search.
        add_path (str): Path relative to the original base directory (used for recursion).
    Returns:
        data_loc (list): Paths of the found JSONL files, relative to base_path.
    """
    data_loc = []
    for c in os.listdir(base_path):
        full_path = f'{base_path}/{c}'
        if os.path.isdir(full_path):
            data_loc.extend(data_search(full_path, f'{add_path}/{c}'))
        elif os.path.isfile(full_path) and os.path.splitext(c)[1] == '.jsonl':
            data_loc.append(f'{add_path}/{c}')

    return data_loc


def extract(base_path: str, data_loc: list) -> list:
    """
    Extracts articles from the given JSONL files.

    Args:
        base_path (str): The base directory where the JSONL files are located.
        data_loc (list): List of paths to the JSONL files.
    Returns:
        files (list): For each file, a list of (id, article) tuples.
    """
    files = []
    for loc in data_loc:
        paragraphs = []
        with open(base_path + loc, 'r', encoding='utf-8') as d:
            for line in d:
                data_json = json.loads(line)
                paragraphs.append((data_json['id'], data_json['article']))

        _LOGGER.info(f'{loc} length: {len(paragraphs)}')
        files.append(paragraphs)

    return files


async def translate_bulk(save_path: str, data_loc: list, source: list) -> None:
    """
    Translates the articles from English to German with Google Translate and saves one output file per input file.
    On failure, the request is retried after an increasing waiting time.

    Args:
        save_path (str): The directory where the translated data will be saved.
        data_loc (list): List of paths to the original JSONL files.
        source (list): For each file, a list of (id, article) tuples.
    """
    async with Translator() as translator:
        for loc, items in zip(data_loc, source):
            translated = []
            for cnt, (id, p) in enumerate(items, start=1):
                _LOGGER.info(f'progress: {cnt}/{len(items)}')
                err_cnt = 0
                while True:
                    try:
                        translation = await translator.translate(p, src='en', dest='de')
                        break
                    except Exception as e:
                        waiting_time = 180 + 60 * err_cnt
                        _LOGGER.error(f'Error occurred!\n{e}\n\nTry again in {waiting_time}s..\n')
                        await asyncio.sleep(waiting_time)
                        err_cnt += 1
                translated.append((id, translation))

            saving(save_path, loc, translated)


def saving(save_path: str, loc: str, result: list):
    """
    Saves the translated data to a JSON file.

    Args:
        save_path (str): The directory where the translated data will be saved.
        loc (str): The location of the original data file (relative to the source directory).
        result (list): A list of (id, googletrans.Translated) tuples.
    """
    _LOGGER.info('Saving...')

    path = save_path + loc
    os.makedirs(os.path.dirname(path), exist_ok=True)

    data_list = [
        {
            'id': id,
            'original': translation.origin,
            'translation': translation.text,
        } for id, translation in result
    ]
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data_list, f, indent=4)


async def main():
    data_path = '/home/prml/workspace/kir/data/cnn_dailymail/source_data'
    save_path = '/home/prml/workspace/kir/data/cnn_dailymail/augmented_data'

    # (1) Search for the original data
    _LOGGER.info('Searching the original data..')
    data_loc = data_search(data_path)

    # (2) Extract articles from the original data
    _LOGGER.info('Extracting paragraphs..')
    source = extract(data_path, data_loc)

    # (3) Translate the articles
    _LOGGER.info('Translating..')
    await translate_bulk(save_path, data_loc, source)

    _LOGGER.info('Data was successfully augmented!')


if __name__ == '__main__':
    asyncio.run(main())
