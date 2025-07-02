# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from pathlib import Path
import pickle
from typing import Literal


DATA_DIR = Path('~/Data/EARSHOT').expanduser()
BUILTIN_DATA_DIR = Path(__file__).parent / 'data'

VOWELS = ('AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 'IH', 'IY', 'OW', 'OY', 'UH', 'UW')
CONSONANTS = ('B', 'C', 'CH', 'D', 'DH', 'F', 'G', 'HH', 'JH', 'K', 'L', 'M', 'N', 'NG', 'P', 'R', 'S', 'SH', 'T', 'TH', 'V', 'W', 'Y', 'Z', 'ZH')
PHONES = VOWELS + CONSONANTS


def read_words(
        lexicon: Literal['earshot', 'burgundy', 'MALD-NEIGHBORS-1000'] = 'earshot',
):
    if lexicon.startswith('MALD'):
        path = BUILTIN_DATA_DIR / f'Words-{lexicon}.txt'
        words = [line.split() for line in path.open()]
        return [row[0] for row in words if len(row) == 2]
    return (BUILTIN_DATA_DIR / f'Words-{lexicon.capitalize()}.txt').read_text().splitlines()


def read_pronunciations(
        lexicon: Literal['earshot', 'burgundy', 'MALD-NEIGHBORS-1000'] = 'earshot',
        split_phones: bool = False,  # turn pronunciations into list of str
):
    if lexicon == 'burgundy':
        lexicon = 'MALD-NEIGHBORS-1000'
    pronunciation_file = DATA_DIR / f'Pronunciation-{lexicon}.pickle'
    with pronunciation_file.open('rb') as file:
        pronunciations = pickle.load(file)
    if split_phones:
        pronunciations = {word: pronunciation.split(' ') for word, pronunciation in pronunciations.items()}
    return pronunciations
