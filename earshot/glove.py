# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from pathlib import Path
from typing import Dict

import numpy


DATA_DIR = Path('~/Data').expanduser()


def read_glove(n: int = 50) -> Dict[str, numpy.ndarray]:
    path = DATA_DIR / 'Corpus' / 'glove.6B' / f'glove.6B.{n}d.txt'
    with path.open() as f:
        lines = (line.split(maxsplit=1) for line in f)
        glove = {word.upper(): numpy.fromstring(coefs, "f", sep=" ") for word, coefs in lines}
    return glove
