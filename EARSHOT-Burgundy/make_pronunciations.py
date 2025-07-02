# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from pathlib import Path
import pickle

from trftools.align import TextGrid

from earshot.earshot_lexicon import read_words


DATA_DIR = Path('~/Data/EARSHOT').expanduser()


def read_grids(
        lexicon: str = 'MALD-NEIGHBORS-1000',
):
    if not lexicon.startswith('MALD'):
        raise ValueError(f"{lexicon=}: no TextGrids available")
    grid_dir = DATA_DIR / 'MALD' / 'fixed-words'
    grids = {}
    for word in read_words(lexicon):
        grid = TextGrid.from_file(grid_dir / f'{word}.TextGrid').strip_stress()
        grids[word] = grid
    return grids


if __name__ == '__main__':
    lexicon = 'MALD-NEIGHBORS-1000'
    pronunciation_file = DATA_DIR / f'Pronunciation-{lexicon}.pickle'
    grids = read_grids(lexicon)
    mald_pd = {}
    for word, grid in grids.items():
        for r in grid.realizations:
            if not r.is_silence():
                break
        mald_pd[word] = ' '.join(r.phones)

    with pronunciation_file.open('wb') as file:
        pickle.dump(mald_pd, file, pickle.HIGHEST_PROTOCOL)
