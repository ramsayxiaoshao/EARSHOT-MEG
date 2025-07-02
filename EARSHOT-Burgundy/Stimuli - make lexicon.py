# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Databases

# +
from pathlib import Path

from eelbrain import *
import numpy
import trftools

import speech.lexicon


DATASET_ROOT = Path('/Volumes/Seagate BarracudaFastSSD/EARSHOT/Burgundy')
# -

# ## MALD

# +
MALD_DIR = Path('/Volumes/Seagate BarracudaFastSSD/Corpus/MALD/MALD1_rw')
MALD_GRID_DIR = MALD_DIR.parent / 'fixed-words'
MALD_WORDS = [path.stem for path in MALD_DIR.glob('*.wav')]
for broken in ['FRESHMANS', 'BUTTERCREAM']:
    MALD_WORDS.remove(broken)  # TextGrids broken

assert len(MALD_WORDS) == 26791, f'glob error ({len(MALD_WORDS)})'
len(MALD_WORDS), ', '.join(MALD_WORDS[:10])
# -

GRIDS = {}
MALD_PD = {}
for word in MALD_WORDS:
    grid = trftools.align.TextGrid.from_file(MALD_GRID_DIR / f'{word}.TextGrid').strip_stress()
    GRIDS[word] = grid
    for r in grid.realizations:
        if not r.is_silence():
            break
    MALD_PD[word] = ' '.join(r.phones)

MALD_PD['DISHWASHER']

# ## Lemmalex

LEMMALEX = load.tsv('/Volumes/Seagate BarracudaFastSSD/Corpus/lemmalex.txt')

LEMMAS = [w.upper() for w in LEMMALEX['Item']]

# ## Burgundy

# +
path = Path('../../Burgundy/Burgundy_item_data.txt')
with path.open('rt') as file:
    file.readline()
    BURGUNDY_WORDS = [line.split('\t', 1)[0].upper() for line in file]

print(f"{len(BURGUNDY_WORDS)} – {', '.join(BURGUNDY_WORDS[:10])}, ...")
# -

dst = Path('/Volumes/Seagate BarracudaFastSSD/EARSHOT/Burgundy/Burgundy-items.txt')
dst.write_text('\n'.join(sorted(BURGUNDY_WORDS)))

print(', '.join([w for w in BURGUNDY_WORDS if w not in LEMMAS]))

print(', '.join(BURGUNDY_WORDS))

# ## SUBTLEX/CMUPD/ Lexicon

SUBTLEX = {word: item['FREQcount'] for word, item in trftools.dictionaries.read_subtlex().items()}
CMUPD = trftools.dictionaries.read_cmupd(strip_stress=True)

missing_burgundy = [w for w in BURGUNDY_WORDS if w not in SUBTLEX]
print(f"{len(missing_burgundy)} missing from SUBTLEX")

CMUPD['ANATHEMA'], MALD_PD['SUMMONS']

PRONUNCIATIONS = {
    **CMUPD,
    **MALD_PD,
}

lexicon = speech.lexicon.generate_lexicon(PRONUNCIATIONS, SUBTLEX, default_activation=1)  # Some stimulus words are not in SUBTLEX

# # Lexicon Statistics
# ## Find all neighbors

NEIGHBOR_FILE = Path('burgundy_neighbors.pickle')
if NEIGHBOR_FILE.exists():
    NEIGHBORS = load.unpickle(NEIGHBOR_FILE)
else:
    NEIGHBORS = {}
    for word, pronunciation in MALD_PD.items():
        NEIGHBORS[word] = lexicon.neighbors(pronunciation)
    save.pickle(NEIGHBORS, NEIGHBOR_FILE)

# +
all_neighbors = set()
for word, word_neighbors in NEIGHBORS.items():
    all_neighbors.update([n.graphs for n in word_neighbors])

mald_neighbors = all_neighbors.intersection(MALD_WORDS)
print(f"Total of {len(all_neighbors)} neighbors, {len(mald_neighbors)} in MALD")  # , sum(len(ns) for ns in neighbors.values())
# -

# ## Subset of neighbors

# +
# in subtlext
min_count = 100
# min_count = 1000
frequent_neighbors = {word: [n for n in ns if SUBTLEX.get(n.graphs, 0) > min_count] for word, ns in NEIGHBORS.items()}
mald_neighbors = {word: [n for n in ns if n.graphs in MALD_WORDS] for word, ns in frequent_neighbors.items()}

all_neighbors = set(BURGUNDY_WORDS)
for word, ns in frequent_neighbors.items():
    all_neighbors.update([word.graphs for word in ns])
# -

print(f"Total of {len(all_neighbors)} words, including neighbors with frequency count > {min_count}")
mean_count = sum(len(ns) for ns in frequent_neighbors.values()) / len(frequent_neighbors)
print(f"Average of {mean_count:.2f} neighbors per target")
all_mald_neighbors = all_neighbors.intersection(MALD_WORDS)
print(f"Of those in MALD: {len(all_mald_neighbors)}")
mean_count = sum(len(ns) for ns in mald_neighbors.values()) / len(mald_neighbors)
print(f"Average of {mean_count:.2f} neighbors per target")

NEIGHBORS['SUMMONS'], frequent_neighbors['SUMMONS']

# ## Evaluate surprisal in small lexicon
# Compare to theoretical surprisal


small_words = all_neighbors.union(BURGUNDY_WORDS)
pronunciations = {w: PRONUNCIATIONS[w] for w in small_words}
small_lexicon = speech.lexicon.generate_lexicon(pronunciations, SUBTLEX, default_activation=1)  # Some stimulus words are not in SUBTLEX

# Same without frequency
small_lexicon_nf = speech.lexicon.generate_lexicon(pronunciations, {}, default_activation=1)  # Some stimulus words are not in SUBTLEX

# Same with log frequency
SUBTLEX_LOG = {word: item['Lg10WF'] for word, item in trftools.dictionaries.read_subtlex().items()}
small_lexicon_lf = speech.lexicon.generate_lexicon(pronunciations, SUBTLEX_LOG, default_activation=0)  # Some stimulus words are not in SUBTLEX

p = PRONUNCIATIONS[BURGUNDY_WORDS[0]]
print(lexicon.surprisal(p))
print(small_lexicon.surprisal(p))
print(small_lexicon_nf.surprisal(p))
print(small_lexicon_lf.surprisal(p))

rows = []
for word in BURGUNDY_WORDS:
    pronunciation = PRONUNCIATIONS[word]
    s = lexicon.surprisal(pronunciation)
    if len(s) == 1:
        continue
    ss = small_lexicon.surprisal(pronunciation)
    r_s = numpy.corrcoef(s, ss)[0, 1]
    ss_nf = small_lexicon_nf.surprisal(pronunciation)
    r_s_nf = numpy.corrcoef(s, ss_nf)[0, 1]
    ss_lf = small_lexicon_lf.surprisal(pronunciation)
    r_s_lf = numpy.corrcoef(s, ss_lf)[0, 1]
    
    e = lexicon.entropy(pronunciation)
    es = small_lexicon.entropy(pronunciation)
    r_e = numpy.corrcoef(e, es)[0, 1]
    es_nf = small_lexicon_nf.entropy(pronunciation)
    r_e_nf = numpy.corrcoef(e, es_nf)[0, 1]
    es_lf = small_lexicon_lf.entropy(pronunciation)
    r_e_lf = numpy.corrcoef(e, es_lf)[0, 1]
    
    rows.append((word, r_s, r_e, r_s_nf, r_e_nf, r_s_lf, r_e_lf))
corr_ds = Dataset.from_caselist(['word', 'r_surprisal', 'r_entropy', 'r_surprisal_nf', 'r_entropy_nf', 'r_surprisal_lf', 'r_entropy_lf'], rows)

corr_ds

corr_ds.n_cases

# +
import seaborn
from matplotlib import pyplot

df = corr_ds.as_dataframe()

figure, axes = pyplot.subplots(1, 3, figsize=(18, 6))
args = dict(data=df, fill=True, cut=0, clip=(-1, 1))
seaborn.kdeplot(x='r_surprisal', y='r_entropy', ax=axes[0], **args)
seaborn.kdeplot(x='r_surprisal_nf', y='r_entropy_nf', ax=axes[1], **args)
seaborn.kdeplot(x='r_surprisal_lf', y='r_entropy_lf', ax=axes[2], **args)
axes[0].set_title('Actual Frequencies')
axes[1].set_title('All Frequencies = 1')
axes[2].set_title('Log(Frequency)')
for ax in axes:
    ax.set_ylim(-1, 1)
    ax.set_xlim(-1, 1)
# -

# ## Save lexicon
# Export word and frequency

small_lexicon

DST = DATASET_ROOT / f'MALD-NEIGHBORS-{min_count}.txt'
if not DST.exists():
    DST.write_text('\n'.join(f'{word}\t{int(SUBTLEX[word])}' for word in sorted(all_mald_neighbors)))
DST = DATASET_ROOT / f'MALD-NEIGHBORS-{min_count}.pickle'
if not DST.exists():
    save.pickle(small_lexicon, DST)

# # Fix homophones
# Remove homophones

MALD_PATH = '../earshot/data/Words-MALD-NEIGHBORS-1000.txt'
data = load.tsv(MALD_PATH, names=['word', 'count'])

assert all(w in data['word'] for w in BURGUNDY_WORDS)

index = data['word'].isnotin(BURGUNDY_WORDS)

data[index, 'count'].min()

lexicon = cohort.lexicon.generate_lexicon({word: MALD_PD[word] for word in data['word']})

homophones = lexicon.find_homophones()
ns = set([len(group) for group in homophones.values()])
print(f"{len(homophones)} homophones, groups of {ns}")

remove = []
merge_to = {}
for p, (w1, w2) in homophones.items():
    in_b = '   '
    arrow = '-->'
    if SUBTLEX[w1] >= SUBTLEX[w2]:
        w_high, w_low = w1, w2
    else:
        w_high, w_low = w2, w1
    if w_low in BURGUNDY_WORDS:
        if w_high in BURGUNDY_WORDS:
            in_b = 'BB '
            merge_to[w_low] = w_high
        else:
            in_b = 'B  '
            arrow = '<--'
            remove.append(w_high)
    else:
        if w_high in BURGUNDY_WORDS:
            in_b = ' B '
        remove.append(w_low)
            
    print(in_b, p, w_low, arrow, w_high)

data['homophone'] = Factor(['rm' if w in remove else merge_to.get(w, '') for w in data['word']])

data.sub("homophone")

data.save_txt(MALD_PATH, header=False) 
