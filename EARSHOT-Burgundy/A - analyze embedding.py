# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.6
#   kernelspec:
#     display_name: EARSHOT
#     language: python
#     name: earshot
# ---

# # Setup
# Test whether proximity of embedding reflects cohort neighborhood status

# +
from collections import defaultdict
from pathlib import Path

import colormath
from eelbrain import *
import numpy
import matplotlib
from matplotlib import pyplot
import scipy.spatial
import seaborn
import tensorflow
import trftools
from trftools.align import TextGrid

import earshot
from earshot.earshot_lexicon import PHONES, read_pronunciations
from earshot.train_earshot import Model

import rc

MALD_DIR = Path('~/Data/Corpus/MALD').expanduser()
MALD_GRID_DIR = MALD_DIR / 'fixed-words'

ARGS = dict(n_bands=64, batch_size=32, steps_per_epoch=25, seed=0)
ARGS_50 = dict(n_bands=64, batch_size=32, steps_per_epoch=50, patience=250, seed=0)
# -

# ## Load model

# +
trainer = Model('MALD-1000-train', hidden='512', target_space='OneHot', **ARGS)
model = trainer.load_model()

dense = model.layers[-1]

b = numpy.asarray(dense.weights[1])
b.shape

w = numpy.asarray(dense.weights[0])
w.shape

w_inv = scipy.linalg.pinv(w)
w_inv.shape
# -

# ## Load pronunciations

# +
mald_pd = read_pronunciations('MALD-NEIGHBORS-1000', split_phones=True)
mald_pd

subtlex = trftools.dictionaries.read_subtlex()
subtlex = {word: subtlex[word]['Lg10WF'] for word in mald_pd}


# -

# ## N-shared matrix

# +
def n_shared(a, b):
    for i in range(min(len(a), len(b))):
        if a[i] != b[i]:
            break
    return i


n_words = trainer.lexicon.n_words
i, j = n_words - 2, n_words - 1
n_pdist = n_words * i + j - ((i + 2) * (i + 1)) // 2
print(n_pdist)
n_shared_matrix = numpy.zeros(int(n_pdist) + 1)

for j, word_j in enumerate(trainer.lexicon.words):
    for i, word_i in enumerate(trainer.lexicon.words[:j]):
        # dist(u=X[i], v=X[j]) is stored in entry m * i + j - ((i + 2) * (i + 1)) // 2.
        index = int(n_words * i + j - ((i + 2) * (i + 1)) // 2)
        n_shared_matrix[index] = n_shared(mald_pd[word_i], mald_pd[word_j])
# -

# # Neighbors
# For main text examples

# load GloVe
GLOVE = earshot.glove.read_glove(300)

WORDS = list(mald_pd)
VECTORS = numpy.array([GLOVE[word] for word in WORDS])

# ## find all cohorts

sum(GLOVE['UNFORTUNATELY']**2)

(VECTORS ** 2).sum(1).mean()

VECTORS.shape


def neighbors(word=None, vector=None, n=10):
    if word is not None:
        assert vector is None
        vector = GLOVE[word]
    pdist = scipy.spatial.distance.cdist(vector[None,:], VECTORS)[0]
    argsort = numpy.argsort(pdist)
    return [WORDS[i] for i in argsort[1:1+n]] 


', '.join(neighbors('VALLEY'))  # VALLEY VALUE

', '.join(neighbors('VALUE'))  # VALLEY VALUE

# +
N = 3

cohorts = defaultdict(list)
for word, pronunciation in mald_pd.items():
    if len(pronunciation) >= N:
        cohorts[tuple(pronunciation[:N])].append(word)

# +
n_identity = 0
n_different = 0

for prefix, words in cohorts.items():
    if len(words) > 1:
        word_list = sorted(words, key=lambda w: subtlex[w], reverse=True)

        glove_mean = sum([GLOVE[word] for word in word_list]) / len(word_list)
        pdist = scipy.spatial.distance.cdist(glove_mean[None,:], VECTORS)[0]
        argsort = numpy.argsort(pdist)
        mean_words = [WORDS[i] for i in argsort[:3]] 
        if mean_words[0] in word_list:
            n_identity += 1
            continue
        n_different += 1
        
        print(f"{' '.join(prefix)}: {', '.join(word_list)}  --> {', '.join(mean_words)}")
print(f"{n_identity} mean is identical with a target")
print(f"{n_different} mean is different")
# -

pdist.shape

mald_pd['BEETLE']

# # Test sorting words phonologically

# +
# Sort words sich that they are sorted by phones
# - cohort neighbors adjacent
# - phones separated by vowel and consonant
pronunciations = list(mald_pd.values())

def key(
    index: int,  # lexicon word intex
):  # --> List of PHONE indices
    phones = pronunciations[index]
    phone_is = [PHONES.index(phone) for phone in phones]
    return phone_is

phone_sort_index_b = sorted(range(trainer.lexicon.n_words), key=key)

# [trainer.lexicon.words[i] for i in phone_sort_index]
# -

numpy.all(phone_sort_index_b == trainer.lexicon.phone_sort)

# # Describe embedding
#
# `w` is a (256, 2985) weight matrix
# - 2985 words
# - 256 inputs
#
# Input
# - LSTM hidden nodes have `tanh` activation function -> input will be between (-1, 1)
#
# Output is one-hot
# - Sigmoid activation -> automatically in [0, 1]
# - --> The goal is to move the output up or down
#
#
# Question:
# - How likely are two candidates to be co-activated?
# - How close are two candidates if the inputs are considered a space?

# ## Weight image

pyplot.figure(figsize=(20, 5))
pyplot.imshow(w, interpolation='none', aspect='auto')
pyplot.title("w")
pyplot.ylabel('Hidden unit')
pyplot.xlabel('Word')

# ## Weight histograms
# - Bias is inhibiting all words
# - Some inputs drive all words down/up; More likely down, because 
#   - most words have to be 0 most of the time
#   - there 
#   - $tanh$ output $[-1, 1]$ to $\sigma$ activation $[0, 1]$

fig, axes = pyplot.subplots(1, 2, figsize=(15, 3))
seaborn.kdeplot(b, ax=axes[0])
axes[0].set_title('Bias')
seaborn.kdeplot(w.ravel(), ax=axes[1])
axes[1].set_title('Weights')

# ## Inputs with all same sign

# Some inputs push all words into the same direction
# They are likely inhibiting words during silence
print(f"Weights that are exactly 0 in all of w:  {(w == 0).sum()}")
all_same = (w.max(1) <= 0) | (w.min(1) >= 0)
print(f"Inputs for which all weights have the same sign:  {all_same.sum()}")

b.shape

# +
fig, axes = pyplot.subplots(1, 2, figsize=(15, 3))
seaborn.kdeplot(w[all_same].T, ax=axes[0], legend=False)
axes[0].set_title('Weights for variable sign inputs')

seaborn.kdeplot(w[~all_same].T, ax=axes[1], legend=False)
axes[1].set_title('Weights for constant sign inputs')

# +
# Selectiveness: inputs that increase a few words while inhibiting most
greater_0 = w > 0
ratio = greater_0.sum(1) / w.shape[1]

ax = seaborn.histplot(ratio[~all_same])
ax.axvline(0.5, color='red')
ax.set_xlabel("Ratio")
ax.set_title("Ratio of positive weights (in variable sign inputs)")
# axes[1].set_title('Weights')
# -

# ## Subset of inputs (HUs)
# Weights in random inputs 

fig, axes = pyplot.subplots(3, 10, figsize=(15, 6), sharex=True, sharey=True)
for i, ax in enumerate(axes.ravel()):
    seaborn.kdeplot(w[i*2:i*2+2].T, ax=ax, legend=False)
# pyplot.title('Each input')

# # Fig: Pairwise distances
# Analysis?
# - Word pairwise distance, correlate with phonetic edit distance
#
# Scipy `pdist`:
# - for each `i` and `j` where `i < j` 
# - `m` is the number of original observations. 
# - `dist(u=X[i], v=X[j])` is stored in entry `m * i + j - ((i + 2) * (i + 1)) // 2.`

pdist = scipy.spatial.distance.pdist(w.T)
pdist.shape

# ## Figure

# +
PLOT_INDEX = n_shared_matrix < 8
FIG_ARGS = dict(frameon=False)
R_INDEXES = {
    'Full': slice(None),
    'Up to 3': n_shared_matrix < 4,
    'Up to 4': n_shared_matrix < 5,
    '5+': n_shared_matrix >= 5,
}


def plot_pdist(hidden=None, color='.7', glove=None, pdist=None, ytick_distance=2, size=2, top=None, **model_args):
    if pdist is None:
        if glove is not None:
            trainer_ = Model('MALD-1000-train', hidden=hidden or '512', target_space=f'Glove-{glove}c', **ARGS)
            w_ = trainer_.lexicon.word_target.embedding
        else:
            trainer_ = Model('MALD-1000-train', hidden=hidden, target_space='OneHot', **model_args)
            model_ = trainer_.load_model()
            dense_ = model_.layers[-1]
            w_ = numpy.asarray(dense_.weights[0]).T
        pdist = scipy.spatial.distance.pdist(w_)
    # stats
    tests = {}
    for label, index in R_INDEXES.items():
        tests[label] = corr = test.Correlation(pdist[index], n_shared_matrix[index])
        # r, p = scipy.stats.pearsonr(pdist[index], n_shared_matrix[index])
        # print(f"{label}: {r=:.2f}, {p=:.3f}")
        print(f"{label:7}: {corr}")

    # figure
    bottom = .24 if size < 2 else .22
    left = .26 if size < 2 else .22
    figure, ax = pyplot.subplots(
        figsize=(size, size),
        gridspec_kw=dict(bottom=bottom, left=left),
        **FIG_ARGS)
    seaborn.violinplot(x=n_shared_matrix[PLOT_INDEX], y=pdist[PLOT_INDEX], color=color, ax=ax)
    ax.set_ylim(0, top)
    ax.grid(axis='y', color='0.8')  # , linestyle='--'
    if hidden is not None:
        pyplot.title(hidden)
    pyplot.ylabel('Euclidean distance')
    pyplot.xlabel('Shared onset phonemes', loc='right')
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(ytick_distance))
    ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%i'))
    seaborn.despine()
    return tests


plot_pdist('512', **ARGS)
pyplot.title('Learned space (512)')
# pyplot.savefig(rc.DST / 'Output space - pdist - 512 OneHot.pdf')
plot.figure_outline()
# -

# ### Deep models

HIDDEN = ['512', '320x320', '256x256x256', '192x192x192x192']
for hidden in HIDDEN:
    tests = plot_pdist(hidden, size=1.7, top=10, **ARGS)
    n = hidden.count('x') + 1
    r = tests['Full'].r
    r_str = f'{r:.2f}'[2:]
    pyplot.title(f"{n} layer ($r=-{r_str}$)")
    if hidden != HIDDEN[0]:
        pyplot.ylabel('')
        pyplot.xlabel('')
    pyplot.savefig(rc.DST / f'Output space - pdist - comp-{hidden}.pdf')

# ### Cohort loss

HIDDEN = ['512', '320x320', '256x256x256', '192x192x192x192']
LOSS = 1024
for hidden in HIDDEN:
    tests = plot_pdist(hidden, size=1.7, top=8.1, **ARGS, loss=f'dw{LOSS}to10')
    n = hidden.count('x') + 1
    r = tests['Full'].r
    r_str = f'{r:.2f}'[2:]
    pyplot.title(f'{n} layers ($r=-{r_str}$)')
    if hidden != HIDDEN[0]:
        pyplot.ylabel('')
        pyplot.xlabel('')
    pyplot.savefig(rc.DST / f'Output space - pdist - comp-dw{LOSS}-{hidden}.pdf')

# ### GloVe

# Color = desaturated Glove color
rgb = colormath.color_objects.sRGBColor.new_from_rgb_hex('#56B4E9')
hsl = colormath.color_conversions.convert_color(rgb, colormath.color_objects.HSLColor)
hsl.hsl_l = 0.7
rgb = colormath.color_conversions.convert_color(hsl, colormath.color_objects.sRGBColor)
glove_color = rgb.get_value_tuple()

plot_pdist(glove=50, color=glove_color)
pyplot.title('GloVe (N=50)')

plot_pdist(glove=300, color=glove_color, ytick_distance=4)
pyplot.title('GloVe (N=300)')
pyplot.savefig(rc.DST / 'Output space - pdist - Glove-300.pdf')

# # DEV - Glove

trainer_glove_50 = Model('MALD-1000-train', hidden='2048', target_space='Glove-50c', **ARGS)
model_glove_50 = trainer_glove_50.load_model()

# + jupyter={"outputs_hidden": false}
trainer_glove_50.lexicon.word_target.embedding.shape

# + jupyter={"outputs_hidden": false}
pdist_glove_50 = scipy.spatial.distance.pdist(trainer_glove_50.lexicon.word_target.embedding)
# -

scipy.stats.pearsonr(pdist_glove_50, n_shared_matrix)

# +

# Figure
figure, ax = pyplot.subplots(**fig_kw)
seaborn.violinplot(x=n_shared_matrix[index], y=pdist_glove_50[index], color=color, ax=ax)
ax.set_ylim(0, None)
ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(5))
pyplot.title('Glove (N=50)')
pyplot.ylabel('Distance')
pyplot.xlabel('N shared onset phonemes')
ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%i'))

pyplot.savefig(rc.DST / 'Output space - pdist - Glove-50.pdf')
plot.figure_outline()
# -

# ## Matrix

# ### Target: n shared phones

# +
n_shared_sq = scipy.spatial.distance.squareform(n_shared_matrix)
# numpy.fill_diagonal(pdist_sq, 0) already 0

kwargs = dict(interpolation='none', vmin=0, vmax=7, cmap='jet')
fig, axes = pyplot.subplots(1, 3, figsize=(7, 3), width_ratios=[10,10,1])
axes[1].sharey(axes[0])
axes[1].sharex(axes[0])
axes[0].set_title("Alphabetical order")
axes[0].imshow(n_shared_sq, **kwargs)
axes[1].set_title("Phone based order")
im = axes[1].imshow(n_shared_sq[phone_sort_index][:, phone_sort_index], **kwargs)
pyplot.colorbar(im, cax=axes[2])
_ = pyplot.setp(axes[1].get_yticklabels(), visible=False)
# -

# ### Distances

trainer_ = Model('MALD-1000-train', hidden='192x192x192x192x192x192-96', target_space='OneHot', loss='dw256to20', **ARGS_50)
VMIN, VMAX = 1, 6
# VMIN, VMAX = 1, 6
model_ = trainer_.load_model()
dense_ = model_.layers[-1]
w_ = numpy.asarray(dense_.weights[0]).T
pdist = scipy.spatial.distance.pdist(w_)

pdist_sq = scipy.spatial.distance.squareform(pdist)[phone_sort_index][:, phone_sort_index]

pdist.min(), pdist.max()

# numpy.fill_diagonal(pdist_sq, numpy.nan)
numpy.fill_diagonal(pdist_sq, 10)

numpy.sum(pdist < 3)

VMIN, VMAX = 0, 10
pyplot.imshow(pdist_sq[phone_sort_index][:, phone_sort_index], cmap='afmhot_r', vmin=VMIN, vmax=VMAX, interpolation='none')
pyplot.colorbar()

# Normalized for each word
pdist_sq_n = pdist_sq.copy()
pdist_sq_n /= pdist_sq_n.sum(1)[:, None] / (trainer.lexicon.n_words - 1)
numpy.fill_diagonal(pdist_sq_n, 2)

pyplot.imshow(pdist_sq_n[phone_sort_index][:, phone_sort_index], cmap='afmhot_r', vmin=0, vmax=1.5, interpolation='none')
pyplot.colorbar()

pdist_sqpdist_sq.shape

# ## Split by constant sign



# ## Inverse of W
# Why?

pyplot.figure(figsize=(20, 5))
pyplot.imshow(w_inv.T, interpolation='none', aspect='auto')
pyplot.title("inv(w)")
pyplot.ylabel('Hidden unit')
pyplot.xlabel('Word')

# # Compare with GloVe

trainer_glove = Model('MALD-1000-train', hidden='2048', target_space='Glove-50c', **ARGS)

pyplot.figure(figsize=(20, 3))
pyplot.imshow(trainer_glove.lexicon.word_target.embedding.T, aspect='auto', interpolation='none')
trainer_glove.lexicon.word_target.embedding.shape

pdist_glove = scipy.spatial.distance.pdist(trainer_glove.lexicon.word_target.embedding)

scipy.stats.pearsonr(pdist_glove, n_shared_matrix)


