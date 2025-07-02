# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Setup

# +
from itertools import repeat
from pathlib import Path

from eelbrain import *
import cohort.lexicon
import matplotlib
from matplotlib import pyplot
import numpy
import seaborn

from trftools.align import TextGrid
from trftools.dictionaries._arpabet import IPA

import earshot
from earshot.find_competitors import read_competitors
from earshot.earshot_lexicon import read_pronunciations
from earshot.train_earshot import Model
from earshot.report import Example

from data import GRID_DIR
import rc


ARGS = dict(n_bands=64, batch_size=32, steps_per_epoch=25)
HIDDEN_FLAT = ['512', '320x320', '256x256x256', '192x192x192x192']
# -

competitors = read_competitors('MALD-Neighbors-1000')
BURGUNDY_WORDS = earshot.earshot_lexicon.read_words('burgundy')
print(len(BURGUNDY_WORDS))
pronunciations = read_pronunciations('MALD-NEIGHBORS-1000', split_phones=True)
lexicon = cohort.lexicon.generate_lexicon(read_pronunciations('MALD-NEIGHBORS-1000'))

# +
ACTIVATION_COLORS = {
    'target': '#D55E00',
    'v-cohort': '#E69F00',
    'rhyme': '#CC79A7',
    'unrelated': plot.Style('#000000', linestyle=':'),
}
ACTIVATION_UTS_ARGS = dict(h=2, frame=False, bottom=0, xlim=(0, .6), clip=True, sub="(trained == False) & (kind != 'cohort')", colors=ACTIVATION_COLORS)
MALD_ONLY = {**ACTIVATION_UTS_ARGS, 'sub': "(trained == False) & (kind != 'cohort') & (speaker == 'MALD')"}

p = plot.ColorList(ACTIVATION_COLORS)

COLORS = {
    'unrelated': plot.Style('k', linestyle=(0, (1, 0.5))),  # linestyle=':'
    # **plot.colors_for_oneway(map(str, range(8)), cmap='viridis'),
    # 'target': '#D55E00',
    **plot.colors_for_oneway(map(str, range(1, 11)), cmap='plasma'),
    # 'target': 'k',
    # 'target': '#009E73',
    # 'rhyme': plot.Style(ACTIVATION_COLORS['rhyme'], linestyle=':'),
}
COLORS.pop('8')
COLORS['target'] = COLORS.pop('9')
# COLORS.pop('9')
COLORS.pop('10')

p = plot.ColorList(COLORS)
# -

# ## Competitors

# +
n_shared_phonemes = {}
for word, cdict in competitors.items():
    for i in range(8):
        if cdict[i]:
            n_shared_phonemes[word] = i

pyplot.hist(n_shared_phonemes.values(), numpy.arange(9)-0.5)
# -

for n_min in range(8):
    n_words = sum(n >= n_min for n in n_shared_phonemes.values())
    print(f"{n_min} or more shared: {n_words} words")


# # Figure 4
# ## Plotting

def plot_activation(
    ds_activation, 
    fold=None, 
    burgundy=None, 
    mald: bool = None,  # use only MALD items
    correct: bool = None,  # use only targets from correct (or incorrect) trials
    n_min=None, 
    axes=None, 
    bottom=None, 
    top=None,
):
    sub = ds_activation['kind'].isin(COLORS)
    # Test-set
    if fold == 'test':
        sub &= ds_activation.eval("trained == False")
    elif fold == 'train':
        sub &= ds_activation.eval("trained == True")
    elif fold:
        raise ValueError(f"{fold=}")
    # Burgundy words (for those I have the competitors)
    if burgundy is not None:
        sub_ = ds_activation['word'].isin(BURGUNDY_WORDS)
        if not burgundy:
            sub_ = ~sub_
        sub &= sub_
    # MALD speaker only
    if mald is not None:
        sub_ = ds_activation['speaker'] == 'MALD'
        if not mald:
            sub_ = ~sub_
        sub &= sub_
    # Correct response
    if correct is True:
        sub &= ds_activation['correct'].x
    elif correct is False:
        sub &= ~ds_activation['correct'].x
    else:
        assert correct is None
    # Minimum number shared phonemes        
    if n_min:
        n_min_vec = numpy.array([n_shared_phonemes[word] for word in ds_activation['word']])
        sub &= n_min_vec >= n_min
    # Plot
    if 'Glove-' in ds_activation.info['model']:
        args = dict(ylabel='Distance', bottom=0)
    else:
        args = dict(ylabel='Activation', top=1, bottom=-0.01)
    if top is not None:
        args['top'] = top
    if bottom is not None:
        args['bottom'] = bottom
    p = plot.UTSStat('activation', 'kind', sub=sub, data=ds_activation, h=2, w=2, frame=False, clip=True, colors=COLORS, legend=False, xlim=(0, .5), axes=axes, **args)
    ax = p.axes[0]
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.2))
    ax.grid()
    return p


# ## Load and prepare data
# - Sizing is hard-coded for 3 rows

# +
def load_data(
    spec: dict, 
    fold: str = None, 
    stimulus: str = 'NoSil Inf',
):
    kwargs = {'seed': 0, **spec}
    trainer = Model(**kwargs)
    example = Example(trainer)

    results, ds_correct, ds_softmax, ds_activation = example.load_activation(stimulus, pad=-1)
    if fold == 'test':
        ds_activation = ds_activation.sub("trained == False")
    elif fold == 'train':
        ds_activation = ds_activation.sub("trained")
    elif fold:
        raise ValueError(f"{FOLD=}")
    return ds_activation  # was in plot_data


def choice_probability(activation_data: Dataset):
    # Divide by total activation -> choice probability
    
    item_data = activation_data.sub("kind != 'total'")
    # print(data_items['activation'].min(), data_items['activation'].max())

    total_data = activation_data.sub("kind == 'total'")
    total_dict = dict(zip(total_data.zip('word', 'speaker'), total_data['activation']))
    for i, key in enumerate(item_data.zip('word', 'speaker')):
        item_data['activation'].x[i] /= total_dict[key].x
    return item_data  # was in plot_data_n


def relative_activation(choice_data: Dataset):
    # Relative activation of target by competitor length
    data = choice_data.sub("kind == 'target'")
    data['kind'] = Factor([n_shared_phonemes[word] for word in data['word']])
    return data  # was in plot_data_target


# load test data for plot development
spec = dict(lexicon_name='MALD-1000-train', hidden=HIDDEN_FLAT[0], target_space='OneHot', **ARGS)
activation_data = load_data(spec)
choice_data = choice_probability(activation_data)
target_data = relative_activation(choice_data)


# -


# ## Plot: With response probability
#
# <div class="alert alert-block alert-info">
# ⚠️ Not every word has `N=7` cohort competitors. In words that do, total activation is going to be higher. 
#     Hence, when normalizing by total activation, the average of `N=7` will drop more than the average of `target`.
#     Effectively, `N=7` will be divided by a larger number than `total`.
# </div>
#

# +
def plot_both(
    activation_data: Dataset,
    target_data: Dataset,
    mald: bool = None,  # use only MALD items
    correct: bool = None,  # use only targets from correct (or incorrect) trials
    top: float = 1,
    ylabel: str = None,
    x_labels: bool = False,
    y_labels: bool = False,
    y_ticks: bool = False,
    fname: str = None,
):
    figure, axes = pyplot.subplots(
        1, 2, figsize=(2., 0.9), sharex='col', sharey='row',
        gridspec_kw=dict(left=.16, wspace=1., right=0.95, bottom=.4, hspace=1., top=0.9), 
    )
    
    args = dict(mald=mald, correct=correct, bottom=0, top=top)
    # Activation values
    plot_activation(activation_data, axes=axes[0], **args)
    # Relative activation of target
    p = plot_activation(target_data, axes=axes[1], **args)

    for ax in axes:
        ax.tick_params('both', length=0, which='both')
        ax.set_xlabel('')
        ax.set_xticks([0.1, 0.2, 0.3, 0.4], minor=True)
        if not x_labels:
            ax.tick_params('x', labelbottom=False)

        ax.set_xticks([0.000, 0.500])
        ax.grid(which='both', color='0.8')
        ax.set_ylabel('')
        ax.set_yticks([0, top])
        ax.tick_params('y', length=0, which='both')
        if not y_ticks:
            ax.tick_params('y', labelleft=False)

    if y_labels:
        axes[0].set_ylabel('Activation')#, loc='top')
        axes[1].set_ylabel('Probability')#, loc='bottom')
    elif ylabel:
        axes[0].set_ylabel(ylabel)#, loc='top')
    if x_labels:
        axes[0].set_xlabel('Time (ms)')#, loc='top')

    if fname:
        figure.savefig(rc.DST / f'Cohort activation R2 {fname}.pdf')

# Test plots
############
# plot_both(activation_data, target_data, x_labels=True)
# plot.figure_outline()
# plot_both(activation_data, target_data, y_labels=True, y_ticks=True)
# plot.figure_outline()
# plot_both(activation_data, target_data, y_ticks=True)
# plot.figure_outline()
# plot_both(activation_data, target_data)
# plot.figure_outline()


# +
def make_plot(
    stimulus: str = 'NoSil Inf',
    fold: str = None,
    mald: bool = None,  # use only MALD items
    correct: bool = None,  # use only targets from correct (or incorrect) trials
    top: float = 1,
    ylabel: str = None,
    x_labels: bool = False,
    y_labels: bool = False,
    y_ticks: bool = False,
    fname: str = None,
    **spec,
):
    activation_data = load_data(spec, fold, stimulus)
    choice_data = choice_probability(activation_data)
    target_data = relative_activation(choice_data)
    plot_both(activation_data, target_data, mald, correct, top, ylabel, x_labels, y_labels, y_ticks, fname)


# What to plot (uncomment section to plot)
args = dict(hidden=HIDDEN_FLAT[0], target_space='OneHot', **ARGS)
args_4l = dict(hidden=HIDDEN_FLAT[3], target_space='OneHot', **ARGS)

# MALD stimuli
##############
make_plot(lexicon_name='MALD-1000-train', **args, mald=True, 
          x_labels=True, y_labels=True, y_ticks=True, fname='MALD-all')
make_plot(lexicon_name='MALD-1000-test', **args, mald=True, fold='test', 
          fname='MALD-test')
make_plot(lexicon_name='MALD-1000-test', **args, mald=True, fold='test', correct=True, 
          fname='MALD-test-correct')

# Synthetic stimuli
###################
make_plot(lexicon_name='MALD-1000-train', **args, mald=False, fold='train',
          x_labels=True, y_labels=True, y_ticks=True, fname='syn-train')
make_plot(lexicon_name='MALD-1000-train', **args, mald=False, fold='test',
          fname='syn-test')
make_plot(lexicon_name='MALD-1000-train', **args, mald=False, fold='test', correct=True,
          fname='syn-test-correct')

# Cohort loss
#############
for c in [16, 64, 256, 1024, 4096]:
    make_plot(lexicon_name='MALD-1000-train', loss=f'dw{c}to10', **args, mald=True,
              ylabel=f'c={c}', fname=f'c{c}-MALD-all')
for c in [1024]:
    make_plot(lexicon_name='MALD-1000-train', loss=f'dw{c}to10', **args_4l, mald=True,
              fname=f'c{c}-4l-MALD-all')
    make_plot(lexicon_name='MALD-1000-train', loss=f'dw{c}to10', **args_4l, mald=True, correct=True,
              fname=f'c{c}-4l-MALD-all-correct')

# Cohort loss - test set
########################
for c in [1024]:
    make_plot(lexicon_name='MALD-1000-test', loss=f'dw{c}to10', **args_4l, mald=True, fold='test',
              fname=f'c{c}-4l-MALD-test')
    make_plot(lexicon_name='MALD-1000-test', loss=f'dw{c}to10', **args_4l, mald=True, fold='test', correct=True,
              fname=f'c{c}-4l-MALD-test-correct')


# + [markdown] jupyter={"outputs_hidden": false}
# # Colors

# + jupyter={"outputs_hidden": false}
labels = {
    'unrelated': '0',
    # '1': '1 shared phoneme',
    # '2': '2 ...',
    'target': 'Target',
}
p = plot.ColorList(COLORS, reversed(COLORS), w=1., labels=labels, shape='line', linewidth=1)#, label_position='left')
p.save(rc.DST / 'Cohort activation colors.pdf')
plot.figure_outline()

# + [markdown] jupyter={"outputs_hidden": false}
# # Competitor Spectrograms
# -

# For Lexicon
trainer = Model('MALD-1000-train', hidden='512', target_space='OneHot', seed=0, **ARGS)

# + jupyter={"outputs_hidden": false}
# Find word with a good set of competitors
from collections import defaultdict


all_n_shared = {}
for target in BURGUNDY_WORDS:
    n_shared = all_n_shared[target] = defaultdict(list)
    target_p = pronunciations[target]
    target_n = len(target_p)
    for word, pronunciation in pronunciations.items():
        if word == target:
            continue
        word_p = pronunciations[word]
        for n in range(min(target_n, len(word_p))):
            if word_p[n] != target_p[n]:
                break
        if n > 0:
            n_shared[n].append(word)

# + jupyter={"outputs_hidden": false}
for word, neighbors in all_n_shared.items():
    if len(neighbors) >= 5:
        print(f"{word}: {len(neighbors)}")

# + jupyter={"outputs_hidden": false}
pronunciations['CONSERVE'], pronunciations['CANNOT'], 

# + jupyter={"outputs_hidden": false}
WORDS = ['CONSERVE', 'CONCERNED', 'CONCEAL', 'CONFESS', 'CAR']  # , 'CHAMELEON'
N = 70
TARGET = WORDS[0]
TARGET_GRID = TextGrid.from_file(GRID_DIR / f'{TARGET}.TextGrid').strip_stress()

figure, axes = pyplot.subplots(len(WORDS), figsize=(2, 2))#, sharex=True)
pyplot.subplots_adjust(top=0.95, bottom=.2)

for word, ax in zip(WORDS, axes):
    x = trainer.lexicon.inputs['MALD', word]
    y = numpy.zeros((64, N))
    y[:, :len(x)] = x.T
    ax.imshow(y, cmap='Blues', vmin=0, vmax=0.05, aspect=1/5, origin='lower')
    # ax.set_yticks(())
    # ax.set_frame_on(False)
    ax.grid(False)
    grid = TextGrid.from_file(GRID_DIR / f'{word}.TextGrid').strip_stress()
    for t, ph in zip(grid.times, grid.phones):
        if ph != ' ':
            ax.text(t*100, 32, IPA[ph])#, color='white')
    
    if word != WORDS[-1]:
        seaborn.despine(ax=ax, left=True, bottom=True)
        ax.tick_params(bottom=False, left=False, labelleft=False, labelbottom=False)
    else:
        seaborn.despine(ax=ax, left=True)
        ax.tick_params(left=False, labelleft=False)

    if word == TARGET:
        continue
    for t, ph, ph_t in zip(grid.times, grid.phones, TARGET_GRID.phones):
        if ph != ph_t:
            ax.axvline(t*100, color='red', linestyle='--')
            break
            
ax = axes[1]
# ax.set_ylabel('Frequency')
ax = axes[-1]
ax.set_xlim(0, N)
ax.set_xlabel('Time (ms)')
ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(20))
ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, pos: f'{x*10:g}'))

pyplot.savefig(rc.DST / 'Cohort activation - illustration.pdf')
plot.figure_outline()
# -

# # TextGrid analysis of MALD stimuli

GRID_DIR = Path('~').expanduser() / 'Data' / 'Corpus' / 'MALD' / 'fixed-words'

GRIDS = {}
mald_word_phones = {}
mald_word_times = {}
mald_words = {}
for word in pronunciations:
    grid = TextGrid.from_file(GRID_DIR / f'{word}.TextGrid')
    GRIDS[word] = grid
    # grid = grid.strip_stress()
    phones = list(zip(grid.phones, grid.times))
    # cut initial silence
    while phones[0][0] == ' ':
        phones.pop(0)
    # t0 = phones[0][1]
    # phones = [(p, t-t0) for p, t in phones]
    mald_words[word] = phones
    mald_word_phones[word] = [p for p, t in phones]
    mald_word_times[word] = [t for p, t in phones]
    # print(phones)

stressed_lexicon = cohort.lexicon.generate_lexicon({word: ' '.join(pron).strip() for word, pron in mald_word_phones.items()})
stressed_lexicon.words[5]

# uniqueness point
i_up = 1
# 1: disambiguating phonems
# 2: pre-disambiguating phoneme (likely to contain co-articulation)
mald_ups = {}
for word, phones in mald_word_phones.items():
    cohort = list(mald_word_phones.values())
    for i in range(1, len(phones)+1):
        prefix = phones[:i]
        cohort = [c_phones for c_phones in cohort if c_phones[:i] == prefix]
        if len(cohort) == 1:
            mald_ups[word] = up = mald_word_times[word][i-i_up]
            break
    else:
        mald_ups[word] = up = mald_word_times[word][-i_up]
    print(f"{word:10}: {' '.join(prefix):20}; {up:.3f}")
    # if word == 'ABROAD':
    #     break

print(mald_word_phones['ACCENT'])
print(mald_word_times['ACCENT'])

pyplot.hist(mald_ups.values())

x = numpy.arange(0, .501, 0.01)
max_n_shared = max(n_shared_phonemes.values())
n_x = len(x)

# ## Target activation

# +
y = numpy.zeros_like(x)
for t in mald_ups.values():
    i = int(round(t * 100))
    y[i:] += 1
y /= len(mald_ups)

y_burgundy = numpy.zeros_like(x)
for word in BURGUNDY_WORDS:
    t = mald_ups[word]
    i = int(round(t * 100))
    y_burgundy[i:] += 1
y_burgundy /= len(BURGUNDY_WORDS)
# -

JOBS = {
    'all': y,
    'burgundy': y_burgundy,
}
for label, y_ in JOBS.items():
    fig, ax = pyplot.subplots(
        figsize=(0.85, 0.7),
        gridspec_kw=dict(left=.25, wspace=1., right=0.88, bottom=.3, hspace=1., top=0.91), 
    )
    ax.plot(x, y_, color=COLORS['target'])
    ax.set_clip_on(False)
    ax.set_ylim(0, 1.0)
    ax.set_xlim(0, 0.506)
    ax.set_xticks([0.1, 0.2, 0.3, 0.4], minor=True)
    ax.set_xticks([0.000, 0.500], [0, 500])
    ax.grid(which='both', color='0.8')
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_yticks([0, 1])
    ax.tick_params(which='both', length=0, labelbottom=False)
    # ax.tick_params(length=0, which='both')
    seaborn.despine(ax=ax)
    # fig.tight_layout()
    # pyplot.savefig(rc.DST / f'Cohort activation - theoretical-MALD-{label}.pdf')
    plot.figure_outline()

# ## UP ~ `n_shared`
#

y = numpy.zeros((max_n_shared, n_x))
y_burgundy = numpy.zeros_like(y)
for word, t in mald_ups.items():
    i = int(round(t * 100))
    n_shared = n_shared_phonemes[word]
    if n_shared:
        n_shared -= 1
    y[n_shared, i:] += 1
    if word in BURGUNDY_WORDS:
        y_burgundy[n_shared, i:] += 1
y /= y.max(1)[:,None]
y_burgundy /= y_burgundy.max(1)[:,None]

JOBS = {
    'all': y,
    'burgundy': y_burgundy,
}
for label, y_ in JOBS.items():
    figure, axes = pyplot.subplots(
        1, 2, figsize=(2., 0.9), sharex='col', sharey='row',
        gridspec_kw=dict(left=.16, wspace=1., right=0.95, bottom=.4, hspace=1., top=0.9), 
    )
    axes[0].remove()
    ax = axes[1]
    for n, y_n in enumerate(y_):
        color = COLORS[str(n+1)]
        ax.plot(x, y_n, color=color)
    ax.set_clip_on(False)
    ax.set_ylim(0, 1.0)
    ax.set_xlim(0, 0.50)
    ax.set_xticks([0.1, 0.2, 0.3, 0.4], minor=True)
    ax.set_xticks([0.000, 0.500], [0, 500])
    ax.grid(which='both', color='0.8')
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_yticks([0, 1], ['0', '1'])
    ax.tick_params('y', labelleft=True, length=0)
    if label == 'all':
        ax.set_ylabel('Proportion\nunique')
    else:
        ax.tick_params('x', which='both', length=0, labelbottom=False)
    seaborn.despine(ax=ax)
    # fig.tight_layout()
    pyplot.savefig(rc.DST / f'Cohort activation - theoretical-MALD-{label}-UP.pdf')
    plot.figure_outline()

# ## Simulate proportional activation
# At each time point:
# - target activation should be 1 / n_cohort

# +
n_lexicon = len(stressed_lexicon.words)
n_cohort = numpy.empty((n_lexicon, n_x))

for i_word, word in enumerate(stressed_lexicon.words):
    cohort_sizes = list(stressed_lexicon.cohort_size(word.pronunciations[0]))
    cohort_sizes.insert(0, n_lexicon)
    grid = GRIDS[word.graphs]
    ts = [int(round(t * 100)) for t in grid.times]
    ts.insert(0, 0)
    ts.append(int(round(grid.tstop * 100)))
    for i, size in enumerate(cohort_sizes):
        n_cohort[i_word, ts[i]:ts[i+1]] = size
    n_cohort[i_word, ts[i+1]:] = 1
# -

pyplot.imshow(n_cohort, aspect='auto')

data = Dataset({
    'word': Factor([word.graphs for word in stressed_lexicon.words]),
    'n_shared': Var([n_shared_phonemes[word.graphs] for word in stressed_lexicon.words]),
    'activation': NDVar(1 / n_cohort, (Case, UTS(0, 0.01, n_x))),
    'burgundy': Var([word.graphs in BURGUNDY_WORDS for word in stressed_lexicon.words]),
})

table.frequencies('n_shared', data=data)

data = data.sub("n_shared > 0")

JOBS = {
    'all': None,
    'burgundy': 'burgundy',
}
for label, sub in JOBS.items():
    figure, axes = pyplot.subplots(
        1, 2, figsize=(2., 0.9), sharex='col', sharey='row',
        gridspec_kw=dict(left=.16, wspace=1., right=0.95, bottom=.4, hspace=1., top=0.9), 
    )
    axes[0].remove()
    ax = axes[1]
    p = plot.UTSStat('activation', 'n_shared', sub=sub, colors=COLORS, legend=False, axes=ax, data=data)
    ax.set_clip_on(False)
    ax.set_ylim(0, 1.0)
    ax.set_xlim(0, 0.50)
    ax.set_xticks([0.1, 0.2, 0.3, 0.4], minor=True)
    ax.set_xticks([0.000, 0.500], [0, 500])
    ax.grid(which='both', color='0.8')
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_yticks([0, 1], ['0', '1'])
    if label == 'all':
        ax.set_ylabel('Probability')
        ax.tick_params('y', labelleft=True, length=0)
    else:
        ax.tick_params('x', which='both', length=0, labelbottom=False)
        ax.tick_params('y', labelleft=False, length=0)
    seaborn.despine(ax=ax)
    # fig.tight_layout()
    pyplot.savefig(rc.DST / f'Cohort activation - theoretical-MALD-{label}-by-n.pdf')
    plot.figure_outline()
