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

# Other figure components in the following scripts:
# - Embedding structure: `A - analyze embedding.ipynb`
# - Cohort competition: `A - Cohort activation.ipynb`
#
# # Setup

# +
from functools import partial

from eelbrain import *
from eelbrain._stats.stats import dispersion
from eelbrain.plot._colors import adjust_hls
import matplotlib
from matplotlib import pyplot
import numpy
import seaborn

import trftools.roi

import earshot
from earshot.find_competitors import read_competitors
from earshot.train_earshot import Model, Word, gen_input_from_words, gen_rand_data
from earshot.report import Example

from jobs import e, WHOLEBRAIN
from constants2 import model
from data import load_performance, load_model_roi_data, load_roi_data, load_variance_components, load_variance_component_tests
from rc import LOSS, LOSS_COLORS
import rc


TEST = dict(smooth=0.005, metric='det')
TRF_TEST = dict(smooth=0.005)

CORTEX = ('.8', '.6')
ROI_COLOR = plot.unambiguous_color('blue')#, 100, 100)


ARGS = dict(n_bands=64, batch_size=32, steps_per_epoch=25, seed=0, patience=200)

HIDDEN_1_LAYER = ['512', '1024', '2048']

HIDDEN_FLAT = [
    '512', '320x320', '256x256x256', '192x192x192x192',
]

LAYER_COLORS = plot.colors_for_oneway([1, 2, 3, 4, 5, 6, 7], cmap='pink')

# https://davidmathlogic.com/colorblind/#%23000000-%23E69F00-%2356B4E9-%23009E73-%23F0E442-%230072B2-%23D55E00-%23CC79A7
OUTPUT_COLORS = {
    'Auditory': '#808080',
    'Glove-50c': '#56B4E9', 
    'Glove-300c': '#0072B2', 
    'Sparse-10of300': '#E69F00', 
    'Sparse-10of900': '#D55E00',
    'OneHot': '#000000', 
}
LABELS = {
    'Acoustic': 'Acoustic',
    'Glove-50c': 'GloVe 50 dim', 
    'Glove-300c': 'GloVe 300 dim', 
    'Sparse-10of300': 'SRV 10 of 300', 
    'Sparse-10of900': 'SRV 10 of 900',
    'OneHot': 'Localist', 
    'sequential': 'Sequential',
    'skip': 'Skip connections',
    'constant': 'c = 1 (original)',
}
for loss in [16, 64, 256, 1024, 4096]:
    LABELS[f'dw{loss}to10'] = f'c = {loss}'


def get_sem_func(data, model, y='det_roi', match='subject'):
    sem = dispersion(y, model, match, 'SEM', True, data=data)
    
    def sem_func(x):
        x_mean = x.mean()
        return x_mean - sem, x_mean + sem
    
    return sem_func



# -

# # Color schemes

p = plot.ColorList(OUTPUT_COLORS, list(OUTPUT_COLORS)[1:], labels=LABELS, shape='line', linewidth=1)
p.save(rc.DST / 'Colors Output.pdf')
plot.figure_outline()

styles = {
    'New tokens': plot.Style('k', linestyle='-'),
    'Trained tokens': plot.Style('k', linestyle='--'),
}
p = plot.ColorList(styles, shape='line', w=1.3, linewidth=1)#.2)
p.save(rc.DST / 'Colors cv-set.pdf')
plot.figure_outline()

# + jupyter={"outputs_hidden": false}
colors = {k: OUTPUT_COLORS[k] for k in ['OneHot', 'Glove-50c', 'Glove-300c']}
p = plot.ColorList(colors, labels=LABELS, shape='line', w=1., linewidth=1)
p.save(rc.DST / 'Colors Output Subset.pdf')
plot.figure_outline()
# -

p = plot.ColorList(LAYER_COLORS, [1, 2, 3, 4][::-1], shape='line', w=1.1, linewidth=1)
p.save(rc.DST / 'Colors Layers.pdf')
plot.figure_outline()

# + jupyter={"outputs_hidden": false}
p = plot.ColorList(LOSS_COLORS, labels=LABELS, shape='line', w=1.1, linewidth=1)
p.save(rc.DST / 'Colors Loss R1.pdf')
plot.figure_outline()
# -

styles = {
    'Flat (K-means)': plot.Style('k', linestyle='--'),
    'Deep': plot.Style('k', linestyle='-'),
}
p = plot.ColorList(styles, shape='line', w=1.3, linewidth=1)
# p.save(rc.DST / 'Colors deep vs k-means.pdf')
plot.figure_outline()

# # Model activation
# ## Load data

# +
import burgundy


events = burgundy.e.load_selected_events()

meg_ds = burgundy.e.load_epochs()
meg = meg_ds[0, 'meg']
# -

trainer = Model('MALD-1000-train', hidden='512', target_space='OneHot', **ARGS)
example = Example(trainer)

n_words = trainer.lexicon.n_words
cmap = matplotlib.colormaps.get_cmap("jet")
WORD_COLORS = cmap.resampled(n_words)(range(n_words))

# +
FIRST_EVENT = 0
DURATION = 120  # s

PAD_START = 1  # s of silence at beginning; can be cut for plot
T_MIN = events[FIRST_EVENT, 'T']
T_MAX = T_MIN + DURATION
SRATE = 100
LAST_SAMPLE = DURATION * SRATE

WORDS = []
for t, word in events[FIRST_EVENT:].zip('T', 'item'):
    if t > T_MAX:
        break
    i0 = int(round((t - T_MIN + PAD_START) * SRATE))
    word = Word(i0, 'MALD', word.upper(), trainer.lexicon)
    WORDS.append(word)

onsets = [t for word in WORDS for t in [word.t0, word.t1] if t < LAST_SAMPLE]

WORDS[:10]
# -

# ## Model Overview

_, data = example._predict(words=WORDS)
# sort outputs by phoneme
data['targets'] = data['targets'][:, trainer.lexicon.phone_sort]
data['outputs'] = data['outputs'][:, trainer.lexicon.phone_sort]

# +
HU_COLOR = '#E69F00'  # '#FE6100'
HU_ON_COLOR = '#D55E00'
ENV_COLOR = '#304F81'  # '#785EF0'

TSTART = 43 # 60 # 4.2 + 15
DURATION = 15

def normalize(y):
    out = y - y.min()
    out /= out.max()
    return out


START = int(TSTART * SRATE)
STOP = START + int(DURATION * SRATE) + 1
height_ratios = [
    2,  # MEG
    2,  # Audio
    2,  # HU
    2,  # HU summary
    2,  # Output
]
gridspec_kw = dict(height_ratios=height_ratios, left=.11, bottom=.12, top=.98, right=.98, hspace=.75)
figure, axes = pyplot.subplots(len(height_ratios), 1, sharex=True, figsize=(6.8, 3.5), gridspec_kw=gridspec_kw)
a = 0

# MEG
ax = axes[a]
ax.plot(meg.x[::10, START:STOP].T, color='k')
ax.set_ylabel('MEG (T)')
ax.set_yticks(())
seaborn.despine(ax=ax, left=True, bottom=True)
ax.tick_params(bottom=False)
a += 1

# Audio
vmax = 0.05
ax = axes[a]
ax.imshow(data['inputs'][START:STOP].T, origin='lower', aspect='auto', cmap='Blues', vmin=.001, vmax=.07)  # , vmax=vmax, cmap='viridis')#, cmap='inferno')
ax.set_ylabel('Stimulus\n(Hz)')
ax.set_yticks([0, 63], ['50', '10k'])
seaborn.despine(ax=ax, bottom=True, trim=True)
ax.tick_params(bottom=False)
a += 1

# hidden units
vmax = 0.5
ax = axes[a]
y = data['hidden'][START:STOP].T
ax.imshow(y, origin='lower', aspect='auto', cmap='bwr', vmin=-vmax, vmax=vmax)
ax.set_ylabel('Hidden\nunits')
ax.set_yticks([1, y.shape[0]])
seaborn.despine(ax=ax, bottom=True, trim=True)
ax.tick_params(bottom=False)
a += 1

# Summary
ax = axes[a]
# envelope
# y = numpy.sum(inputs[START:STOP], 1)
# ax.plot(normalize(y), label='Acoustic envelope', color=ENV_COLOR, linestyle='--', linewidth=1)
# hidden magnitude
hidden = data['hidden'][START-1:STOP]
y = numpy.abs(hidden[1:]).sum(1)
y_on = numpy.diff(abs(hidden), axis=0)
y_on = numpy.clip(y_on, 0, None).sum(1)
ax.plot(normalize(y), label='Magnitude', color=HU_COLOR, linewidth=1)
ax.plot(normalize(y_on), label='Change', color=HU_ON_COLOR, linewidth=1, linestyle='--')
ax.legend(loc=(.825,.5))#'right')
# format
ax.set_ylabel('RNN\nactivity')
seaborn.despine(ax=ax, bottom=True, trim=True)
ax.tick_params(bottom=False)
# legend = figure.legend(loc=4, bbox_to_anchor=(1, 0.2), framealpha=1)
a += 1

# output units
ax = axes[a]
ax.set_prop_cycle(color=WORD_COLORS)
output_lines = ax.plot(data['outputs'][START:STOP])
ax.set_ylabel('Word\nrecognition')
ax.set_yticks((0, 1))
ax.set_ylim(0)
ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(5 * SRATE))
ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, pos: f'{x/SRATE:g}'))
seaborn.despine(ax=ax, trim=True)
a += 1

# Information
axes[-1].set_xlabel('Time (s)')
# figure.tight_layout()
figure.align_ylabels(axes)

# figure.savefig(rc.DST / f'Model Overview.pdf')
# plot.figure_outline()

# +
# With training target:
for line in output_lines:
    line.remove()

ax.set_prop_cycle(color=WORD_COLORS)
output_lines = ax.plot(data['targets'][START:STOP])

figure.savefig(rc.DST / f'Model Overview - targets.pdf')
figure
# -

# Steps for building figure
axes[3].remove()
figure.savefig(rc.DST / f'Model Overview - 2.pdf')
axes[2].remove()
figure.savefig(rc.DST / f'Model Overview - 1.pdf')
axes[4].remove()
figure.savefig(rc.DST / f'Model Overview - 0.pdf')

# ## Different output spaces
# Used to explain model architecture for different output spaces

e_onehot = Example(Model('MALD-1000-train', target_space='OneHot', hidden='2048', **ARGS))
e_glove = Example(Model('MALD-1000-train', target_space='Glove-50c', hidden='2048', **ARGS))

# +
plot_args = dict(
    h=2.5, w=1.4, times=400, seed=9,
    gridspec_kw=dict(left=.15, right=0.95, bottom=.18, top=.98, hspace=4.5),
    cmap='Blues',
)

fig, words = e_glove.output(**plot_args)
fig.savefig(rc.DST / f'Output example 2 split glove.pdf')
plot.figure_outline()

fig, words = e_onehot.output(**plot_args)
for ax in fig.axes:
    ax.set_ylabel('')
ax.set_xlabel('')
ax.set_yticks(())
fig.savefig(rc.DST / f'Output example 2 split onehot.pdf')
plot.figure_outline()
# -

# # Model performance

# + jupyter={"outputs_hidden": false}
performance_args = dict(legend=False, linewidth=1, marker=None, y='wer', units='speaker')
PERFORMANCE_ARGS = dict(**ARGS, by_speaker=True)#, stimuli='Sil Inf')

# + [markdown] jupyter={"outputs_hidden": false}
# ## Output space
# -

OUT = ['Sparse-10of300', 'Sparse-10of900', 'Glove-50c', 'Glove-300c', 'OneHot']
ds = load_performance(hidden=HIDDEN_1_LAYER, target_space=OUT, **PERFORMANCE_ARGS)
ds['dataset'] = Factor(ds['trained'], labels={1: 'train', 0: 'test'})
df = ds.as_dataframe()

figure, ax = pyplot.subplots(
    figsize=(1.6, 2), gridspec_kw=dict(left=.3, wspace=1., right=0.9, bottom=.2, hspace=1., top=0.95), 
)
sds = ds.sub("~trained")
sem = get_sem_func(sds, 'target_space % hidden', 'wer', 'speaker')
p = seaborn.pointplot(
    data=sds.as_dataframe(), x='hidden', hue='target_space',
    ax=ax, errorbar=sem, palette=OUTPUT_COLORS, hue_order=OUTPUT_COLORS, **performance_args,
)
sds = ds.sub("trained")
sem = get_sem_func(sds, 'target_space % hidden', 'wer', 'speaker')
p = seaborn.pointplot(
    data=sds.as_dataframe(), x='hidden', hue='target_space',
    ax=ax, errorbar=sem, palette=OUTPUT_COLORS, hue_order=OUTPUT_COLORS, **performance_args,
    linestyles='--',
)
ax.set_xlim(-0.2, 2.2)
ax.set_xlabel('Hidden units')
ax.set_ylim(0, 0.75)
ax.set_ylabel('Word error rate (%)')
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, 0, ''))
ax.grid(color='k', alpha=.1)
seaborn.despine(figure)
pyplot.savefig(rc.DST / '1 layer performance.pdf')
plot.figure_outline()

# ## Deep models

# + jupyter={"outputs_hidden": false}
ds = load_performance(hidden=HIDDEN_FLAT[:4], target_space=['OneHot', 'Glove-50c', 'Glove-300c'], **PERFORMANCE_ARGS)
# -

figure, ax = pyplot.subplots(
    figsize=(1.6, 2), gridspec_kw=dict(left=.3, wspace=1., right=0.9, bottom=.2, hspace=1., top=0.95), 
)
sds = ds.sub("~trained")
sem = get_sem_func(sds, 'target_space % layers', 'wer', 'speaker')
p = seaborn.pointplot(
    data=sds.as_dataframe(), x='layers', hue='target_space',
    ax=ax, errorbar=sem, palette=OUTPUT_COLORS, hue_order=OUTPUT_COLORS, **performance_args,
)
sds = ds.sub("trained")
sem = get_sem_func(sds, 'target_space % layers', 'wer', 'speaker')
p = seaborn.pointplot(
    data=sds.as_dataframe(), x='layers', hue='target_space',
    ax=ax, errorbar=sem, palette=OUTPUT_COLORS, hue_order=OUTPUT_COLORS, **performance_args,
    linestyles='--',
)
ax.set_xlim(-0.2, 3.2)
ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))
ax.set_xlabel('Layers')
ax.set_ylim(0, 0.75)
ax.set_ylabel('Word error rate (%)')
ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.1))
ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.05))
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, 0, ''))
ax.grid(color='k', alpha=.1)
ax.grid(color='k', alpha=.1, which='minor', linestyle='--')
seaborn.despine(figure)
pyplot.savefig(rc.DST / 'Performance - depth with glove R1.pdf')
plot.figure_outline()

# + [markdown] jupyter={"outputs_hidden": false}
# ## Cohort loss
# -

ds = load_performance(hidden=HIDDEN_FLAT[:4], loss=LOSS, target_space='OneHot', **PERFORMANCE_ARGS)

# + jupyter={"outputs_hidden": false}
figure, ax = pyplot.subplots(
    figsize=(1.6, 2), gridspec_kw=dict(left=.3, wspace=1., right=0.9, bottom=.2, hspace=1., top=0.95), 
)
sds = ds.sub("~trained")
sem = get_sem_func(sds, 'loss % layers', 'wer', 'speaker')
p = seaborn.pointplot(
    data=sds.as_dataframe(), x='layers', hue='loss',
    ax=ax, errorbar=sem, palette=LOSS_COLORS, hue_order=LOSS_COLORS, **performance_args,
)
sds = ds.sub("trained")
sem = get_sem_func(sds, 'loss % layers', 'wer', 'speaker')
p = seaborn.pointplot(
    data=sds.as_dataframe(), x='layers', hue='loss',
    ax=ax, errorbar=sem, palette=LOSS_COLORS, hue_order=LOSS_COLORS, **performance_args,
    linestyles='--',
)
ax.set_xlim(-0.2, 3.2)
ax.set_xlabel('Layers')
ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))
ax.set_ylim(0, 0.5)
ax.set_ylabel('Word error rate (%)')
ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.05))
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, 0, ''))
ax.grid(color='k', alpha=.1)
ax.grid(color='k', alpha=.1, which='minor', linestyle='--')
seaborn.despine(figure)
pyplot.savefig(rc.DST / 'Performance - cohort loss R1.pdf')
plot.figure_outline()
# -

# ### Tests

test.ANOVA('wer', 'loss * layers.as_factor() * speaker', "~trained", ds)

for layers in range(1, 5):
    result = test.TTestRelated('wer', 'loss', 'dw16to10', 'constant', 'speaker', f"~trained & (layers == {layers})", ds)
    display(result)

# # MEG anatomical
#

# ## Output space
# Test whether Glove output models are better than localist models anywhere
#
# ### GloVe > Localist

BASE = "gt-log8 + phone-p0"
COMPARISONS = {
    f"{hidden}": f"{BASE} + {model(hidden, target_space='Glove-50c')} > {BASE} + {model(hidden, target_space='OneHot')}"
    for hidden in ['512', '1024', '2048']
}
e.show_model_test(COMPARISONS, **WHOLEBRAIN, smooth=0.005)

# ### Localist > GloVe

BASE = "gt-log8 + phone-p0"
COMPARISONS = {
    f"{hidden}": f"{BASE} + {model(hidden, target_space='Glove-50c')} < {BASE} + {model(hidden, target_space='OneHot')}"
    for hidden in ['512', '1024', '2048']
}
e.show_model_test(COMPARISONS, **WHOLEBRAIN, smooth=0.005)

# # MEG ROI

# +
ROI = 'STG301'
Y_LABEL = "Neural prediction (%∆)"
MEG_Y = 'det_roi_delta'
MEG_TRIM = 0.2

roi_args = dict(legend=False, markersize=0, linewidth=1)

ds_acoustic = load_roi_data("gt-log8 + phone-p0", ROI)
ds_acoustic[:, 'target_space'] = 'Auditory'
# -


# ## ROI label

# from appleseed":
roi_label = trftools.roi.mne_label(ROI, subjects_dir=e.get('mri-sdir'))
color = plot.unambiguous_color('blue')
for hemi in ['lh', 'rh']:
    brain = plot.brain.brain('fsaverage', surf='pial', cortex=CORTEX, hemi=hemi, w=400, h=300, mask=False)
    brain.add_label(getattr(roi_label, hemi), color, lighting=True)
    brain.save_image(DST / f'ROI {ROI} {hemi}-v.png', 'rgba', fake_transparency='white')
    brain.close()


# + [markdown] jupyter={"outputs_hidden": false}
# ## Flat - Output space

# + jupyter={"outputs_hidden": false}
TARGET = ['Glove-50c', 'Glove-300c', 'Sparse-10of300', 'Sparse-10of900', 'OneHot']
ds = load_model_roi_data(ROI, hidden=HIDDEN_1_LAYER, target_space=TARGET)
ds['det_roi_delta'] = ds['det_roi'] - 1
sem = get_sem_func(ds, 'hidden % target_space')
df = ds.as_dataframe()

# + jupyter={"outputs_hidden": false}
figure, ax = pyplot.subplots(
    figsize=(1.6, 2), gridspec_kw=dict(left=.3, wspace=1., right=0.9, bottom=.2, hspace=1., top=0.95), 
)
ax = seaborn.pointplot(data=df, y=MEG_Y, x='hidden', hue='target_space', units='subject', palette=OUTPUT_COLORS, errorbar=sem, **roi_args)#, ax=p.ax)
ax.set_xlim(-MEG_TRIM, 2 + MEG_TRIM)
ax.set_ylim(0, .04)
ax.set_ylabel(Y_LABEL)
ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.01))
ax.grid(color='.8')
ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.01))
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0, symbol=''))
ax.set_xlabel('Hidden units')
seaborn.despine()
pyplot.savefig(rc.DST / 'MEG ~ n, output space.pdf')
plot.figure_outline()

# + [markdown] jupyter={"outputs_hidden": false}
# ### Overall ANOVA

# + jupyter={"outputs_hidden": false}
test.ANOVA('det_roi', "hidden * target_space * subject", data=ds)
# -

test.ANOVA('det_roi', "hidden * target_space * subject", sub="target_space.isin(('Sparse-10of300', 'Sparse-10of900'))", data=ds)

test.ANOVA('det_roi', "hidden * target_space * subject", sub="target_space.isin(('OneHot', 'Glove-50c'))", data=ds)

test.ANOVA('det_roi', "hidden * target_space * subject", sub="target_space.isin(('OneHot', 'Glove-300c'))", data=ds)

# + jupyter={"outputs_hidden": false}
for hidden in HIDDEN_1_LAYER:
    doc = test.pairwise('det_roi', 'target_space', 'subject', f"hidden == '{hidden}'", data=ds, corr=False, title=f"Pairwise - {hidden}")
    display(doc)

# -

test.pairwise('det_roi', 'hidden', 'subject', f"target_space == 'OneHot'", data=ds, corr=False, title=f"Pairwise - Localist ~ n_units")

test.TTestRelated('det_roi', 'hidden % target_space', ('512', 'OneHot'), ('1024', 'Sparse-10of900'), 'subject', data=ds)

test.TTestRelated('det_roi', 'hidden % target_space', ('512', 'OneHot'), ('1024', 'Sparse-10of300'), 'subject', data=ds)

test.TTestRelated('det_roi', 'hidden % target_space', ('512', 'OneHot'), ('1024', 'Glove-50c'), 'subject', data=ds)

test.TTestRelated('det_roi', 'hidden % target_space', ('512', 'OneHot'), ('2048', 'Glove-50c'), 'subject', data=ds)

# ## Deep models

# ### Deep Vs. K-means

dfs = {}
for target in ['OneHot', 'Glove-50c', 'Glove-300c']:
    ds_deep = load_model_roi_data(hidden=HIDDEN_FLAT[:4], target_space=target, k=None)
    ds_deep['predictors'] = ds_deep['layers']#.astype(int)
    k = [2, 3, 4, 8, 16, 32, 64]
    ds_flat = load_model_roi_data(hidden='512', target_space=target, k=k)  # [2, 4, 8, 16, 32])#, 64])
    ds_flat['predictors'] = ds_flat['k']

    for kind, ds in [('deep', ds_deep), ('flat', ds_flat)]:
        ds['det_roi_delta'] = ds['det_roi'] - 1    
        dsa = ds.aggregate(f'predictors % subject', drop=['hemi'])
        sem = get_sem_func(dsa, 'hidden % k')
        df = ds.as_dataframe()
        dfs[target, kind] = df, sem

dsa.aggregate('hidden % k', drop_bad=True)

# +
figure, ax = pyplot.subplots(
    figsize=(3, 2), gridspec_kw=dict(left=.3, wspace=1., right=0.95, bottom=.2, hspace=1., top=0.95), 
)
for (target, kind), (df, sem) in dfs.items():
    seaborn.pointplot(
        data=df, y=MEG_Y, x='predictors', units='subject', 
        errorbar=sem, color=OUTPUT_COLORS[target],
        linestyles={'deep': '-', 'flat': '--'}[kind],
        ax=ax,
        **roi_args)
ax.set_ylim(0, 0.1)
ax.set_ylabel(Y_LABEL)
ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.01))
ax.grid(color='.8')
ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.01))
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0, symbol=''))
ax.set_xlabel('Layers/K')

seaborn.despine(figure)
pyplot.savefig(rc.DST / 'MEG ~ n predictors.pdf')
plot.figure_outline()
# -

# ### Deep, ~K % layers
# Use K=32 for all

dss = {}
for target_space in ['OneHot', 'Glove-50c', 'Glove-300c']:
    k = [2, 4, 8, 16, 32, 64]
    w = 1.8
    ds = load_model_roi_data(hidden=HIDDEN_FLAT[:4], target_space=target_space, k=k)

    ds['det_roi_delta'] = ds['det_roi'] - 1
    ds['k_'] = ds['k'].astype(int).as_factor()
    ds['layers_'] = ds['layers'].astype(int).as_factor()    
    dsa = ds.aggregate(f'layers_ % k_ % subject', drop=['hemi'])
    sem = get_sem_func(dsa, 'layers_ % k_')
    dsa['k'] = dsa['k'].astype(int)
    df = dsa.as_dataframe()

    figure = pyplot.figure(figsize=(w, 2))
    pyplot.subplots_adjust(left=.3, bottom=.2, right=.95, top=.9)
    # args = {**roi_args, 'legend': 'auto'}
    ax = seaborn.pointplot(data=df, y=MEG_Y, x='k', hue='layers', units='subject', errorbar=sem, palette=LAYER_COLORS, **roi_args)#, ax=p.ax)
    ax.set_title(f"{LABELS[target_space]}")
    ax.set_ylim(0, 0.10)
    # ax.set_ylabel(Y_LABEL)
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.01))
    ax.grid(color='.8')
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.01))
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0, symbol=''))
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0, symbol=''))
    if target_space == 'OneHot':
        ax.set_ylabel(Y_LABEL)
    else:
        ax.set_ylabel('')
        # ax.yaxis.set_ticklabels([])
    ax.set_xlabel('K')
    
    seaborn.despine(figure)
    pyplot.savefig(rc.DST / f'MEG ~ K % n layers {target_space}.pdf')
    plot.figure_outline()
    dss[target_space] = dsa
    # break



ds = dss['OneHot']
test.ANOVA('det_roi', 'k_ * layers_ * subject', data=ds)

test.TTestRelated('det_roi', 'layers_ % k_', ('1', '32'), ('3', '64'), 'subject', data=dss['OneHot'])

# ### Test of best models

ds = combine([
    load_model_roi_data(hidden=HIDDEN_FLAT[0], target_space='OneHot', k=32),
    load_model_roi_data(hidden=HIDDEN_FLAT[3], target_space='Glove-50c', k=16),
    load_model_roi_data(hidden=HIDDEN_FLAT[3], target_space='Glove-300c', k=32),
])
ds['det_roi_delta'] = ds['det_roi'] - 1
dsa = ds.aggregate(f'target_space % subject', drop=['hemi'])
sem = get_sem_func(dsa, 'target_space')
# df = ds.as_dataframe()

test.pairwise('det_roi', 'target_space', match='subject', data=dsa)

# ## Cohort loss

# ### Deep, k=32

ds = combine([
    load_model_roi_data(hidden=HIDDEN_FLAT[:4], target_space='OneHot', loss=LOSS, k=32),
], incomplete='drop')
ds['det_roi_delta'] = ds['det_roi'] - 1
dsa = ds.aggregate(f'hidden % loss % subject', drop=['hemi'])
sem = get_sem_func(dsa, 'hidden % loss')
df = dsa.as_dataframe()

# +
figure, ax = pyplot.subplots(
    figsize=(1.6, 2), gridspec_kw=dict(left=.3, wspace=1., right=0.9, bottom=.2, hspace=1., top=0.95), 
)
# figure = pyplot.figure(figsize=(2.2, 2))
seaborn.pointplot(data=df, y=MEG_Y, x='layers', hue='loss', units='subject', palette=LOSS_COLORS, errorbar=sem, **roi_args, ax=ax)
ax.set_ylim(0, 0.12)
ax.set_ylabel(Y_LABEL)
ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.01))
ax.grid(color='.8')
ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.02))
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0, symbol=''))
ax.set_xlabel('Layers')

seaborn.despine(figure)
pyplot.savefig(rc.DST / 'MEG ~ cohort loss R1.pdf')
plot.figure_outline()
# -

test.TTestRelated('det_roi', 'layers', '1', '4', match='subject', sub="loss == 'dw1024to10'", data=dsa)

# ### Pick best mode for each c

dss = []
dsa_averages = dsa.aggregate('loss % layers', drop_bad=True)
for loss in dsa_averages['loss'].cells:
    ds_loss = dsa_averages.sub(f"loss == '{loss}'")
    i_max = ds_loss['det_roi'].argmax()
    best_layers = ds_loss[i_max, 'layers']
    dss.append(dsa.sub(f"(loss == '{loss}') & (layers == {best_layers})"))
best_data = combine(dss)

test.pairwise('det_roi', 'loss', 'subject', data=best_data, corr='Bonferroni')

# ### by loss

# +
figure = pyplot.figure(figsize=(2.2, 2))
pyplot.subplots_adjust(left=.3, bottom=.2, right=.95, top=.95)
ax = seaborn.pointplot(data=df, y=MEG_Y, x='loss', hue='layers', units='subject', palette=LAYER_COLORS, errorbar=sem, **roi_args)#, ax=p.ax)
# ax = seaborn.pointplot(data=df, y=MEG_Y, x='loss', units='subject', errorbar=sem, **roi_args)#, ax=p.ax)
ax.set_ylim(0, 0.12)
ax.set_ylabel(Y_LABEL)
ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.01))
ax.grid(color='.8')
ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.02))
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0, symbol=''))
ax.set_xlabel('Layers')
ax.tick_params('x', rotation=45)

seaborn.despine(figure)
# pyplot.savefig(rc.DST / 'MEG ~ cohort loss by loss.pdf')
plot.figure_outline()
# -

# #### Average all losses

dsa_base = dsa.sub("loss_n == '0'")
del dsa_base['loss_n']
dsa_base[:, 'loss'] = 'constant'
dsa_loss = dsa.sub("loss_n != '0'").sub("loss != 'dw16to10'").sub("loss != 'dw4096to10'").aggregate("subject % layers", drop_bad=True)
dsa_loss[:, 'loss'] = 'modified'
dsa_loss_binary = combine([dsa_base, dsa_loss])

# +
palette = {
    'constant': LOSS_COLORS['constant'],
    'modified': LOSS_COLORS['dw256to10'],
}
df = dsa_loss_binary.as_dataframe()
figure = pyplot.figure(figsize=(2.2, 2))
pyplot.subplots_adjust(left=.3, bottom=.2, right=.95, top=.95)
ax = seaborn.pointplot(data=df, y=MEG_Y, x='layers', hue='loss', units='subject', palette=palette, errorbar=sem, **roi_args)#, ax=p.ax)
# ax = seaborn.pointplot(data=df, y=MEG_Y, x='loss', units='subject', errorbar=sem, **roi_args)#, ax=p.ax)
ax.set_ylim(0, 0.12)
ax.set_ylabel(Y_LABEL)
ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.01))
ax.grid(color='.8')
ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.02))
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0, symbol=''))
ax.set_xlabel('Layers')

seaborn.despine(figure)
# pyplot.savefig(rc.DST / 'MEG ~ cohort loss avergae.pdf')
plot.figure_outline()
# -

for layers in range(1, 5):
    result = test.TTestRelated(MEG_Y, 'loss', 'modified', 'constant', 'subject', sub=f"layers == {layers}", data=dsa_loss_binary)
    print(result)

test.ANOVA('det_roi', 'subject * hidden * loss', data=dsa_loss_binary)

test.ANOVA('det_roi', 'subject * hidden * loss', data=dsa)

test.ANOVA('det_roi', 'subject * hidden * loss', data=dsa, sub="loss.isin(('constant', 'dw1024to10'))")

for loss in [64, 256, 1024]:
    print(loss)
    for hidden in dsa['hidden'].cells:
        print(hidden)
        res = test.TTestRelated('det_roi', 'loss', f'dw{loss}to10', 'constant', 'subject', sub=f"hidden == '{hidden}'", data=dsa)
        print(res)
        # print(f'corrected: {res.p * 8:.4f}')
    print()

# ## Variance partitioning

# +
# norm: divided by baseline model in each hemisphere
rnn = partial(model, HIDDEN_FLAT[0], target_space='OneHot', loss=f'dw1024to10', k=32)

ds_full = load_roi_data(f"gt-log8 + phone-p0 + {rnn()}")
dss_reduced = {
    'onset': load_roi_data(f"gt-log8 + phone-p0 + {rnn(transform='sum')}"),
    'sum': load_roi_data(f"gt-log8 + phone-p0 + {rnn(transform='onset')}"),
    'rnn': load_roi_data(f"gt-log8 + phone-p0"),
    'auditory': load_roi_data(f"phone-p0 + {rnn()}"),
}
# Variance component in full 'det_roi' attributable to key:
for key, ds_reduced in dss_reduced.items():
    ds_full[key] = ds_full['det_roi'] - ds_reduced['det_roi']
# Drop hemisphere
hemi_ds = ds_full.aggregate("subject", drop='hemi')
# -

# ### RNN sum vs. onset

value_rnn = hemi_ds['rnn'].mean()  # Attributable to RNN
print(f"RNN: {value_rnn:.1%}, thereof:")
for key in ['onset', 'sum']:  # 
    value = hemi_ds[key].mean()
    prop = value / value_rnn
    print(f"{key}: {prop:.1%}")
    print(test.TTestOneSample(key, 'subject', data=hemi_ds))


# ### RNN vs auditory

value_full = hemi_ds['det_roi'].mean()
for x in ['rnn', 'auditory']:
    print(x)
    value = hemi_ds[x].mean()
    prop = value / value_full
    print(f"{prop:.1%} of full")
    print(test.TTestOneSample(x, 'subject', data=hemi_ds))


# ### By hemi: RNN vs auditory

# long ds for RNN ANOVA
long_dss = []
for key in ['rnn', 'auditory']:
    ds = ds_full['subject', 'hemi']
    ds[:, 'x'] = key
    ds['det_roi'] = ds_full[key]
    long_dss.append(ds)
long_ds = combine(long_dss)

test.ANOVA('det_roi', 'subject * hemi * x', data=long_ds)

p = plot.Barplot('det_roi', 'hemi % x', 'subject', data=long_ds, corr=False, h=3)

# ### By hemi: sum/onset

# long ds for RNN ANOVA
long_dss = []
for key in ['onset', 'sum']:
    ds = ds_full['subject', 'hemi']
    ds[:, 'x'] = key
    ds['det_roi'] = ds_full[key]
    long_dss.append(ds)
long_ds = combine(long_dss)

test.ANOVA('det_roi', 'subject * hemi * x', data=long_ds)

# Barely significant cross-over

p = plot.Barplot('det_roi', 'hemi % x', 'subject', data=long_ds, corr=False, h=3)
