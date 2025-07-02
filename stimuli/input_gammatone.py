# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Gammatone inputs
# Convert wave files to gammatone spectrograms

# +
from itertools import product
from pathlib import Path
from random import Random

import tqdm
from eelbrain import *
import librosa
import numba
import numpy
import trftools

INPUT_ROOT = Path('/Volumes/Seagate BarracudaFastSSD/EARSHOT')
WAV_ROOT = INPUT_ROOT / 'EARSHOT_NEW_WAVS'


@numba.jit(nopython=True)
def find_first(vec):
    for i in range(len(vec)):
        if vec[i] != 0:
            return i
    return 0


@numba.jit(nopython=True)
def find_last(vec):
    for i in range(len(vec) - 1, 0, -1):
        if vec[i] != 0:
            return i + 1
    return 0



# -

# # Silence

waves = {}
for path in WAV_ROOT.glob('*/*.WAV'):
    wave, srate = librosa.core.load(path)
    start = find_first(wave)
    stop = find_last(wave)
    wave2 = wave[start: stop]
    duration = len(wave)
    if (duration - stop) > 1000:
        wave = wave2
        uts = UTS(0, 1/srate, len(wave))
        waves[path.stem] = NDVar(wave, uts, name=path.stem)
        if len(waves) >= 9:
            break

plot.UTS(waves.values(), ncol=3)

gts = [trftools.gammatone_bank(wav, 50, 10000, n=100, tstep=0.001) for wav in waves.values()]
gts = [trftools.pad(gt, -0.010, gt.time.tstop+0.010) for gt in gts]
p = plot.Array(gts, ncol=3, interpolation='none', vmax=0.1)

gts = [trftools.gammatone_bank(trftools.pad(wav, -0.010, wav.time.tstop+0.010), 50, 10000, n=100, tstep=0.001) for wav in waves.values()]
p = plot.Array(gts, ncol=3, interpolation='none', vmax=0.1)
display(p)

# + [markdown] tags=[]
# # Examples
# -

waves = {}
for speaker in ['PRINCESS', 'FRED']:
    for word in ['LARK', 'LARD', 'LAST']:
        path = WAV_ROOT / speaker.title() / f'{word}_{speaker}.WAV'
        wave, srate = librosa.core.load(path)
#         wave, trim = librosa.effects.trim(wave, frame_length=32, hop_length=16)
        uts = UTS(0, 1/srate, len(wave))
        waves[speaker, word] = NDVar(wave, uts, name=path.stem)


p = plot.UTS(waves.values(), ncol=3)


# ## N Bands
#
# Assuming frequency resolution of 1/6 octave (Milekhina et al., 2018), $f_{i+1} = \frac{7}{6} f_i$

def make(start, n=64):
    out = [start]
    for i in range(n):
        out.append(out[-1] * 9/8)
    print(' '.join(f'{f:.0f}' for f in out))


make(50, 50)

# ## 100 Bands

gts = [trftools.gammatone_bank(wav, 50, 10000, n=100, tstep=0.01) for wav in waves.values()]
p = plot.Array(gts, ncol=3)

# ## 50 Bands

gts = [trftools.gammatone_bank(wav, 50, 10000, n=50, tstep=0.01) for wav in waves.values()]
p = plot.Array(gts, ncol=3, interpolation='none')

# ## 64 bands

gts = [trftools.gammatone_bank(wav, 50, 10000, n=64, tstep=0.01) for wav in waves.values()]
p = plot.Array(gts, ncol=3, interpolation='none')

# # Generate all

# + tags=[]
n_bands = 50
res = 100  # Hz

tstep = 1 / res
DST_ROOT = INPUT_ROOT / f'GAMMATONE_{n_bands}_{res}'
DST_ROOT.mkdir(exist_ok=True)
for path in tqdm.tqdm(WAV_ROOT.glob('*/*.WAV'), total=15*2063):
    word, speaker = path.stem.split('_')
    wave, srate = librosa.core.load(path)
    # trim silence
    start = find_first(wave)
    stop = find_last(wave)
    wave = wave[start: stop]
    # NDVar
    uts = UTS(0, 1/srate, len(wave))
    wav = NDVar(wave, uts, name=path.stem)
    # gammatones
    gt = trftools.gammatone_bank(wav, 50, 10000, n=n_bands, tstep=tstep)
    dst_dir = DST_ROOT / path.parent.name
    dst_dir.mkdir(exist_ok=True)
    save.pickle(gt.x, dst_dir / f'{path.stem}.pickle')

# -



gts = [trftools.gammatone_bank(wav, 50, 10000, n, tstep=0.01) for n in (40, 50, 100)]

fmtxt.FloatingLayout([plot.Array(gt) for gt in gts])

# # Generate Burgundy

# +
BURGUNDY_ROOT = Path('/Volumes/Seagate BarracudaFastSSD/EARSHOT/Burgundy')
BURGUNDY_WAVE = BURGUNDY_ROOT / 'Wave'

MALD_SRC = Path('/Volumes/Seagate BarracudaFastSSD/Corpus/MALD/MALD1_rw')

LEXICON_FILE = BURGUNDY_ROOT / 'MALD-NEIGHBORS-1000.txt'
LEXICON = [line[:line.find('\t')] for line in LEXICON_FILE.open()]
BURGUNDY_ITEMS_FILE = BURGUNDY_ROOT / 'Burgundy-items.txt'

SPEAKERS = [path.name for path in BURGUNDY_WAVE.iterdir() if not path.name.startswith('.')]
SPEAKERS.append('MALD')

# +
n_bands = 64
res = 100  # Hz

tstep = 1 / res
DST_ROOT = BURGUNDY_ROOT / f'GAMMATONE_{n_bands}_{res}'
DST_ROOT.mkdir(exist_ok=True)

total = len(SPEAKERS) * len(LEXICON)
mald_data = []
# SPEAKERS = ['MALD']
for speaker, word in tqdm.tqdm(product(SPEAKERS, LEXICON), total=total):
    name = f'{word}_{speaker.upper()}'
    dst_dir = DST_ROOT / speaker
    dst_dir.mkdir(exist_ok=True)
    dst = dst_dir / f'{name}.pickle'
    if dst.exists():
        continue
    if speaker == 'MALD':
        src = MALD_SRC / f'{word}.wav'
    else:
        src = BURGUNDY_WAVE / speaker / f'{word.lower()}_{speaker}.wav'
    wave, srate = librosa.core.load(src)
    # trim silence
    start = find_first(wave)
    stop = find_last(wave)
    wave = wave[start: stop]
    # Store for making predictors
    if speaker == 'MALD':
        mald_data.append([word, start / srate, (stop - start) / srate])
#     continue
    # NDVar
    uts = UTS(0, 1/srate, len(wave))
    wav = NDVar(wave, uts, name=name)
    # gammatones
    gt = trftools.gammatone_bank(wav, 50, 10000, n=n_bands, tstep=tstep)
    save.pickle(gt.x, dst)

# -
# ## Delay due to cutting silence

# +
mald_ds = Dataset.from_caselist(['word', 'start', 'duration'], mald_data)
all_delays = dict(mald_ds.zip('word', 'start'))

burgundy_items = BURGUNDY_ITEMS_FILE.read_text().split()

{word: all_delays[word] for word in burgundy_items if all_delays[word]}
