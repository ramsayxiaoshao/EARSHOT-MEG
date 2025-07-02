"""
Creates predictors for model activity.
Uses events exported from burgundy.
"""
from pathlib import Path

from eelbrain import *
import numpy

from earshot.train_earshot import Model, Word, gen_input_from_words

from constants2 import trainer as trainer_key


EVENT_DIR = Path('~/Data/Burgundy/events').expanduser()
PREDICTOR_DIR = Path('~/Data/Burgundy/predictors').expanduser()
ACTIVATION_DIR = Path('~/Data/Burgundy/model-activation').expanduser()
GAMMATONE_DIR = Path(f'~/Data/EARSHOT/Burgundy/GAMMATONE_64_100/MALD').expanduser()

# settings
PAD_START = 1
PAD_STOP = 2
SAMPLINGRATE = 100

# collect models
MODELS = {}

# 1. Output space base models
#############################
LEXICON = 'MALD-1000-train'
# LEXICON = 'MALD-1000-test'
# LEXICON = 'MALD-1000-testr'
SEED = 0
DO = {
    'OneHot': [
        '512', '1024', '2048',
        '320x320', '256x256x256', '192x192x192x192',
    ],
    'Sparse-10of900': [
        '512', '1024', '2048',
    ],
    'Sparse-10of300': [
        '512', '1024', '2048',
    ],
    'Glove-300c': [
        '512', '1024', '2048',
        '320x320', '256x256x256', '192x192x192x192',
    ],
    'Glove-50c': [
        '512', '1024', '2048',
        '320x320', '256x256x256', '192x192x192x192',
    ],
}
ARGS_LIST = [
    dict(steps_per_epoch=25),
    # dict(steps_per_epoch=50, patience=250),
]
print("** Collecting models **")
for target_space, hiddens in DO.items():
    for hidden in hiddens:
        for args in ARGS_LIST:
            model = Model(LEXICON, target_space=target_space, hidden=hidden, seed=SEED, n_bands=64, batch_size=32, **args)
            print(f"{model.name}")
            if not model.checkpoints:
                print(f" Skipping: No trained model")
                continue
            if (last_epoch := model.checkpoints[-1]) <= 1000:
                print(f" Skipping: Last epoch {last_epoch}")
                continue
            key = trainer_key(hidden, target_space=target_space, lexicon=LEXICON, seed=SEED)
            MODELS[key] = model
            break

# 2. Down-weighted error
######################################
target_space = 'OneHot'
for hidden in [
    '512', '320x320', '256x256x256', '192x192x192x192',
]:
    for loss_w in [16, 64, 256, 1024, 4096]:  # [None, 16, 64, 256, 1024, 4096]:
        loss = f'dw{loss_w}to10' if loss_w else None
        for args in ARGS_LIST:
            model = Model(LEXICON, target_space=target_space, hidden=hidden, loss=loss, seed=SEED, n_bands=64, batch_size=32, **args)
            print(f"{model.name}")
            if not model.checkpoints:
                print(f" Skipping: No trained model")
                continue
            if (last_epoch := model.checkpoints[-1]) <= 1000:
                print(f" Skipping: Last epoch {last_epoch}")
                continue
            key = trainer_key(hidden, loss, target_space=target_space, lexicon=LEXICON, seed=SEED)
            MODELS[key] = model
            break

# Load events
EVENTS = {}
for path in EVENT_DIR.glob('*.pickle'):
    EVENTS[path.stem] = load.unpickle(path)

# Silence cut from MALD words for EARSHOT
WORD_DELAY = {
    'GEESE': 0.007709750566893424,
    'GRAPE': 0.021496598639455782,
    'REAP': 0.0025396825396825397,
    'TAINT': 0.012698412698412698,
}
# Homophones that were replaced in model training

MERGE = {'DEVISE': 'DEVICE', 'VEIN': 'VAIN'}
MERGE_PATTERN = {src: load.unpickle(GAMMATONE_DIR / f'{dst}_MALD.pickle').T for src, dst in MERGE.items()}

print("** Generating predictors **")
PREDICTOR_DIR.mkdir(exist_ok=True)
ACTIVATION_DIR.mkdir(exist_ok=True)
for key, trainer in MODELS.items():
    print(f'{key}: {trainer.name}')
    for subject, events in EVENTS.items():
        raw_dst = ACTIVATION_DIR / f'{subject} Burgundy~{key}.pickle'
        existing_paths = list(PREDICTOR_DIR.glob(f'{subject} Burgundy~{key}-hu*.pickle'))
        if existing_paths and raw_dst.exists():
            print(f"{subject} exists, skipping.")
            continue
        print(f"{subject} generating...")
        events_t0 = events[0, 'T']

        # find words
        words = []
        for t, word in events.zip('T', 'item'):
            word = word.upper()
            delay = WORD_DELAY.get(word, 0)
            i0 = int(round((t + delay - events_t0 + PAD_START) * SAMPLINGRATE))
            if word in MERGE:
                pattern = MERGE_PATTERN[word]
                word = MERGE[word]
            else:
                pattern = None
            word_obj = Word(i0, 'MALD', word, trainer.lexicon, pattern)
            words.append(word_obj)

        # generate EARSHOT output
        stimuli = gen_input_from_words(iter(words), trainer.lexicon, -1, target_embedded_region=bool(trainer.loss_weights_to))
        data = trainer.predict(stimuli)
        uts = UTS(-PAD_START, 1/SAMPLINGRATE, len(data['inputs']))

        # Envelope
        # --------
        dst = PREDICTOR_DIR / f'{subject} Burgundy~Earshot-gammatone-1.pickle'
        if not dst.exists():
            x = NDVar(data['inputs'].sum(1), uts, 'gammatone-1')
            save.pickle(x, dst)

        # Hidden unit activity
        # --------------------
        raw = {}
        for tag, hidden in data.items():
            if not tag.startswith('hidden'):
                continue
            tag = tag[6:]
            # Hidden unit absolute activation
            name = f'hu{tag}-abs-sum'
            dst = PREDICTOR_DIR / f'{subject} Burgundy~{key}-{name}.pickle'
            if not dst.exists():
                hidden_magnitude = NDVar(abs(hidden).sum(1), uts, name)
                save.pickle(hidden_magnitude, dst)
            # Hidden unit absolute onset
            name = f'hu{tag}-abs-onset'
            dst = PREDICTOR_DIR / f'{subject} Burgundy~{key}-{name}.pickle'
            if not dst.exists():
                y = numpy.diff(abs(hidden), axis=0, prepend=0)
                y = numpy.clip(y, 0, None)
                hidden_onset = NDVar(y.sum(1), uts, name)
                save.pickle(hidden_onset, dst)

        # Raw activations
        # ---------------
        if not raw_dst.exists():
            for key_ in ['inputs', 'outputs', 'targets']:
                del data[key_]
            data['loss'] = numpy.array(data['loss'])
            save.pickle(data, raw_dst)

    print('\n')
