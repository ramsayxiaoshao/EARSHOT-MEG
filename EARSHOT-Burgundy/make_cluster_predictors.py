# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""
Creates predictors for K-means decomposition of model activity.
Uses model activation saved by `make predictors.py`.
"""
import argparse
from collections import Counter
from pathlib import Path

from eelbrain import UTS, Categorial, Dataset, NDVar, load, save
import numpy
from trftools.sklearn import KMeans

from constants2 import trainer as trainer_key


PREDICTOR_DIR = Path('~/Data/Burgundy/predictors').expanduser()
ACTIVATION_DIR = Path('~/Data/Burgundy/model-activation').expanduser()
EVENT_DIR = Path('~/Data/Burgundy/events').expanduser()
SUBJECTS = [path.stem for path in EVENT_DIR.glob('*.pickle')]

KS = [2, 3, 4, 8, 16, 32, 64]

# from make_predictors
PAD_START = 1
SAMPLINGRATE = 100


def load_hidden(subject, key) -> numpy.ndarray:  # shape = (time, unit)
    data = load.unpickle(ACTIVATION_DIR / f'{subject} Burgundy~{key}.pickle')
    if 'hidden' in data:
        return data['hidden']
    keys = [key_ for key_ in sorted(data) if key_.startswith('hidden')]
    return numpy.concatenate([data[key_] for key_ in keys], 1)


def main(k, key, hidden_desc, by_layer=False):
    BY_LAYER_INFO_PATH = ACTIVATION_DIR / f"{k}means {key} by-layer.txt"
    if BY_LAYER_INFO_PATH.exists():
        print(f"Skipping (already exists: {BY_LAYER_INFO_PATH.name})")
        return

    MODEL_PATH = ACTIVATION_DIR / f"{k}means {key}.pickle"
    if MODEL_PATH.exists():
        labels = load.unpickle(MODEL_PATH)
    else:
        data = numpy.concatenate([load_hidden(subject, key) for subject in SUBJECTS], 0)
        hidden = NDVar(data, (UTS(-1, 1/100, len(data)), Categorial('unit', map(str, range(data.shape[1])))))
        hidden_abs = hidden.abs()
        hidden_abs /= hidden_abs.std('time')
        model = KMeans(k, n_init='auto', random_state=0)
        labels = model.fit_predict(hidden_abs, 'unit')
        save.pickle(labels, MODEL_PATH)

    # find split by layer
    if by_layer:
        n_layers = hidden_desc.count('x') + 1
        n_units = {int(desc) for desc in hidden_desc.split('x')}
        assert len(n_units) == 1
        n_units = n_units.pop()
        layer = numpy.repeat(numpy.arange(n_layers), n_units)
        items = []
        for label in range(k):
            k_layers = layer[labels.x == label]
            layer_count = Counter(k_layers)
            main_layer = max(layer_count.items(), key=lambda item: item[1])[0]
            items.append([label, *(layer_count[i] for i in range(n_layers)), main_layer])
        by_layer_info = Dataset.from_caselist(['k', *(f'l{i}' for i in range(n_layers)), 'main'], items)
        by_layer_info.save_txt(BY_LAYER_INFO_PATH)
    else:
        n_layers = by_layer_info = None

    for subject in SUBJECTS:
        dst = PREDICTOR_DIR / f'{subject} Burgundy~{key}-{k}means-sum.pickle'
        dst_o = PREDICTOR_DIR / f'{subject} Burgundy~{key}-{k}means-onset.pickle'
        if dst.exists() and dst_o.exists():
            if not by_layer:
                continue
            means = load.unpickle(dst)
            onsets = load.unpickle(dst_o)
        else:
            data = abs(load_hidden(subject, key))
            means = []
            onsets = []
            for label in range(k):
                label_data = data[:, labels.x==label]
                means.append(label_data.mean(1))
                onset = numpy.diff(label_data, axis=0, prepend=0)
                onset = numpy.clip(onset, 0, None)
                onsets.append(onset.mean(1))
            uts = UTS(-PAD_START, 1 / SAMPLINGRATE, len(data))
            cluster = Categorial('cluster', [str(i) for i in range(k)])
            means = NDVar(numpy.array(means), (cluster, uts))
            onsets = NDVar(numpy.array(onsets), (cluster, uts))
            save.pickle(means, dst)
            save.pickle(onsets, dst_o)

        if not by_layer:
            continue

        for i in range(n_layers):
            index = by_layer_info['main'] == i
            dst = PREDICTOR_DIR / f'{subject} Burgundy~{key}-{k}means-l{i}-sum.pickle'
            dst_o = PREDICTOR_DIR / f'{subject} Burgundy~{key}-{k}means-l{i}-onset.pickle'
            save.pickle(means[index], dst)
            save.pickle(onsets[index], dst_o)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('hidden', type=str)
    parser.add_argument('--k', type=int, choices=[-1, *KS], default=32)
    parser.add_argument('--loss', type=str, default=None)
    parser.add_argument('--target', default='OneHot', choices=['OneHot', 'Glove-50c', 'Glove-300c'])
    parser.add_argument('--by-layer', default=False, action='store_true')
    args = parser.parse_args()

    key = trainer_key(args.hidden, target_space=args.target, lexicon='MALD-1000-train', seed=0, loss=args.loss)
    if args.k == -1:
        ks = KS
    else:
        ks = [args.k]

    for k in ks:
        print(f"{key} - {k} means")
        main(k, key, args.hidden, by_layer=args.by_layer)
