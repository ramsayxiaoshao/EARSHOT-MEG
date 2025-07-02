# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from itertools import product
import fnmatch
from pathlib import Path
from typing import Tuple

from earshot.earshot_lexicon import DATA_DIR
from earshot.train_earshot import Model
from earshot.report import Example
from eelbrain import testnd, Dataset, combine
import joblib

from constants2 import model


GRID_DIR = Path('/Volumes/Seagate BarracudaFastSSD/Corpus/MALD/fixed-words')
MEMORY = joblib.Memory(DATA_DIR / 'joblib')


class NoTrainedModel(Exception):
    pass


def add_variables(ds):
    hidden_cells = ds['hidden'].cells
    hidden_labels = {hidden: hidden.count('x') + 1 for hidden in hidden_cells}
    ds['layers'] = ds['hidden'].as_var(hidden_labels)
    if 'loss' in ds:
        ds['loss'].update_labels({'': 'constant'})


def load_performance(
        lexicon='MALD-1000-train',
        stimuli: str = 'NoSil inf',
        by_speaker: bool = False,
        **parameters,
):
    parameters['lexicon_name'] = lexicon
    variables = {k: v if isinstance(v, (list, tuple)) else [v] for k, v in parameters.items()}
    if 'seed' in variables:
        seeds = variables.pop('seed')
    else:
        seeds = None
    rows = []
    keys = tuple(variables)
    for values in product(*variables.values()):
        values = ['' if v is None else v for v in values]  #
        if seeds is None:
            rows.append(load_best(keys, values, by_speaker, stimuli=stimuli))
        else:
            for seed in seeds:
                rows.append(load_best(keys, values, by_speaker, seed, stimuli))
    ds = combine(rows)
    ds['speaker'].random = True
    add_variables(ds)
    return ds


def load_best(
        keys,
        values,
        by_speaker: bool = False,
        seed: int = 0,
        stimuli: str = 'NoSil inf',
):
    return _load(keys, values, by_speaker, seed, stimuli)


@MEMORY.cache
def _load(
        keys,
        values,
        by_speaker: bool = False,
        seed: int = 0,
        stimuli: str = 'NoSil inf',
):
    args = {k: v for k, v in zip(keys, values)}
    model = Model(seed=seed, **args)
    if model.checkpoints:
        checkpoint = model.checkpoints[-1]
    else:
        checkpoint = -1
    # Results
    example = Example(model)
    results, ds_correct = example.load_results(stimuli)
    layers = values[keys.index('hidden')].count('x') + 1
    # word error rate
    del ds_correct['overall']
    x = 'speaker % trained' if by_speaker else 'trained'
    ds = ds_correct.aggregate(x, drop_bad=True)
    ds['wer'] = 1 - ds['last10']
    ds[:, 'layers'] = layers
    ds[:, 'seed'] = seed
    ds[:, 'checkpoint'] = checkpoint
    for key, value in zip(keys, values):
        ds[:, key] = value
    return ds


@MEMORY.cache
def load_roi_data(
        model: str = f"gt-log8 + phone-p0",
        roi: str = 'STG301',
        norm: bool = True,
):
    ""
    import trftools.roi
    from burgundy import e
    from jobs import STG, WHOLEBRAIN

    hemis = ('lh', 'rh')
    src = e.load_src(ndvar=True, mrisubject='fsaverage')
    rois = trftools.roi.mask_roi(roi, src)
    rois = [hemi_roi.sub(hemi_roi == 1) for hemi_roi in rois]

    if norm is True:
        norms = load_roi_norm(roi)
    elif norm is False:
        norms = (None, None)
    else:
        raise TypeError(f"{norm=}")

    e.reset()
    parameters = STG if roi.startswith('STG') else WHOLEBRAIN
    ds_raw = e.load_trfs(-1, model, **parameters, trfs=False)
    dss = []
    for hemi, hemi_roi, hemi_norm in zip(hemis, rois, norms):
        hemi_ds = ds_raw['subject',]
        hemi_ds['det_roi'] = ds_raw['det'].mean(hemi_roi)
        hemi_ds[:, 'hemi'] = hemi
        if hemi_norm:
            hemi_ds['det_roi'] /= hemi_norm
        dss.append(hemi_ds)
    return combine(dss)


def load_model_roi_data(
        roi: str = 'STG301',
        norm: bool = True,
        base: str = f"gt-log8 + phone-p0",
        hemi: bool = False,
        **parameters,
):
    variables = {k: v if isinstance(v, (list, tuple)) else [v] for k, v in parameters.items()}

    dss = []
    keys = tuple(variables)
    for values in product(*variables.values()):
        # values = ['' if v is None else v for v in values]  #
        args = {k: v for k, v in zip(keys, values)}
        ds = load_roi_data(f"{base} + {model(**args)}", roi, norm)
        for key, value in args.items():
            if value is None:
                if key in ('k',):
                    value = 1
                else:
                    value = ''
            elif key == 'nodes' and value == '':
                value = 'all'
            ds[:, key] = value
        if not hemi:
            ds = ds.aggregate('subject', drop='hemi')
            for key in ['k']:
                if key in ds:
                    ds[key] = ds[key].astype(int)
        dss.append(ds)
    ds = combine(dss)
    add_variables(ds)
    return ds


@MEMORY.cache
def load_roi_norm(
        roi: str = None,
):
    "Use acoustic model as 100%"
    ds0 = load_roi_data('gt-log8 + phone-p0', roi, False)
    ds = ds0.aggregate('hemi', drop_bad=True)
    dets = {hemi: det for det, hemi in ds.zip('det_roi', 'hemi')}
    lh, rh = dets['lh'], dets['rh']
    return lh, rh


@MEMORY.cache
def load_variance_components(x: str, parameters: dict, components: Tuple[str, ...] = None):
    from burgundy import e

    if components is None:
        components = ['gammatone*']
        for layer in range(9):
            if f'-hu{layer}-local-' in x:
                components.append(f'*_hu{layer}_local_*')
                components.append(f'*_hu{layer}_out_*')
            elif f'-hu{layer}-' in x:
                components.append(f'*_hu{layer}_*')
            else:
                break
    print(components)

    e.reset()
    x_log = {}
    rows = []
    for subject in e:
        x_log[subject] = {}
        trf = e.load_trf(x, **parameters, partition_results=True)
        job = e._trf_job(x, **parameters)
        y, x_ = job.args[:2]

        y_pred = trf.cross_predict(x_, scale='normalized')
        y_normalized = (y - trf.y_mean) / trf.y_scale
        y_residual = y_normalized - y_pred
        ss_y = (y_normalized ** 2).sum('time')
        proportion_explained = 1 - ((y_residual ** 2).sum('time') / ss_y)
        rows.append([subject, 'full', proportion_explained])

        for expression in components:
            x_component = [xi for xi in x_ if not fnmatch.fnmatch(xi.name, expression)]
            key = expression.strip('_*')
            x_log[subject][key] = [xi.name for xi in x_component]

            y_pred = trf.cross_predict(x_component, scale='normalized')
            y_residual = y_normalized - y_pred
            proportion_explained = 1 - ((y_residual ** 2).sum('time') / ss_y)
            rows.append([subject, key, proportion_explained])

    info = {
        'xs': x_log,
        'components': [c.strip('_*') for c in components],
    }
    return Dataset.from_caselist(['subject', 'x', 'det'], rows, info=info)


@MEMORY.cache
def load_variance_component_tests(x: str, components: Tuple[str, ...] = None):
    data = load_variance_components(x, components)
    data['det'] = data['det'].smooth('source', 0.005, 'gaussian')
    ress = {}
    for component in data.info['components']:
        res = testnd.TTestRelated('det', 'x', 'full', component, 'subject', data=data, samples=1000, tfce=True)
        ress[component] = res
    return ress
