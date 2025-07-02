# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from typing import Literal, Union


def model(
        hidden: Union[int, str],
        layer: int = None,
        nodes: Literal['', '-local', '-out'] = None,
        transform: Literal['onset', 'sum', 'rel_entr'] = None,
        loss: str = None,
        target_space: Literal['OneHot', 'Glove-50c', 'Glove-300c', 'Sparse-10of300', 'Sparse-10of900'] = 'OneHot',
        lexicon: str = 'MALD-1000-train',
        seed: int = 0,
        k: int = None,  # K-Means
        k_drop_layer: int = None,  # Baseline model for testing this layer
):
    "Name of one or multiple MEG predictors"
    if transform == 'rel_entr':
        assert layer is None
        assert nodes is None
        predictor_desc = transform
    else:
        if layer is None and k is None:
            n_layers = hidden.count('x') + 1 - hidden.count('xd')
            return ' + '.join([model(hidden, layer_, nodes, transform, loss, target_space, lexicon, seed) for layer_ in range(n_layers)])
        elif nodes is None:
            has_subset = k is None and ('-' in hidden.split('x')[layer])
            all_nodes = ['-local', '-out'] if has_subset else ['']
            return ' + '.join([model(hidden, layer, nodes_, transform, loss, target_space, lexicon, seed, k, k_drop_layer) for nodes_ in all_nodes])
        elif transform is None:
            return ' + '.join([model(hidden, layer, nodes, transform_, loss, target_space, lexicon, seed, k, k_drop_layer) for transform_ in ['onset', 'sum']])
        if k_drop_layer is not None:
            n_layers = hidden.count('x') + 1 - hidden.count('xd')
            assert n_layers > 1
            layers = sorted(set(range(n_layers)).difference({k_drop_layer}))
            return ' + '.join([model(hidden, layer_, nodes, transform, loss, target_space, lexicon, seed, k) for layer_ in layers])
        elif k is not None:
            items = [f"{k}means"]
            if layer is not None:
                items.append(f'l{layer}')
            items.append(transform)
            predictor_desc = '-'.join(items)
        else:
            layer_desc = layer if 'x' in hidden else ''
            predictor_desc = f"hu{layer_desc}{nodes}-abs-{transform}"
    trainer_desc = trainer(hidden, loss, target_space, lexicon, seed)
    return f"{trainer_desc}-{predictor_desc}"


def trainer(
        hidden: Union[int, str],
        loss: str = None,
        target_space: Literal['OneHot', 'Glove-50c', 'Sparse-10of300', 'Sparse-10of900'] = 'OneHot',
        lexicon: str = 'MALD-1000-train',
        seed: int = 0,
):
    items = ['Earshot', 'LSTM', hidden, target_space]
    if loss:
        items.append(loss)
    if lexicon.startswith('MALD-1000-'):
        items.append(f"M1K-{lexicon[10:]}")
    else:
        raise ValueError(f"{lexicon=}")
    if seed:
        items.append(f's{seed}')
    return '-'.join(items)
