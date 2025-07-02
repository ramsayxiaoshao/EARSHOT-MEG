from __future__ import annotations

import argparse
from collections import defaultdict
import dataclasses
import datetime
from functools import cached_property
from itertools import chain, combinations, product
from pathlib import Path
import pickle
import queue
from random import Random
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
from typing import Dict, Generator, Iterator, List, Literal, Optional, Sequence, Tuple, Union

import numpy
import scipy.spatial.distance
import scipy.special
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import set_random_seed

from . import earshot_lexicon
from . import _op
from . import combinatorics
from ._loss import CohortWeightedLoss, DownWeightCompetitors
from .earshot_lexicon import DATA_DIR, VOWELS, PHONES
from .glove import read_glove
from .mykeras import Backtrack, Subset


STIMULUS_SEQUENCES = {
    # 'Sil-50ms': (1, 0, 50, 100),
    # '50Sil-50ms': (2, 0, 50, 100),
    'NoSil': (0, 0, 0, 0),
    'Sil': (1, 0, 20, 40),
    # '50Sil': (2, 0, 20, 50),
    '2': (2, 2, 20, 50),
    # '23': (2, 3, 20, 50),
    # longer phrases
    # '30': (3, 0, 20, 50),
    # '31': (3, 1, 20, 50),
    # '32': (3, 2, 20, 50),
    # '3': (3, 3, 20, 50),
    # even longer
    # '52': (5, 2, 20, 50),
}
# Function defaults, see gen_words() below
N_WORDS, N_PHRASES, MIN_SILENCE, MAX_SILENCE = STIMULUS_SEQUENCES['2']
SNR = numpy.inf


def format_e(x):
    s = f'{x:.0e}'
    assert float(s) == x
    return s.replace('e-0', 'e-')


def diff(list_from, list_subtract):
    return [w for w in list_from if w not in list_subtract]


def softmax(x, axis, cool=None, inplace=False):
    if cool:
        if inplace:
            x *= cool
        else:
            x = x * cool
    return scipy.special.softmax(x, axis)


def get_git_revision_hash() -> str:
    # https://stackoverflow.com/a/21901260
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()


@dataclasses.dataclass
class Target:
    name: str
    items: List[str] = dataclasses.field(repr=False)
    coding: str = 'OneHot'  # == Lexicon.target_space
    embedding: numpy.ndarray = None    # item x embedding dimension
    mapping: Dict[str, str] = None  # secondary mapping to target items
    n: int = dataclasses.field(init=False)  # number of items
    ndim: int = dataclasses.field(init=False)  # number of embedding dimensions

    def __post_init__(self):
        self.n = len(self.items)
        # embedding
        if self.embedding is None:
            if self.coding == 'OneHot':
                self.embedding = numpy.eye(self.n)
            else:
                raise ValueError(f'{self.coding} embedding unspecified')
        self.ndim = self.embedding.shape[1]
        # for binary embeddings, use min for softmax
        self._is_binary_vector = self.coding.startswith(('Sparse', 'Balanced'))

    @cached_property
    def index(self) -> Dict[str, int]:
        return {item: index for index, item in enumerate(self.items)}

    @cached_property
    def map_index(self) -> Dict[str, int]:
        return {source: self.index[target] for source, target in self.mapping.items()}

    @cached_property
    def _embedding_with_silence(self) -> numpy.ndarray:
        return numpy.concatenate([self.embedding, numpy.zeros((1, self.ndim))], axis=0)

    @cached_property
    def _binary_embedding_indices(self):
        if not self._is_binary_vector:
            raise RuntimeError(f"Invalid for non-binary embedding {self!r}")
        return numpy.array([numpy.flatnonzero(pattern) for pattern in self.embedding])

    @cached_property
    def _max_dist(self):
        return numpy.sqrt(numpy.max(numpy.sum(self.embedding**2, 1)))

    def activation(
            self,
            output: numpy.ndarray,  # time x semantics
            method: str = None,
    ) -> numpy.ndarray:  # time, word
        if output.ndim == 3 and len(output) == 1:
            output = output[0]

        if self.coding == 'OneHot':
            if method:
                raise TypeError(f'{method=} with OneHot')
            return output

        if method is None:
            if self._is_binary_vector:
                method = 'min'
            else:  # Glove
                method = 'dist'

        if method == 'min':
            return _op.binary_vector_activation_min(output, self._binary_embedding_indices)
        elif method == 'mean':
            order = self._binary_embedding_indices[0].sum()
            return _op.binary_vector_activation_mean(output, self._binary_embedding_indices, order)
        elif method == 'dist':
            dist = scipy.spatial.distance.cdist(output, self.embedding)
            return dist
        else:
            raise ValueError(f"{method=}")

    def softmax(
            self,
            output: numpy.ndarray,  # time x semantics
            silence: bool = True,
            cool: float = None,
    ) -> numpy.ndarray:  # time, word
        """Transform model output to probability distribution over words, including silence.

        Probability distribution != confidence.
        Just by the nature of the semantic space, unpreferred alternatives will
        always occupy a range of distances.
        """
        if output.ndim == 3 and len(output) == 1:
            output = output[0]
        if self.coding == 'OneHot':
            if not silence:
                return softmax(output, 1, cool)
            # silence should be seen in that there is no word
            max_output = output.max(1, keepdims=True)
            silence = numpy.subtract(1, max_output, out=max_output)
            activation = numpy.concatenate((output, silence), 1)
            return softmax(activation, 1, cool, True)
        elif self._is_binary_vector:
            if silence:
                raise NotImplementedError(f'{silence=} for binary vectors')
            activation = self.activation(output)
            return softmax(activation, 1, cool, True)
        # Glove: distance based
        if silence:
            embedding = self._embedding_with_silence
        else:
            embedding = self.embedding
        dist = scipy.spatial.distance.cdist(output, embedding)
        # For softmax, bigger is better
        dist *= -1
        return softmax(dist, 1, cool, True)


@dataclasses.dataclass()
class Lexicon:
    """
    MALD lexicon:

    - `MALD-1000`: frequency of 1000 as cutoff for inclusion in the lexicon (only option currently)
      - `ø`: even train/test split in all speakers
      - `-nomald`: exclude MALD speaker
        - `-{n}`: only use n words
      - `-test`: hold-out MALD speaker for test
      - `-train`: all Burgundy items from the MALD speaker in the training set
    """
    # Lexicon
    name: str = '2056'
    # Targets
    target_space: str = 'OneHot'  # 'OneHot', 'Glove-n', 'Sparse-n', 'Balanced-n'
    target_shape: str = 'box'  # 'box', 'ramp', 'offset-n'
    # inputs
    n_bands: int = 50

    # secondary
    data_dir: Path = dataclasses.field(init=False)
    base_name: Literal['earshot', 'burgundy', 'MALD-NEIGHBORS-1000'] = dataclasses.field(init=False)
    n_words: int = dataclasses.field(init=False)
    speakers: List[str] = dataclasses.field(init=False)
    _mald_cutoff: str = dataclasses.field(init=False, default=None)
    _split: str = dataclasses.field(init=False, default=None)
    target_length: int = dataclasses.field(init=False, default=0)
    _target_space_kind: Literal['OneHot', 'Glove', 'Sparse'] = dataclasses.field(init=False)
    _target_space_n: int = dataclasses.field(init=False, default=None)
    _target_n: int = dataclasses.field(init=False, default=None)  # only for Sparse
    _center_embedding: bool = dataclasses.field(init=False)
    target_shape_type: Literal['box', 'ramp', 'offset'] = dataclasses.field(init=False)
    # input_variance: numpy.ndarray = dataclasses.field(init=False)

    def __post_init__(self):
        mald_tag = None
        if self.name.isnumeric():
            self.base_name = 'earshot'
            self.data_dir = DATA_DIR
            self.n_words = int(self.name)
            self._split = 'even'
        elif match := re.match(r'MALD-(1000)(?:-(test|train|nomald(?:-\d+)?))?$', self.name):
            self._mald_cutoff, mald_tag = match.groups()
            self.n_words = 2934
            if mald_tag is None:
                self._split = 'even'
            elif mald_tag in ('test', 'train'):
                self._split = mald_tag
            elif match := re.match(r'(nomald)(?:-(\d+))?', mald_tag):
                self._split = 'even'
                _, n = match.groups()
                if n:
                    n = int(n)
                    assert n < self.n_words
                    self.n_words = n
            else:
                raise ValueError(f'{self.name=}')
            assert self._mald_cutoff == '1000'
            self.base_name = 'MALD-NEIGHBORS-1000'
            self.data_dir = DATA_DIR / 'Burgundy'
        elif match := re.match('burgundy-(test|train)', self.name):
            self.n_words = 1000
            self._split = match.group(1)
            self.base_name = 'burgundy'
            self.data_dir = DATA_DIR / 'Burgundy'
        else:
            raise ValueError(f'{self.name=}')

        # Speakers
        path = self.data_dir / 'Speakers.txt'
        self.speakers = path.read_text().splitlines()
        if mald_tag == 'nomald':
            self.speakers = [s for s in self.speakers if s != 'MALD']

        # Target space
        if (match := re.match(r"^(OneHot|Glove|Sparse|Balanced)(?:-(\S+))?$", self.target_space)) is None:
            raise ValueError(f"{self.target_space=}")
        self._target_space_kind, options = match.groups()
        target_space_dims = center = None
        if self._target_space_kind == 'OneHot':
            if options:
                raise ValueError(f"{self.target_space=}")
        elif self._target_space_kind == 'Glove':
            if (match := re.match(r"^(\d+)(c)?$", options)) is None:
                raise ValueError(f"{self.target_space=}")
            target_space_dims, center = match.groups()
        elif self._target_space_kind == 'Sparse':
            if (match := re.match(r"^(\d+)(?:of(\d+))?$", options)) is None:
                raise ValueError(f"{self.target_space=}")
            n_sparse, target_space_dims = match.groups()
            self._target_n = int(n_sparse)
        elif self._target_space_kind == 'Balanced':
            if (match := re.match(r"^(\d+)$", options)) is None:
                raise ValueError(f"{self.target_space=}")
            n_sparse, = match.groups()
            self._target_n = int(n_sparse)
        else:
            raise NotImplementedError(f"{self.target_space=}")
        if target_space_dims:
            self._target_space_n = int(target_space_dims)
        self._center_embedding = bool(center)
        # Target location
        if (match := re.match(r"(box|ramp|offset)(?:-(\d+))?", self.target_shape)) is None:
            raise ValueError(f"{self.target_shape=}")
        self.target_shape_type, target_length = match.groups()
        if target_length:
            assert self.target_shape_type != 'box'
            self.target_length = int(target_length)
        # Split train/test
        self.train_items, self.test_items = self._split_items()

    @cached_property
    def input_variance(self) -> numpy.ndarray:
        # calculate training item variance for noise
        gt_var = numpy.zeros(self.n_bands)
        for key in self.train_items:
            gt_var += self.inputs[key].var(0)
        gt_var /= len(self.train_items)
        return gt_var

    @cached_property
    def pronunciations(self) -> Dict[str, Tuple[str, ...]]:  # {word: phones}
        pronunciations = earshot_lexicon.read_pronunciations(self.base_name)
        return {word: tuple(pronunciations[word].split()) for word in self.words}

    @cached_property
    def words(self) -> List[str]:
        if self.name.isnumeric():
            return earshot_lexicon.read_words()[:self.n_words]
        words = earshot_lexicon.read_words(self.base_name)
        if self.n_words < len(words):
            assert not self.base_name.startswith('MALD')
            random = Random(0)
            words = random.sample(words, self.n_words)
        if self.n_words > len(words):
            raise RuntimeError(f"{self.n_words=}, {len(words)=}")
        return words

    @cached_property
    def word_index(self) -> Dict[str, int]:
        return {word: i for i, word in enumerate(self.words)}

    @cached_property
    def phone_sort(self):
        """Sort words such that they are ordered by phones

         - Cohort neighbors are adjacent
         - Phones are separated by vowel and consonant
        """
        pronunciations = earshot_lexicon.read_pronunciations(self.base_name, split_phones=True)

        def key(
                index: int,  # lexicon word index
        ) -> List[int]:  # --> List of PHONE indices
            phones = pronunciations[self.words[index]]
            phone_is = [PHONES.index(phone) for phone in phones]
            return phone_is

        return sorted(range(self.n_words), key=key)

    @cached_property
    def word_target(self) -> Target:
        # Generate targets
        if self._target_space_kind == 'Glove':
            glove = read_glove(self._target_space_n)
            embedding = numpy.array([glove[w] for w in self.words])
        elif self._target_space_kind == 'Sparse' and self._target_space_n:
            # Generate patterns as indices {1, 2, 4} -> [0, 1, 1, 0, 1, 0, ...]
            assert self._target_n > 1
            rng = Random(42)
            patterns = []
            # All patterns have to differ in at least n_unique indices
            n_unique = 2
            n_iter_this = 0
            while len(patterns) < self.n_words:
                pattern = set(rng.sample(range(self._target_space_n), self._target_n))
                if any(len(pattern.difference(p)) <= n_unique for p in patterns):
                    n_iter_this += 1
                    if n_iter_this > 100000:
                        raise RuntimeError("Number allowed iterations exceeded while attempting to satisfy uniqueness constraint for sparse output space")
                    continue
                patterns.append(pattern)
                n_iter_this = 0
            # Generate embedding
            embedding = numpy.zeros((self.n_words, self._target_space_n))
            for i, pattern in enumerate(patterns):
                embedding[i, list(pattern)] = 1
        elif self._target_space_kind == 'Sparse':
            assert self._target_n > 1
            n_targets = self._target_n + 2
            while scipy.special.comb(n_targets, self._target_n, exact=True) < self.n_words:
                n_targets += 1
            embedding = numpy.zeros((self.n_words, n_targets))
            for i, pattern in zip(range(self.n_words), combinations(range(n_targets), self._target_n)):
                for j in pattern:
                    embedding[i, j] = 1
        elif self._target_space_kind == 'Balanced':
            # each word should activate 50% of the targets
            n_dims = self._target_n * 2
            n_combinations = combinatorics.n_possibilities(0, n_dims, self._target_n)
            if self.n_words > n_combinations:
                raise ValueError(f"{self.target_space=}: not enough patterns for {self.n_words} words")
            random = Random(42)
            indexes = random.sample(range(n_combinations), self.n_words)
            embedding = numpy.zeros((self.n_words, n_dims))
            for i, index in enumerate(indexes):
                for j in combinatorics.combination(n_dims, self._target_n, index):
                    embedding[i, j] = 1
        elif self._target_space_kind == 'OneHot':
            assert self._target_n is None
            embedding = None
        else:
            raise ValueError(f"{self.target_space=}")
        if self._center_embedding:
            embedding -= embedding.mean(0)
        return Target('word', self.words, self.target_space, embedding)

    @cached_property
    def speaker_target(self) -> Target:
        return Target('speaker', self.speakers)

    @cached_property
    def inputs(self) -> Dict[Tuple[str, str], numpy.ndarray]:  # (time, frequency)
        gammatone_root = self.data_dir / f'GAMMATONE_{self.n_bands}_100'
        gts = {}
        for speaker, word in product(self.speakers, self.words):
            dirname = speaker if speaker == 'MALD' else speaker.title()
            path = gammatone_root / dirname / f'{word}_{speaker}.pickle'
            with path.open('rb') as file:
                gts[speaker, word] = pickle.load(file).T
        return gts

    def phone_target(
            self,
            kind: Literal['ph', 'v'],
            position: int,
    ):
        if kind == 'ph':
            target_items = PHONES
            if any(position >= len(phones) for phones in self.pronunciations.values()):
                target_items = (*target_items, '_')
            mapping = {word: phones[position] if position < len(phones) else '_' for word, phones in self.pronunciations.items()}
        elif kind == 'v':
            assert position == 0
            target_items = VOWELS
            mapping = {}
            for word, phones in self.pronunciations.items():
                for phone in phones:
                    if phone in VOWELS:
                        mapping[word] = phone
                        break
                else:
                    raise RuntimeError(f'{word}: no vowel in {phones}')
        else:
            raise ValueError(f'{kind=}')
        return Target(f'{kind}{position}', target_items, mapping=mapping)

    def _split_items(self) -> (List[Tuple[str, str]], List[Tuple[str, str]]):
        "For cross-validation"
        if self.base_name == 'burgundy':
            assert self.speakers[-1] == 'MALD'
            mald_items = [('MALD', word) for word in self.words]
            if self._split == 'train':
                train_items, test_items = self._even_split(self.speakers[:-1], self.words)
                train_items.extend(mald_items)
            elif self._split == 'test':
                train_items = list(product(self.speakers[:-1], self.words))
                test_items = mald_items
            else:
                raise RuntimeError(f"{self._split=}")
        elif self._split == 'test':  # leave out Burgundy MALD items for testing set
            assert 'MALD' in self.speakers
            other_speakers = [speaker for speaker in self.speakers if speaker != 'MALD']
            all_words = set(self.words)
            burgundy_items = earshot_lexicon.read_words('burgundy')
            burgundy_items = sorted(all_words.intersection(burgundy_items))
            non_burgundy_items = sorted(all_words.difference(burgundy_items))
            train_items, test_items = self._even_split(other_speakers, non_burgundy_items)
            test_items.extend([('MALD', word) for word in burgundy_items])
            train_items.extend([('MALD', word) for word in non_burgundy_items])
            train_items.extend(product(other_speakers, burgundy_items))
        elif self._split == 'train':  # train on all Burgundy MALD items
            assert self.speakers[-1] == 'MALD'
            steps = numpy.diff(numpy.linspace(0, self.n_words, len(self.speakers) + 1).round()).astype(int)
            random = Random(0)
            all_words = list(self.words)
            burgundy_words = earshot_lexicon.read_words('burgundy')
            mald_test_words = random.sample(diff(all_words, burgundy_words), steps[0])
            words_not_in_test = diff(all_words, mald_test_words)
            test_items = [('MALD', word) for word in mald_test_words]
            train_items = [('MALD', word) for word in words_not_in_test]
            for speaker, step in zip(self.speakers[:-1], steps[:-1]):
                test_words = random.sample(words_not_in_test, step)
                words_not_in_test = diff(words_not_in_test, test_words)
                test_items.extend((speaker, word) for word in test_words)
                train_items.extend((speaker, word) for word in diff(all_words, test_words))
        elif self._split == 'even':
            assert 'MALD' not in self.speakers
            train_items, test_items = self._even_split(self.speakers, self.words)
        else:
            raise ValueError(f"{self.name=}: invalid data split {self._split!r}")
        return train_items, test_items

    @staticmethod
    def _even_split(
            speakers: Sequence[str],
            words: Sequence[str],
    ) -> (List[Tuple[str, str]], List[Tuple[str, str]]):
        # Randomize words
        rng = Random(0)
        words = list(words)
        rng.shuffle(words)
        # Split into equal sized bins
        train_items = []
        test_items = []
        step = len(words) / len(speakers)
        for i, speaker in enumerate(speakers):
            start, stop = int(i * step), int((i + 1) * step)
            train_items.extend([(speaker, word) for word in chain(words[:start], words[stop:])])
            test_items.extend([(speaker, word) for word in words[start:stop]])
        return train_items, test_items

    def softmax(
            self,
            output: numpy.ndarray,  # time x semantics
            **kwargs,
    ) -> numpy.ndarray:  # time, word
        return self.word_target.softmax(output, **kwargs)


@dataclasses.dataclass
class Stimulus:
    lexicon: Lexicon
    code: str
    snr: float
    items: str = dataclasses.field(init=False, repr=False)
    n_words: int = dataclasses.field(init=False, repr=False)
    n_phrases: int = dataclasses.field(init=False, repr=False)
    min_silence: int = dataclasses.field(init=False, repr=False)
    max_silence: int = dataclasses.field(init=False, repr=False)

    def __post_init__(self):
        self.n_words, self.n_phrases, self.min_silence, self.max_silence = STIMULUS_SEQUENCES[self.code]


@dataclasses.dataclass()
class Model:
    lexicon_name: str = '1000'
    # Stimuli
    stimulus_sequence: str = '2'
    snr: float = 20  # number or numpy.inf
    target_space: str = 'Glove-50c'
    target_shape: str = 'box'
    # Model settings
    n_bands: int = 50
    mechanism: Literal['LSTM', 'GRU', 'RNN'] = 'LSTM'
    hidden: str = '512'
    activation: str = None
    activation_x: str = None
    regularize: float = 0
    regularize_x: float = 0
    loss: str = None
    optimizer: str = 'RMSProp'  # https://www.dlology.com/blog/quick-notes-on-how-to-choose-optimizer-in-keras/
    learning_rate: float = 1e-3
    patience: int = 200
    segment_length: int = 1000
    batch_size: int = 10
    steps_per_epoch: int = 80
    seed: int = None
    stimulus_seed: int = 0

    has_subset: bool = dataclasses.field(init=False)
    _layer_spec: List[Tuple[int, int]] = dataclasses.field(init=False, default_factory=list)
    _activation_x_scale: int = dataclasses.field(init=False, default=0)
    _activation_x_shape: str = dataclasses.field(init=False, default=None)
    lexicon: Lexicon = dataclasses.field(init=False)
    stimulus: Stimulus = dataclasses.field(init=False)
    _loss_weight_kind: Literal[None, 'w', 'dw'] = dataclasses.field(init=False, default=None)
    _loss_weight_factor: float = dataclasses.field(init=False, default=None)
    loss_weights_to: Optional[int] = dataclasses.field(init=False, default=None)

    def __post_init__(self):
        # Lexicon
        self.lexicon = Lexicon(self.lexicon_name, self.target_space, self.target_shape, self.n_bands)
        # Training stimulus sequence
        self.stimulus = Stimulus(self.lexicon, self.stimulus_sequence, self.snr)
        # Hidden units
        for item in self.hidden.split('x'):
            if match := re.match(r'^(\d+)(?:-(\d+))?$', item):
                n_hidden, subset = match.groups()
                subset = int(subset) if subset else 0
                self._layer_spec.append((int(n_hidden), subset))
            else:
                raise ValueError(f'{self.hidden=}; invalid section: {item}')
        self.has_subset = any(subset for _, subset in self._layer_spec)
        # Loss
        if self.loss and (match := re.match(r'^(dw)(\d+)(?:(to|for)(\d+))?$', self.loss)):
            assert self.lexicon._target_space_kind == 'OneHot'
            self._loss_weight_kind = match.group(1)
            self._loss_weight_factor = int(match.group(2))
            preposition = match.group(3)
            if preposition is None:
                self.loss_weights_to = None
            elif preposition == 'to':
                self.loss_weights_to = -int(match.group(4))
            elif preposition == 'for':
                self.loss_weights_to = int(match.group(4))
            else:
                raise ValueError(f"{self.loss=}")
        # Model desc
        model_items = [self.mechanism, self.hidden]
        if self.activation:
            model_items.append(self.activation)
        if self.activation_x:
            if not self.has_subset:
                raise ValueError(f"{self.activation_x=} with {self.hidden=}")
            model_items.append(f'x{self.activation_x}')
            scale, self._activation_x_shape = re.match(r'^(\d*)(\D+)$', self.activation_x).groups()
            if scale:
                self._activation_x_scale = int(scale)
                if self._activation_x_scale == 1:
                    raise ValueError(f'{self.activation_x}: omit 1')
        if self.regularize:
            model_items.append(f'reg{format_e(self.regularize)}')
        if self.regularize_x:
            if not self.has_subset:
                raise ValueError(f"{self.regularize_x=} with {self.hidden=}")
            model_items.append(f'regx{format_e(self.regularize_x)}')
        if self.loss:
            model_items.append(self.loss)
        model_items.append(self.optimizer)
        model_items.append(format_e(self.learning_rate))
        if self.patience != 200:
            model_items.append(f'pat{self.patience}')
        model_items.append(str(self.batch_size))
        if self.steps_per_epoch != 80:
            model_items.append(str(self.steps_per_epoch))
        if self.seed is not None:
            model_items.append(f's{self.seed}')
        model_desc = '-'.join(model_items)
        # stimulus desc
        stimulus_items = [self.lexicon_name]
        if self.n_bands != 50:
            stimulus_items.append(f'{self.n_bands}')
        stimulus_items.append(self.stimulus_sequence)
        stimulus_items.append(str(self.snr))
        stimulus_desc = '-'.join(stimulus_items)
        # name for paths
        self.name = f"{model_desc} {self.target_space}-{self.target_shape} {stimulus_desc}"
        self.model_dir = DATA_DIR / 'Models' / self.name

    @classmethod
    def from_args(cls, args: argparse.Namespace):
        return cls(args.lexicon, args.stimulus, args.snr, args.target, args.target_shape, args.n_bands, args.mechanism, args.hidden, args.activation, args.activation_x, args.regularize, args.regularize_x, args.loss, args.optimizer, args.learning_rate, args.patience, batch_size=args.batch_size, steps_per_epoch=args.steps_per_epoch, seed=args.seed, stimulus_seed=args.stimulus_seed)

    def get_optimizer(self):
        # Never learned much: tf.optimizers.Adam(learning_rate=1e-2, beta_1=0.9, beta_2=0.999, epsilon=1e-07, clipvalue=0.5)
        if self.optimizer == 'RMSProp':
            return tf.optimizers.RMSprop(self.learning_rate)
        elif self.optimizer == 'Adam':
            return tf.optimizers.Adam(self.learning_rate)
        else:
            raise ValueError(f'{self.optimizer=}')

    def make_model(
            self,
            segment_length: int = None,
            batch_size: int = None,
            stateful: bool = True,
            output_hidden: bool = False,
    ) -> keras.models.Model:
        if segment_length is None:
            segment_length = self.segment_length
        if batch_size is None:
            batch_size = self.batch_size
        # loss
        if self.loss == 'abs':
            loss = keras.losses.MeanAbsoluteError()
        elif self.loss == 'ms':
            assert self.lexicon._target_space_kind != 'Glove'
            loss = keras.losses.MeanSquaredError()
        elif self._loss_weight_kind == 'dw':
            loss = DownWeightCompetitors(self._loss_weight_factor)
        elif self.lexicon._target_space_kind == 'Glove':
            loss = keras.losses.MeanSquaredError()
        elif self.lexicon._target_space_kind in ('OneHot', 'Sparse', 'Balanced', 'ELU'):
            loss = keras.losses.BinaryCrossentropy()
        else:
            raise RuntimeError(f"{self.lexicon.target_space=}")
        # Ouput layer activation
        if self.lexicon._target_space_kind == 'Glove':
            activation = None
        elif self.lexicon._target_space_kind in ('OneHot', 'Sparse', 'Balanced'):
            activation = 'sigmoid'
        else:
            raise RuntimeError(f"{self.lexicon.target_space=}")
        # Input
        batch_size_arg = batch_size if stateful else None
        input_layer = layer = layers.Input((segment_length, self.lexicon.n_bands), batch_size_arg)
        # RNN
        rnn_args = dict(return_sequences=True, stateful=stateful)
        if self.activation:
            rnn_args['activation'] = self.activation
        hidden_layers = []
        for n_hidden, subset in self._layer_spec:
            # new layer
            if self.mechanism == 'LSTM':
                layer = layers.LSTM(n_hidden, **rnn_args)(layer)
            elif self.mechanism == 'GRU':
                layer = layers.GRU(n_hidden, **rnn_args)(layer)
            elif self.mechanism == 'RNN':
                layer = layers.SimpleRNN(n_hidden, **rnn_args)(layer)
            else:
                raise ValueError(f'{self.mechanism=}')
            hidden_layers.append(layer)
            if self.regularize:
                layer = layers.ActivityRegularization(self.regularize)(layer)
            # Split hidden units
            if subset:
                layer = Subset(subset, scale=self._activation_x_scale)(layer)
                if self._activation_x_shape:
                    layer = layers.Activation(self._activation_x_shape)(layer)
                if self.regularize_x:
                    layer = layers.ActivityRegularization(self.regularize_x)(layer)
                hidden_layers.append(layer)
        # Embedding
        output = layers.Dense(self.lexicon.word_target.ndim, activation=activation)(layer)
        # Model
        if output_hidden:
            output = [*hidden_layers, output]
        model = keras.models.Model(input_layer, output)
        model.compile(loss=loss, optimizer=self.get_optimizer())
        return model

    @cached_property
    def checkpoints(self) -> List[int]:
        return sorted([int(path.stem) for path in self.model_dir.glob('*.hdf5')])

    def load_log(self, raw: bool = False):
        path = self.model_dir / 'log.csv'
        if raw:
            text = path.read_text()
            values = (line.split(',') for line in text.splitlines()[1:])
            return [(int(epoch), float(loss)) for epoch, loss in values]
        import eelbrain

        return eelbrain.load.tsv(path)

    def load_model(
            self,
            epoch: int = -1,
            segment_length: int = None,
            batch_size: int = None,
            stateful: bool = True,
            trainable: bool = True,
            output_hidden: bool = False,
    ) -> keras.models.Model:
        if epoch < 0:
            if not self.checkpoints:
                raise IOError(f"No trained model at {self.model_dir}")
            epoch = self.checkpoints[epoch]
        model = self.make_model(segment_length, batch_size, stateful, output_hidden)
        model.load_weights(self.model_dir / f'{epoch}.hdf5')
        if not trainable:
            for layer in model.layers:
                layer.trainable = False
        return model

    def predict(
            self,
            input_generator: Iterator[Tuple[numpy.ndarray, numpy.ndarray]] = None,
            epoch: int = -1,
            model: keras.models.Model = None,
            words: Sequence[Word] = None,
            snr: float = SNR,  # only used when words are given
    ) -> Dict:
        """Predict and return all activity"""
        if words is not None:
            assert input_generator is None
            words = list(words)
            if any(word.lexicon is not self.lexicon for word in words):
                words = [Word(word.t0, word.speaker, word.word, self.lexicon) for word in words]
            input_generator = gen_input_from_words(iter(words), self.lexicon, snr=snr, target_embedded_region=bool(self.loss_weights_to))
        if model is None:
            model = self.load_model(epoch, batch_size=1, output_hidden=True)
        data = []
        losses = []
        for inputs, targets in input_generator:
            *hidden, outputs = model.predict(inputs, verbose=0)
            data.append((inputs[0], outputs[0], targets[0], *[h[0] for h in hidden]))
            losses.append(model.loss(targets, outputs))
        # concatenate segments: [in, out, targets, h0, h0-sub, h1, h1-sub, ...]
        data = [numpy.concatenate(segs) for segs in zip(*data)]
        out = {'inputs': data[0], 'outputs': data[1], 'targets': data[2], 'loss': losses}
        # remove target-region embedded in targets
        if self.loss_weights_to:
            out['targets'] = out['targets'][..., :-1]
        # hidden unit activations
        multi_layer = len(self._layer_spec) > 1
        i_data = 3
        for i_layer, (n_hidden, subset) in enumerate(self._layer_spec):
            tag = f'hidden{i_layer}' if multi_layer else 'hidden'
            out[tag] = data[i_data]
            if subset:
                out[f'{tag}-local'] = data[i_data][:, subset:]
                i_data += 1
                out[f'{tag}-out'] = data[i_data]
            i_data += 1
        return out


@dataclasses.dataclass()
class Word:
    "Stimulus word token"
    t0: int
    speaker: str
    word: str
    lexicon: Lexicon = dataclasses.field(repr=False)
    pattern: numpy.ndarray = dataclasses.field(repr=False, default=None)  # time | frequency
    target: numpy.ndarray = dataclasses.field(repr=False, default=None)  # dim | time, dim
    part: int = 0  # whether the word continues in the next sequence
    t1: int = None
    t1_weights: int = None  # only works with OneHot box

    def __post_init__(self):
        assert self.t0 >= 0
        if self.pattern is None:
            self.pattern = self.lexicon.inputs[self.speaker, self.word]
        if self.t1 is None:
            self.t1 = self.t0 + self.pattern.shape[0]
            if self.lexicon.target_shape_type == 'offset':
                self.t1 += self.lexicon.target_length
            # For embedded loss weight region
            if self.t1_weights is None:
                pass
            elif self.t1_weights > 0:
                self.t1_weights += self.t0
            elif self.t1_weights < 0:
                self.t1_weights += self.t1
            else:
                raise ValueError(f"{self.t1_weights=}")

    @property
    def t1_pattern(self):
        "End of the input pattern; WARNING: not valid for split words"
        if self.lexicon.target_shape_type == 'offset':
            return self.t1 - self.lexicon.target_length
        return self.t1

    def make_target(
            self,
            target: Target,
    ):
        if target.name == 'word':
            pattern = target.embedding[target.index[self.word]]
        elif target.name == 'speaker':
            pattern = target.embedding[target.index[self.speaker]]
        else:
            pattern = target.embedding[target.map_index[self.word]]

        if self.lexicon.target_shape_type == 'ramp':
            t_word = self.t1 - self.t0
            pattern = numpy.repeat(pattern[None], t_word, axis=0)
            pattern[:-self.lexicon.target_length] *= numpy.linspace(0, 1, t_word-self.lexicon.target_length, endpoint=False)[:, numpy.newaxis]
        return pattern

    def apply(
            self,
            inputs: numpy.ndarray,
            targets: numpy.ndarray = None,
            target: Target = None,
    ):
        # inputs
        t1_pattern = self.t0 + self.pattern.shape[0]
        if t1_pattern > self.t0:
            inputs[self.t0: t1_pattern] += self.pattern
        # outputs
        if targets is None:
            return
        elif self.lexicon.target_shape_type == 'offset':
            t0_target = t1_pattern
        else:
            t0_target = self.t0
        if t0_target < self.t1:
            if self.target is None:
                pattern = self.make_target(target)
            else:
                pattern = self.target
            targets[t0_target: self.t1, :target.ndim] = pattern
        # Loss weights
        if self.t1_weights is not None and self.t1_weights > self.t0:
            targets[self.t0: self.t1_weights, -1] = 1.

    def split(self, t: int, target: Target):
        t_rel = t - self.t0
        target = self.make_target(target)
        if target.ndim == 1:
            target_1 = target_2 = target
        else:
            target_1, target_2 = target[:t_rel], target[t_rel:]
        w1 = dataclasses.replace(self, t1=t, pattern=self.pattern[:t_rel], target=target_1, part=1)
        w2 = dataclasses.replace(self, t0=t, pattern=self.pattern[t_rel:], target=target_2, part=2)
        return w1, w2
    
    def __sub__(self, t):
        return dataclasses.replace(self, t0=self.t0-t, t1=self.t1-t, t1_weights=None if self.t1_weights is None else self.t1_weights-t)
    

def gen_words(
        lexicon: Lexicon,
        items: List[Tuple[str, str]],
        seed: int = 0,
        tmax: int = -1,
        n_words: int = N_WORDS,  # average number of words between silence (phrase length). 0 for no silence; 1 for silence between each word; values > 1 use random generator to average phrase length
        n_phrases: int = N_PHRASES,  # number of phrases by the same speaker (default: random speaker for each word)
        min_silence: int = MIN_SILENCE,  # in samples
        max_silence: int = MAX_SILENCE,  # in samples
        replacement: bool = True,
        loss_weights_to: int = None,
):
    "generate (speaker, word, t_start, t_stop) tuples"
    random = Random(seed)
    assert n_words >= 0
    if n_words:
        assert max_silence > 0
        assert max_silence >= min_silence >= 0
        t = random.randint(min_silence, max_silence)
    else:
        t = 0
    # sort words by speaker for generating phrases
    if n_phrases:
        if not replacement:
            raise NotImplementedError(f"{replacement=} with n_phrases > 0")
        items_by_speaker = defaultdict(list)
        for speaker, word in items:
            items_by_speaker[speaker].append(word)
        speakers = list(items_by_speaker)
        speaker = random.choice(speakers)
        i_phrase = 0
    else:
        items_by_speaker = speaker = speakers = i_phrase = None
        if not replacement:
            items = list(items)
            random.shuffle(items)
            items = iter(items)

    while True:
        # select word
        if n_phrases:
            word = random.choices(items_by_speaker[speaker])[0]
        elif replacement:
            speaker, word = random.choices(items)[0]
        else:
            try:
                speaker, word = next(items)
            except StopIteration:
                return
        # determine timing
        if n_words == 1 or (n_words and random.randint(1, n_words) == 1):
            t += random.randint(min_silence, max_silence)
        word_obj = Word(t, speaker, word, lexicon, t1_weights=loss_weights_to)
        if 0 < tmax < word_obj.t1:
            return
        yield word_obj
        t = word_obj.t1_pattern
        # cycle speakers
        if n_phrases:
            i_phrase += 1
            if i_phrase >= n_phrases:
                speaker = random.choice(speakers)
                i_phrase = 0


def gen_rand_data(
        lexicon: Lexicon,
        items: List[Tuple[str, str]],
        n: int = -1,
        seed: int = 0,
        n_words: int = N_WORDS,  # average phrase length
        n_phrases: int = N_PHRASES,  # number of phrases by the same speaker (default: random speaker for each word)
        min_silence: int = MIN_SILENCE,
        max_silence: int = MAX_SILENCE,
        snr: float = SNR,  # SNR in dB; numpy.inf to skip noise
        segment_length: int = 1000,
        target: Target = None,  # output target (default word)
        loss_weights_to: int = None,
        yield_targets: bool = True,
        yield_words: bool = False,
        replacement: bool = True,
):
    "keep generating data"
    if n_phrases and not replacement:
        raise NotImplementedError(f"{replacement=} with n_phrases > 0")
    iterator = gen_words(lexicon, items, seed, -1, n_words, n_phrases, min_silence, max_silence, replacement, loss_weights_to)
    target_embedded_region = loss_weights_to is not None
    return gen_input_from_words(iterator, lexicon, n, seed, snr, segment_length, target, yield_targets, yield_words, target_embedded_region)


def gen_input_from_words(
        iterator: Iterator[Word],
        lexicon: Lexicon,
        n: int = -1,
        seed: int = 0,
        snr: float = SNR,  # SNR in dB; numpy.inf to skip noise
        segment_length: int = 1000,
        target: Target = None,  # output target (default word)
        yield_targets: bool = True,
        yield_words: bool = False,
        target_embedded_region: bool = False,
) -> Generator[Union[numpy.ndarray, Tuple[numpy.ndarray, ...]], None, None]:
    if target is None:
        target = lexicon.word_target
    np_random = numpy.random.RandomState(seed)
    if snr is numpy.inf:
        noise_std = None
    else:
        # snr = 10 * log(sig / noise)
        # snr / 10 = log(sig / noise)
        # 10 ** (snr / 10) = sig / noise
        # noise = sig / 10 ** (snr / 10)
        noise_std = numpy.sqrt(lexicon.input_variance / (10**(snr / 10)))
    i_segment = 0
    t_start = 0
    next_words = []
    while n < 0 or i_segment < n:
        t_stop = t_start + segment_length
        # shift words in queue
        words, word_queue, next_words = [], next_words, []
        # find words for this stimulus
        for word in chain(word_queue, iterator):
            if word.t1 < t_stop:
                words.append(word)
            elif word.t0 < t_stop:
                w1, w2 = word.split(t_stop, target)
                words.append(w1)
                next_words.append(w2)
            else:
                next_words.append(word)
                break
        else:  # iterator finished
            if not (words or next_words):
                return
        # add words to spectrograms
        words = [w - t_start for w in words]
        if snr is numpy.inf:
            inputs = numpy.zeros((segment_length, lexicon.n_bands), numpy.float32)
        else:
            inputs = np_random.normal(0, noise_std, (segment_length, lexicon.n_bands))

        if yield_targets:
            ndim = target.ndim + bool(target_embedded_region)
            targets = numpy.zeros((segment_length, ndim), numpy.float32)
        else:
            targets = None

        for word in words:
            word.apply(inputs, targets, target)
        # yield stimulus
        if yield_targets:
            if yield_words:
                yield inputs[None], targets[None], words
            else:
                yield inputs[None], targets[None]
        elif yield_words:
            yield inputs[None], words
        else:
            yield inputs[None]  # batch, time, frequency
        # increment counters
        i_segment += 1
        t_start = t_stop


class BatchGenerator:

    def __init__(
            self,
            lexicon: Lexicon,
            items: List[Tuple[str, str]],
            n: int = -1,
            seed: int = 0,
            n_words: int = N_WORDS,  # average phrase length
            n_phrases: int = N_PHRASES,  # number of phrases by the same speaker (default: random speaker for each word)
            min_silence: int = MIN_SILENCE,
            max_silence: int = MAX_SILENCE,
            snr: float = SNR,  # SNR in dB; numpy.inf to skip noise
            segment_length: int = 1000,
            batch_size: int = 1,
            target: Target = None,  # output target (default word)
            loss_weights_to: int = None,
            queue_size: int = 50,
    ):
        random = Random(seed)
        self.stop = threading.Event()
        self.queues = []
        self.threads = []
        for i in range(batch_size):
            data_queue = queue.Queue(queue_size)
            seed_ = random.getrandbits(31)
            thread = threading.Thread(target=self.gen_thread, args=(data_queue, self.stop, lexicon, items, n, seed_, n_words, n_phrases, min_silence, max_silence, snr, segment_length, target, loss_weights_to), name=f"BatchGenerator-{i}")
            thread.start()
            self.queues.append(data_queue)
            self.threads.append(thread)

    @staticmethod
    def gen_thread(
            data_queue: queue.Queue,
            stop: threading.Event,
            *args,
    ):
        for data in gen_rand_data(*args):
            data_queue.put(data)
            if stop.is_set():
                break
        data_queue.put(None)
        return

    def __iter__(self):
        return self

    def __next__(self):
        data = [q.get() for q in self.queues]
        if data[0] is None:
            raise StopIteration
        inputs, targets = zip(*data)
        return numpy.concatenate(inputs, 0), numpy.concatenate(targets, 0)

    def shutdown(self):
        "Shutdown threads (only needed with endless generation)"
        self.stop.set()
        for _ in self:
            pass
        for thread in self.threads:
            thread.join()


def write_info(path: Path, *lines):
    text = '\n'.join([
        f"Host: {socket.gethostname()}",
        f"Time: {datetime.datetime.now():%Y-%m-%d}",
        f"TensorFlow: {tf.__version__}",
        *lines,
    ])
    print(text)
    with path.open('a') as file:
       file.write(text)


# ArgParse for model parameters
# #############################

class SNRAction(argparse.Action):

    def __call__(self, parser, namespace, values, option_string=None):
        snr = numpy.inf if values == 'inf' else int(values)
        setattr(namespace, self.dest, snr)


def add_model_args(parser: argparse.ArgumentParser):
    # Stimuli
    parser.add_argument('--lexicon', '-l', type=str, default='2056')
    parser.add_argument('--stimulus', choices=STIMULUS_SEQUENCES.keys(), default='2')
    parser.add_argument('--snr', default=20, action=SNRAction, help="Stimulus SNR (`inf` or integer)")
    # Targets
    parser.add_argument('--target', '-t', default='Glove-50c', help="Main choices: OneHot; Glove-<n>[c] (n=50, 100, 200; c to center the embedding); Sparse-<n> (n=number of non-zero values for each target).")
    parser.add_argument('--target-shape', choices=('box', 'ramp-10', 'offset-40'), default='box')
    # Model settings
    parser.add_argument('--mechanism', '-m', choices=('LSTM', 'GRU', 'RNN'), default='LSTM')
    parser.add_argument('--hidden', type=str, default='512-50', help="E.g. 512-50x512-50")
    parser.add_argument('--activation', type=str, default=None)
    parser.add_argument('--activation-x', type=str, default=None)
    parser.add_argument('--regularize', type=float, default=0)
    parser.add_argument('--regularize-x', type=float, default=0)
    parser.add_argument('--loss', type=str, default=None, help="Use non-default loss function: abs | ms | w<weight> | w<weight>to<-index>")
    parser.add_argument('--seed', type=int, default=None, help="Random seed for model fitting")
    parser.add_argument('--stimulus-seed', type=int, default=0, help="Random seed for stimulus generation")
    parser.add_argument('--optimizer', choices=['RMSProp', 'Adam'], default='RMSProp')
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--patience', type=int, default=200, help='Abort training after this many epochs without improvement')
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--steps_per_epoch', type=int, default=80)
    parser.add_argument('--n_bands', type=int, default=50)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', '-d', type=str, default='/gpu:0')
    parser.add_argument('--overwrite', action='store_true', default=False)
    parser.add_argument('--resume', action='store_true', default=False)
    add_model_args(parser)
    args = parser.parse_args()

    parameters = Model.from_args(args)
    print("** Train **")
    print(parameters.name)
    if parameters.model_dir.exists():
        if args.overwrite:
            print("Target directory exists, overwriting...")
            shutil.rmtree(parameters.model_dir)
        elif not args.resume:
            print("Target directory exists, aborting.")
            return

    if parameters.seed is not None:
        set_random_seed(args.seed)

    info_file = parameters.model_dir / 'info.txt'
    device = tf.device(args.device)
    with device:
        stimulus_seed = parameters.stimulus_seed
        if parameters.model_dir.exists():
            epoch = parameters.checkpoints[-1]
            write_info(
                info_file,
                f"Resuming training at epoch {epoch} ...",
                f"Device: {args.device}",
                f"Code: {get_git_revision_hash()}",
            )
            model = parameters.load_model()
            stimulus_seed += epoch
        else:
            epoch = 0
            parameters.model_dir.mkdir(parents=True)
            write_info(
                info_file,
                f"Model: {parameters.name}",
                f"Device: {args.device}",
                f"Code: {get_git_revision_hash()}",
            )
            # initialize model
            model = parameters.make_model()
            model.save(parameters.model_dir / '0.hdf5')
        model.summary()
        # callbacks
        callbacks = [
            keras.callbacks.ModelCheckpoint(parameters.model_dir / '{epoch}.hdf5', 'loss', save_best_only=True, save_weights_only=True, save_freq=250 * parameters.steps_per_epoch),
            # keras.callbacks.ReduceLROnPlateau('loss', patience=100, verbose=1, min_delta=0),
            # keras.callbacks.EarlyStopping('loss', patience=500),
            keras.callbacks.CSVLogger(parameters.model_dir / 'log.csv', append=True),
            Backtrack('loss', stop_patience=parameters.patience, target_directory=parameters.model_dir),
        ]
        # fit data
        stimulus = parameters.stimulus
        data = BatchGenerator(parameters.lexicon, parameters.lexicon.train_items, -1, stimulus_seed, stimulus.n_words, stimulus.n_phrases, stimulus.min_silence, stimulus.max_silence, stimulus.snr, parameters.segment_length, parameters.batch_size, loss_weights_to=parameters.loss_weights_to)
        t_start = time.time()
        history_obj = model.fit(data, steps_per_epoch=parameters.steps_per_epoch, epochs=20_000, callbacks=callbacks, initial_epoch=epoch)

    # Save fit time
    dt = time.time() - t_start
    h = int(dt // 3600)
    m = int((dt - h * 3600) // 60)
    s = int(dt % 60)
    with info_file.open('at') as f:
        f.write(f"\nFit time: {h}:{m}:{s}")

    data.shutdown()
    sys.exit("Done")
