"""Evaluate with stimulus from training generator
"""
import argparse
import dataclasses
import pickle
import queue
import sys
import threading
from typing import Dict, Sequence

import numpy
import tensorflow as tf

from .find_competitors import read_competitors
from .train_earshot import STIMULUS_SEQUENCES, Target, Lexicon, Stimulus, Model, Word, SNRAction, gen_rand_data, add_model_args


@dataclasses.dataclass
class Candidate:
    # Word candidate with activation over time
    word: str
    p: numpy.ndarray = dataclasses.field(repr=False)
    # kind: Literal['target', 'competitor']


@dataclasses.dataclass
class WordEpoch:
    # fixed time-window epoch word data
    word: str
    speaker: str
    trained_item: bool
    overall_winner: str
    last_winner: str
    last_10_winner: str
    candidates: Dict[str, Candidate]  # Softmax
    activation: Dict[str, Candidate] = None  # Raw output/distance


class Evaluator:

    def __init__(
            self,
            lexicon: Lexicon,
            target: Target,
            softmax: bool = False,
            cool: int = None,
    ):
        if lexicon.target_shape_type not in ('box', 'offset'):
            raise NotImplementedError(f"{lexicon.target_shape_type=}")
        self.lexicon = lexicon
        self.target = target
        self.softmax = softmax
        self.cool = cool
        self.results = []
        # input segmenting thread
        self._input_queue = queue.Queue(50)
        self._segment_thread = threading.Thread(target=self._segment)
        self._segment_thread.start()
        # analysis thread
        self._word_queue = queue.Queue(250)
        self._process_thread = threading.Thread(target=self._process)
        self._process_thread.start()

    def add_data(
            self,  # Data for one input segment
            words: Sequence[Word],
            output: numpy.ndarray,  # (time x target)
            is_trained: bool,
    ):
        self._input_queue.put((words, output, is_trained))

    def join(self):
        self._input_queue.put(None)
        self._segment_thread.join()
        self._process_thread.join()

    def _segment(self):
        split_output_buffer = {}
        while item := self._input_queue.get():
            words, output, is_trained = item  # cf. self.add_data()
            for word in words:
                if self.lexicon.target_shape_type == 'offset':
                    t0_target = word.t0 + word.pattern.shape[0]
                    t1_target = t0_target + self.lexicon.target_length
                else:
                    t0_target = word.t0
                    t1_target = word.t0 + word.pattern.shape[0]

                if word.part == 0:
                    word_output = output[t0_target: t1_target]
                elif word.part == 1:
                    if t0_target >= 0:
                        split_output_buffer[word.speaker, word.word] = output[t0_target:]
                    else:
                        split_output_buffer[word.speaker, word.word] = None
                    continue
                elif word.part == 2:
                    split_output = split_output_buffer.pop((word.speaker, word.word))  # could fail when repeating a token
                    word_output = output[t0_target: t1_target]
                    if split_output is not None:
                        word_output = numpy.concatenate((split_output, word_output), 0)
                else:
                    breakpoint()
                if not word_output.shape[0]:
                    breakpoint()
                self._word_queue.put((word, word_output, is_trained))
        self._word_queue.put(None)

    def _process(self):
        competitors = read_competitors(self.lexicon.base_name, self.lexicon.words)
        while item := self._word_queue.get():
            word, output, is_trained = item
            activation = self.target.activation(output)
            softmax = self.target.softmax(output, False, self.cool)

            # Word with the largest overall activation
            overall_winner = self.target.items[numpy.argmax(softmax.max(axis=0))]
            # Largest activation at the end of the word
            last_10_winner = self.target.items[numpy.argmax(softmax[-10:].max(axis=0))]

            # Store time courses
            i_target = self.target.index[word.word]
            activation_tc = {
                'target': Candidate(word.word, activation[:, i_target].copy()),
                'total': Candidate('total', activation.sum(1)),
            }
            if self.softmax:
                softmax_tc = {
                    'target': Candidate(word.word, softmax[:, i_target].copy()),
                }
            else:
                softmax_tc = None
            # Competitors
            for category, words in competitors[word.word].items():
                if not words:
                    continue
                index = [self.target.index[competitor] for competitor in words if competitor in self.target.index]
                activation_tc[category] = Candidate(category, activation[:, index].mean(1))
                if self.softmax:
                    softmax_tc[category] = Candidate(category, softmax[:, index].mean(1))

            self.results.append(WordEpoch(word.word, word.speaker, is_trained, overall_winner, '', last_10_winner, softmax_tc, activation_tc))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', '-d', type=str, default='/gpu:0')
    add_model_args(parser)
    # model stage
    parser.add_argument('--epoch', type=int, default=-1, help="Training epoch")
    # test stimuli
    parser.add_argument('--test-stimulus', choices=STIMULUS_SEQUENCES.keys(), default='NoSil')
    parser.add_argument('--test-snr', default=numpy.inf, action=SNRAction, help="Test stimulus SNR (`inf` or integer)")
    parser.add_argument('--cool', type=int, default=None, help="Cooling factor for softmax - higher numbers decrease influence of low probability competitors.")
    parser.add_argument('--overwrite', action='store_true', default=False)
    args = parser.parse_args()
    assert args.epoch == -1

    earshot_parameters = Model.from_args(args)
    stimulus = Stimulus(earshot_parameters.lexicon, args.test_stimulus, args.test_snr)
    name = f'eval {args.test_stimulus} {args.test_snr}'
    if args.cool:
        name += f' cool-{args.cool}'

    print("** Evaluate **")
    print(earshot_parameters.name)
    print(name)

    dst = earshot_parameters.model_dir / f'{name}.pickle'
    if dst.exists():
        if args.overwrite:
            print(f"Target exists, will overwrite...")
        else:
            sys.exit(f"Target already exists, exiting.")

    device = tf.device(args.device)
    with device:
        model = earshot_parameters.load_model(args.epoch, batch_size=1)
        lexicon = Lexicon(args.lexicon, args.target, args.target_shape, args.n_bands)
        evaluator = Evaluator(lexicon, lexicon.word_target, cool=args.cool)
        for is_trained, items in ((True, lexicon.train_items), (False, lexicon.test_items)):
            for inputs, words in gen_rand_data(lexicon, items, -1, 0, stimulus.n_words, stimulus.n_phrases, stimulus.min_silence, stimulus.max_silence, stimulus.snr, yield_targets=False, yield_words=True, replacement=False):
                output = model.predict(inputs, verbose=0)
                evaluator.add_data(words, output[0], is_trained)
        evaluator.join()

    with dst.open('wb') as file:
        pickle.dump(evaluator.results, file)
