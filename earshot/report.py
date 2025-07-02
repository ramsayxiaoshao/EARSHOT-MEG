# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from functools import cached_property
from math import ceil
from typing import Dict, List, Literal

import matplotlib.cm
import numpy
from eelbrain import fmtxt, load, plot, Case, Dataset, NDVar, UTS

from .earshot_lexicon import DATA_DIR
from .train_earshot import SNR, Model, Word, gen_words


class Example:

    def __init__(self, model: Model):
        self.trainer = model

    @cached_property
    def model(self):
        return self.trainer.load_model(batch_size=1, output_hidden=True)

    def _predict(
            self,
            n: int = 1,  # number of segments
            test_items: bool = False,
            *,
            words: List[Word] = None,
            snr: float = SNR,
            **gen_word_args,
    ) -> (List[Word], Dict):
        if words is None:
            items = self.trainer.lexicon.test_items if test_items else self.trainer.lexicon.train_items
            words = gen_words(self.trainer.lexicon, items, tmax=n*self.trainer.segment_length, loss_weights_to=self.trainer.loss_weights_to, **gen_word_args)
        return words, self.trainer.predict(words=words, snr=snr, model=self.model)

    def load_activation(
            self,
            stimuli: str = 'NoSil inf',
            pad: Literal[0, -1] = None,  # epochs last the duration of the pattern
            cool: int = None,
    ) -> (List, Dataset, Dataset, Dataset):  # results, ds_correct, ds_softmax, ds_activation
        results, ds_correct = self.load_results(stimuli, cool)
        out = [results, ds_correct]
        # Activation
        n_times = max(result.activation['target'].p.shape[0] for result in results)
        time = UTS(0, 0.01, n_times)
        for attr in ('candidates', 'activation'):
            if getattr(results[0], attr) is None:
                out.append(None)
                continue
            rows = []
            time_courses = []
            for r in results:
                correct = r.word == r.last_10_winner
                prefix = [r.word, r.speaker, r.trained_item, correct]
                candidates = getattr(r, attr)
                for key, time_course in candidates.items():
                    rows.append([*prefix, key])
                    time_courses.append(time_course.p)
            ds_activation = Dataset.from_caselist(['word', 'speaker', 'trained', 'correct', 'kind'], rows, info={'model': self.trainer.name})
            x = numpy.zeros((len(rows), n_times)) + 1 / self.trainer.lexicon.n_words
            for i, xi in enumerate(time_courses):
                t_stop = len(xi)
                x[i, :t_stop] = xi
                if pad is not None:
                    x[i, t_stop:] = xi[pad]
            ds_activation['activation'] = NDVar(x, (Case, time))
            out.append(ds_activation)
        return out

    def load_log(self):
        return load.tsv(self.trainer.model_dir / 'log.csv')

    def load_results(
            self,
            stimuli: str = 'NoSil inf',
            cool: int = None,
    ) -> (List, Dataset):
        # load results
        ext = f' cool-{cool:g}' if cool else ''
        results = load.unpickle(DATA_DIR / 'Models' / self.trainer.name / f'eval {stimuli}{ext}.pickle')
        # Add n correct
        rows = []
        for r in results:
            overall = r.word == r.overall_winner
            last10 = r.word == r.last_10_winner
            rows.append([r.word, r.speaker, r.trained_item, overall, last10])
        ds_correct = Dataset.from_caselist(['word', 'speaker', 'trained', 'overall', 'last10'], rows)
        return results, ds_correct

    def output(
            self,
            w=20,
            h=2.5,
            times: int = 2000,  # number of time points to plot
            target: bool = False,  # plot training signal (instead of output)
            title: str = None,
            # Figure
            gridspec_kw: dict = None,
            samples: bool = False,
            # Input
            cmap: str = 'Blues',
            vmax: float = 0.1,
            **gen_word_args,
    ):
        from matplotlib import pyplot

        # generate data
        n = int(ceil(times / self.trainer.segment_length))
        words, data = self._predict(n, **gen_word_args)
        if data['inputs'].shape[0] != times:
            for key in data.keys():
                if key == 'loss':
                    continue
                data[key] = data[key][:times]
            words = [w for w in words if w.t0 < times]

        # Colormap
        n_words = self.trainer.lexicon.n_words
        word_cmap = matplotlib.colormaps.get_cmap("jet")
        colors = word_cmap.resampled(n_words)(range(n_words))

        # plot
        fig, axes = pyplot.subplots(2, 1, sharex=True, figsize=(w, h), gridspec_kw=gridspec_kw)
        if title:
            fig.suptitle(title)
        # Input
        axes[0].imshow(data['inputs'].T, origin='lower', aspect='auto', vmax=vmax, cmap=cmap)
        axes[0].set_ylabel('Input')
        axes[0].set_yticks(())
        # Output
        axes[1].set_prop_cycle(color=colors)
        if target:
            output_data = data['targets']
        else:
            output_data = data['outputs']
        if self.trainer.lexicon.target_space == 'OneHot':
            output_data = output_data[:, self.trainer.lexicon.phone_sort]
        axes[1].plot(output_data)
        axes[1].set_ylabel('Output')
        for word in words:
            axes[1].axvline(word.t0, color='.5', linestyle='--')
            axes[1].axvline(word.t1, color='.5', linestyle='-')
        axes[1].set_xlim(0, times)
        if samples:
            axes[1].set_xlabel('Time (sample)')
        else:
            axes[1].set_xlabel('Time (s)')
            axes[1].set_xticklabels([f'{sample / 100:.0f}' for sample in axes[1].get_xticks()])
        if self.trainer.lexicon.target_space.startswith('Glove'):
            ymin, ymax = axes[1].get_ylim()
            ymax = max(ymax, abs(ymin))
            axes[1].set_ylim(-ymax, ymax)
            axes[1].set_yticks(())
        else:
            axes[1].set_ylim(0, 1.2)
        # finalize
        fig.align_ylabels(axes)
        return fig, words

    def recognition(
            self,
            stimuli: str = 'NoSil inf',
            bottom: float = None,
            top: float = None,
            ext: str = '',
            timecourse: str = 'softmax',
            title: str = None,
            w: int = 3,
            h: int = 2,
    ):
        _, ds_correct, ds_softmax, ds_activation = self.load_activation(stimuli, ext)
        # determine y-limits
        if timecourse == 'activation':
            if bottom is None:
                bottom = 0
            if top is None:
                top = 1
        elif self.trainer.target_space.startswith('Glove'):
            if top is None:
                top = 0.008
            if bottom is None:
                bottom = 0.
        elif self.trainer.target_space.startswith('OneHot'):
            if top is None:
                top = 0.0008
            if bottom is None:
                bottom = 0.0003
        elif self.trainer.target_space.startswith(('Sparse', 'Balanced')):
            if top is None:
                top = 0.008
            if bottom is None:
                bottom = 0.
        else:
            raise RuntimeError(self.trainer.target_space)
        # performance
        ds_performance = ds_correct.aggregate("trained", drop_bad=True)
        # competition
        uts_args = dict(w=w, h=h, bottom=bottom, top=top)
        if timecourse == 'softmax':
            uts_args['ds'] = ds_softmax
        elif timecourse == 'activation':
            assert ds_activation is not None, 'activation not computed'
            uts_args['ds'] = ds_activation
        else:
            raise ValueError(f'{timecourse=}')

        doc = fmtxt.Section(title)
        doc.add_figure(None, fmtxt.FloatingLayout([
            ds_performance.as_table(),
            plot.UTSStat('activation', 'kind', sub=f"trained == True", ylabel="Training set", **uts_args),
            plot.UTSStat('activation', 'kind', sub=f"trained == False", ylabel="Testing set", **uts_args),
        ]))
        return doc
