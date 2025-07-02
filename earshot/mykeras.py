# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from pathlib import Path

import tensorflow
from tensorflow import keras
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.engine.input_spec import InputSpec
import numpy as np


class Backtrack(keras.callbacks.Callback):

    def __init__(
            self,
            monitor: str = 'val_loss',
            max_value: float = 1.25,  # only backtrack when the loss increases above max_value * best
            patience: int = 20,  # n (current < best) before backtrack
            stop_patience: int = 200,  # n (current < best) before stopping
            # mode='auto',
            baseline: float = None,
            target_directory: Path = None,
            epoch_filename: str = '{epoch}.hdf5',
    ):
        super(Backtrack, self).__init__()

        self.monitor = monitor
        self.patience = patience
        self.stop_patience = stop_patience
        self.baseline = baseline
        self.max_value = max_value
        self.filepath = str(target_directory / epoch_filename)

        self.wait = 0
        self.wait_to_stop = 0
        self.stopped_epoch = 0
        self.best_epoch = self.best_loss = self.best_weights = None
        self.restored_at = []

        # logging
        self.log_path = target_directory / f'{self.__class__.__name__}.log'
        self.log = []

    def on_train_begin(self, logs=None):
        self.log.append(f'0: Beginning')
        # Allow instances to be re-used
        self.wait = 0
        self.wait_to_stop = 0
        self.stopped_epoch = 0
        self.best_epoch = 0
        if self.baseline is not None:
            self.best_loss = self.baseline
        else:
            self.best_loss = np.Inf  # if self.monitor_op == np.less else -np.Inf
        self.best_weights = None

    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs[self.monitor]
        if current_loss is None:
            return
        elif current_loss <= self.best_loss:
            self.best_epoch = epoch
            self.best_loss = current_loss
            self.wait = self.wait_to_stop = 0
            self.best_weights = self.model.get_weights()
            return
        self.wait += 1
        self.wait_to_stop += 1
        if self.wait_to_stop >= self.stop_patience:
            self.log.append(f'{epoch}: Stopping at {self.stopped_epoch}')
            self.model.stop_training = True
            self.stopped_epoch = epoch
            self.model.set_weights(self.best_weights)
            if self.filepath:
                self.model.save_weights(self.filepath.format(epoch=epoch), overwrite=True, options=tensorflow.train.CheckpointOptions())
        elif self.wait >= self.patience and current_loss > self.best_loss * self.max_value:
            self.log.append(f'{epoch}: Restoring model weights to {self.best_epoch}')
            self.restored_at.append(epoch)
            self.model.set_weights(self.best_weights)
            self.wait = 0

    def on_train_end(self, logs=None):
        self.log_path.write_text('\n'.join(self.log))


class Subset(Layer):
    """Based on Cropping1D"""

    def __init__(self, n, axis=-1, scale=0, name='Subset', **kwargs):
        super(Subset, self).__init__(trainable=False, name=name, **kwargs)
        self.n = int(n)
        self.axis = int(axis)
        assert scale != 1
        self.scale = scale
        if self.axis < 0:
            min_ndim = -self.axis
        else:
            min_ndim = self.axis + 1
        self.input_spec = InputSpec(min_ndim=min_ndim)
        self._index = None

    def compute_output_shape(self, input_shape):
        output_shape = tensor_shape.TensorShape(input_shape).as_list()
        if output_shape[self.axis] is not None:
            output_shape[self.axis] = self.n
        return tensor_shape.TensorShape(output_shape)

    def call(self, inputs):
        if self._index is None:
            axis = self.axis if self.axis >= 0 else len(inputs.shape) + self.axis
            self._index = [slice(None)] * axis + [slice(0, self.n)]
        if self.scale:
            return inputs[self._index] * self.scale
        return inputs[self._index]

    def get_config(self):
        return {**super(Subset, self).get_config(), 'n': self.n, 'axis': self.axis}
