import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.keras import backend
from tensorflow.python.keras.losses import LossFunctionWrapper
from tensorflow.python.keras.utils import losses_utils
from tensorflow.python.ops import math_ops


class DownWeightCompetitors(LossFunctionWrapper):

    def __init__(
            self,
            by: float,
            axis: int = -1,
            reduction: str = losses_utils.ReductionV2.AUTO,
            name: str = 'down_weight_competitors',
    ):
        super().__init__(cohort_weighted_loss, reduction, name, non_target_multiplier=1 / by, target_embedded_region=True, axis=axis)
        self.by = by


def cohort_weighted_loss(y_true, y_pred, target_multiplier=1, non_target_multiplier=1, target_embedded_region=False, axis=-1):
    """Lenient penalty for non-target activations"""
    # input shape: batch, time, word
    y_pred = ops.convert_to_tensor_v2_with_dispatch(y_pred)
    y_true = math_ops.cast(y_true, y_pred.dtype)

    if target_embedded_region:
        region = y_true[..., -1:]
        y_true = y_true[..., :-1]
    else:
        region = tf.reduce_any(y_true > 0, -1, keepdims=True)
        region = tf.cast(region, 'float32')

    crossentropy = backend.binary_crossentropy(y_true, y_pred)

    # by default multiply by 1
    mult = tf.ones_like(y_true)
    # apply scaling to all targets in relevant region
    mult += region * (non_target_multiplier - 1)
    # undo and set scaling for the correct target
    target_component = y_true * (target_multiplier - non_target_multiplier)
    if target_embedded_region:
        target_component *= region
    mult += target_component
    # apply scaling
    crossentropy *= mult

    return backend.mean(crossentropy, axis=axis)
