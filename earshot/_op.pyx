# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
#cython: boundscheck=False, wraparound=False, cdivision=True

# cimport cython
# from cython.view cimport array as cvarray
# from libc.stdlib cimport malloc, free
import numpy as np
cimport numpy as np

ctypedef np.int64_t INT64
ctypedef np.float32_t FLOAT32


def binary_vector_activation_min(
    FLOAT32 [:,:] output,  # (n_times, dim) network output
    INT64 [:,:] indices,  # (n_words, n_indices)
):  # -> time, word
    cdef:
        Py_ssize_t i_time, i_word, i_index
        FLOAT32 min_v, max_v_for_time
        Py_ssize_t n_times = output.shape[0]
        Py_ssize_t n_words = indices.shape[0]
        Py_ssize_t n_indices = indices.shape[1]
        FLOAT32 [:,:] activation = np.empty((n_times, n_words), dtype=np.float32)

    with nogil:
        activation[...] = 0

        for i_time in range(n_times):
            # max_v_for_time = 0
            for i_word in range(n_words):
                min_v = 1
                for i_index in range(n_indices):
                    min_v = min(min_v, output[i_time, indices[i_word, i_index]])
                activation[i_time, i_word] = min_v
                # max_v_for_time = max(max_v_for_time, min_v)
            # activation[i_time, n_words] = 1 - max_v_for_time  # silence

    return np.asarray(activation)


def binary_vector_activation_mean(
    FLOAT32 [:,:] output,  # (n_times, dim) network output
    INT64 [:,:] indices,  # (n_words, n_indices)
    int order,
):
    cdef:
        Py_ssize_t i_time, i_word, i_index
        FLOAT32 v
        Py_ssize_t n_times = output.shape[0]
        Py_ssize_t n_words = indices.shape[0]
        Py_ssize_t n_indices = indices.shape[1]
        FLOAT32 [:,:] activation = np.empty((n_times, n_words), dtype=np.float32)

    with nogil:
        activation[...] = 0

        for i_time in range(n_times):
            for i_word in range(n_words):
                v = 0
                for i_index in range(n_indices):
                    v += output[i_time, indices[i_word, i_index]]
                activation[i_time, i_word] = v / order

    return np.asarray(activation)
