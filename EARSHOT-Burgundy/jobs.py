# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from functools import partial

from burgundy import e

from constants2 import model


WHOLEBRAIN = {
    'raw': 'ica1-20',
    'samplingrate': 50,
    'cv': True,
    'partitions': -4,
    'inv': 'fixed-6-MNE-0',
    'mask': 'wholebrain-2',
    'tstart': -0.100,
    'tstop': 1.000,
    'error': 'l2',
    'selective_stopping': 1,
    # 'partition_results': True,
}
STG = {**WHOLEBRAIN, 'mask': 'STG301'}

JOBS = [
    # Baseline
    e.trf_job(f"gt-log8 + phone-p0", **WHOLEBRAIN),
]

##############
# Output space
##############
JOBS.extend([
    e.trf_job(f"gt-log8 + phone-p0 + {model(hidden, target_space=target)}", **STG)
    for hidden in ['512', '1024', '2048']
    for target in ['Glove-300c', 'Sparse-10of300', 'Sparse-10of900']
])

# Whole-brain: is Glove better anywhere?
JOBS.extend([
    e.trf_job(f"gt-log8 + phone-p0 + {model(hidden, target_space=target)}", **WHOLEBRAIN)
    for hidden in ['512', '1024', '2048']
    for target in ['OneHot', 'Glove-50c']
])


#############
# Deep models
#############
# Deep with multiple layers
JOBS.extend([
    e.trf_job(f"gt-log8 + phone-p0 + {model(hidden, target_space=target, k=k)}", **STG)
    for hidden in ['512', '320x320', '256x256x256', '192x192x192x192']
    for k in [None, 2, 4, 8, 16, 32, 64]  # 2, 4, 8, 16, 32, 64
    for target in ['OneHot', 'Glove-50c', 'Glove-300c']
])
# K=3 for flat models
JOBS.extend([
    e.trf_job(f"gt-log8 + phone-p0 + {model(hidden, target_space=target, k=k)}", **STG)
    for hidden in ['512']
    for k in [3]
    for target in ['OneHot', 'Glove-50c', 'Glove-300c']
])


##############
# Cohort loss
##############
JOBS.extend([
    e.trf_job(f"gt-log8 + phone-p0 + {model(hidden, target_space='OneHot', loss=f'dw{loss}to10', k=k)}", **STG)
    for loss in [16, 64, 256, 1024, 4096]
    for k in [32]
    for hidden in ['512', '320x320', '256x256x256', '192x192x192x192']
])


#############
# Final model
#############
rnn = partial(model, '512', target_space='OneHot', loss=f'dw1024to10', k=32)
JOBS.extend([
    # Envelope vs. onsets
    e.trf_job(f"gt-log8 + phone-p0 + {rnn(transform='sum')}", **STG),
    e.trf_job(f"gt-log8 + phone-p0 + {rnn(transform='onset')}", **STG),
    # Auditory
    e.trf_job(f"phone-p0 + {rnn()}", **STG),
    # Surprisal
    e.trf_job(f"gt-log8 + phone-p0 + {rnn()} + phone-any +@ phone-surprisal", **STG),
    e.trf_job(f"gt-log8 + phone-p0 + phone-any +@ phone-surprisal", **STG),
])
