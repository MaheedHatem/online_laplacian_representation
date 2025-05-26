import numpy as np


def get_summary_str(step=None, info=None, prefix=''):
    summary_str = prefix
    if step is not None:
        summary_str += 'Step {}; '.format(step)
    for key, val in info.items():
        if isinstance(val, (int, np.int32, np.int64)):
            summary_str += '{} {}; '.format(key, val)
        elif isinstance(val, (float, np.float32, np.float64)):
            summary_str += '{} {:.4g}; '.format(key, val)
    return summary_str