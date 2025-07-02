import functools


@functools.cache
def n_possibilities(start: int, stop: int, n_sample: int) -> int:
    if n_sample == 1:
        return stop - start
    elif n_sample == 0:
        return 1
    elif n_sample < 0:
        raise ValueError(f"{n_sample=} (<0)")
    out = 0
    for v0 in range(start, stop-(n_sample-1)):
        out += n_possibilities(v0+1, stop, n_sample-1)
    return out


def combination(stop: int, n_sample: int, index: int):
    values = []
    i = index
    start = 0
    n_on_right = n_sample
    for i_v in range(n_sample):
        n_on_right -= 1
        if n_on_right == 0:
            values.append(start + i)
            break
        for v in range(start, stop - n_on_right):
            n = n_possibilities(v+1, stop, n_on_right)
            if i < n:
                values.append(v)
                start = v + 1
                break
            i -= n
        else:
            raise IndexError(f"{index=} out of valid range")
    # values.append()
    return values
