from scipy.signal import cwt, find_peaks_cwt, ricker, welch

def _get_length_sequences_where(x):
    if len(x) == 0:
        return [0]
    else:
        res = [len(list(group)) for value, group in itertools.groupby(x) if value == 1]
        return res if len(res) > 0 else [0]


def percentile(n):
    '''Calculate n - percentile of data'''
    def percentile_(x):
        return np.nanpercentile(x, n)

    percentile_.__name__ = 'perc%s' % n
    return percentile_


def variation(x):
    mean = x.mean()
    if mean != 0:
        return x.std() / mean
    else:
        return np.nan
    
def abs_energy(x):
    
    return np.nansum(x * x)

def absolute_sum_of_changes(x):
    
     return np.nansum(np.abs(np.diff(x)))
    
def count_above_mean(x):
    
    m = x.mean()
    return np.where(x > m)[0].size

def count_below_mean(x):

    m = x.mean()
    return np.where(x < m)[0].size


def count_above(t):
    def count_above_(x):
        return np.sum(x >= t) / len(x)
    
    count_above_.__name__ = 'count_above%s' % t
    return count_above_

def count_below(t):
    def count_below_(x):
        return np.sum(x <= t) / len(x)
    
    count_above_.__name__ = 'count_above%s' % t
    return count_above_mean

def first_location_of_minimum(x):
    
    return np.argmin(x) / len(x) if len(x) > 0 else np.NaN

def first_location_of_maximum(x):
    return np.argmax(x) / len(x) if len(x) > 0 else np.NaN


def longest_strike_below_mean(x):
    
    return np.max(_get_length_sequences_where(x < x.mean())) if x.size > 0 else 0

def longest_strike_above_mean(x):

    return np.max(_get_length_sequences_where(x > x.mean())) if x.size > 0 else 0


def mean_second_derivative_central(x):

    return (x[-1] - x[-2] - x[1] + x[0]) / (2 * (len(x) - 2)) if len(x) > 2 else np.NaN

def number_crossing_m(x):
    m = x.mean()
    positive = x > m
    return np.where(np.diff(positive))[0].size

def number_cwt_peaks(n):    
    def number_cwt_peaks_(x):

        return len(
            find_peaks_cwt(vector=x, widths=np.array(list(range(1, n + 1))), wavelet=ricker)
        )
    
    return number_cwt_peaks_

def range_count(x):
    
    min_ = x.mean() - 3 * x.std()
    max_ = x.mean() + 3 * x.std()
    return np.sum((x >= min_) & (x < max_))


def ratio_beyond_r_sigma(r):

    def ratio_beyond_r_sigma_(x):
        return np.sum(np.abs(x - x.mean()) > r * x.std()) / x.size
    
    return ratio_beyond_r_sigma_

def root_mean_square(x):
    
    return np.sqrt(np.square(x).mean()) if len(x) > 0 else np.NaN

def symmetry_looking(r):

    def symmetry_looking_(x):
        mean_median_difference = np.abs(x.mean() - x.median())
        max_min_difference = x.max() - x.min()
        return mean_median_difference < (r * max_min_difference)
    
    return symmetry_looking_

def benford_correlation(x):

    x = np.array(
        [int(str(np.format_float_scientific(i))[:1]) for i in np.abs(np.nan_to_num(x))]
    )

    benford_distribution = np.array([np.log10(1 + 1 / n) for n in range(1, 10)])
    data_distribution = np.array([(x == n).mean() for n in range(1, 10)])

    return np.corrcoef(benford_distribution, data_distribution)[0, 1]

def cid_ce(x):
    x = np.diff(x)
    return np.sqrt(np.nansum(x * x))

