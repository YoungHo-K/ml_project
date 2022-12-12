import numpy as np


def get_max_in_list(values):
    if len(values) == 0:
        return 0

    return np.max(values)


def get_min_in_list(values):
    if len(values) == 0:
        return 0

    return np.min(values)


def get_sum_in_list(values):
    if len(values) == 0:
        return 0

    return np.sum(values)


def get_mean_in_list(values):
    if len(values) == 0:
        return 0

    return np.mean(values)


def get_std_in_list(values):
    if len(values) == 0:
        return -1

    return np.std(values)


def get_statistics(values):
    statistic_values = list()

    statistic_values.append(get_max_in_list(values))
    statistic_values.append(get_min_in_list(values))
    statistic_values.append(get_sum_in_list(values))
    statistic_values.append(get_mean_in_list(values))
    statistic_values.append(get_std_in_list(values))

    return statistic_values