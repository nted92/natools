import warnings

import sys
import operator
import types

from collections import Counter
from itertools import groupby
from scipy.spatial.distance import cityblock
import numpy as np

from sklearn.cluster import KMeans

import natools.time as time_utils

from random import randint

from fuzzywuzzy import fuzz


def check_if_numpy_array(py_object):
    """
    :param object: some Python object 
    :return: whether it is of type numpy.array or not (boolean)
    """
    return type(py_object).__module__ == np.__name__


def get_type_mapping_from_list(py_object):
    if isinstance(py_object, str):
        return lambda x: "".join(x)
    elif isinstance(py_object, list):
        return list
    elif isinstance(py_object, tuple):
        return tuple
    elif isinstance(py_object, types.GeneratorType):
        return lambda x: (n for n in x)
    elif check_if_numpy_array(py_object):
        return lambda x: np.array(x)
    else:
        return list


def count_occurrences_in_iterator(iterator):
    """
    :param iterator: list, 1D numpy.array, tuple, string, generator, etc. 
    :return: dict_keys object containing (element, occurences) pairs
    """
    return dict(Counter(iterator).items())


def get_iterator_most_occurring_element(iterator):
    """
    :param iterator: list, 1D numpy.array, tuple, string, generator, etc. 
    :return: most occurring element's value
    """
    occurences = count_occurrences_in_iterator(iterator)
    return max(occurences, key=operator.itemgetter(1))[0]


def get_iterator_most_occurring_element_count(iterator):
    """
    :param iterator: list, 1D numpy.array, tuple, string, generator, etc. 
    :return: most occurring element's count
    """
    occurences = count_occurrences_in_iterator(iterator)
    return max(occurences, key=operator.itemgetter(1))[1]


def get_iterator_without_consecutive_repetitions(iterator):
    """
    :param iterator: list, 1D numpy.array, tuple, string, generator, etc. 
    :return: same type as iterator, after the removal of the consecutive 
    repetitions
    
    E.g. [-1, -1, -1, 0.5, 0.55, 0.55, -1, -1] becomes [-1, 0.5, 0.55, -1]
    """
    new_sequence = [k for k, g in groupby(iterator)]
    return get_type_mapping_from_list(iterator)(new_sequence)


def get_intersection_of_two_lists(list_a, list_b):
    """
    :param list_a: list object 
    :param list_b: list object
    :return: list
    """
    return list(set(list_a) & set(list_b))


def get_union_of_two_lists(list_a, list_b):
    """
    :param list_a: list object 
    :param list_b: list object
    :return: list
    """
    return list(set(list_a) | set(list_b))


def apply_sliding_window_on_iterator(iterator, window_size):
    """
    :param iterator: list, 1D numpy.array, tuple, string, generator, etc. 
    :param window_size: amount of consecutive elements
    :return: list of all the consecutive sub-sequences of size window_size, 
    each of the same type as the iterator input
    """
    if window_size < 1:
        return []

    sequence = list(iterator)
    sequence_length = len(sequence)
    if sequence_length < window_size:
        return []

    sub_sequences = []
    type_mapping = get_type_mapping_from_list(iterator)
    for i in range(sequence_length - window_size + 1):
        sub_sequences.append(type_mapping(sequence[i:window_size + i]))
    return sub_sequences


def extract_sub_iterators_from_iterator(iterator, num_sub_iterators,
                                        minimum_length, maximum_length,
                                        undesired, avoid_duplicates,
                                        avoid_value):
    """
    :param iterator: list, 1D numpy.array, tuple, string
    :param num_sub_iterators: desired number of sub_iterators to extract. Acts
    as an upper bound (can't always manage to extract exactly up to this amount).
    :param minimum_length: minimum size of the sub_iterators to extract
    :param maximum_length: maximum size of the sub_iterators to extract
    :param undesired: list of specific iterators (by value) we don't want to
    extract
    :param avoid_duplicates: boolean, whether to accept duplicates in the
    extracted sub iterators
    :param avoid_value: special condition, in the form of a dictionary whose key
    is a value that can appear in the sequence, and the associated value is
    the number of occurrences of different values we want to see appearing at least 
    :return: list of the extracted sub-iterators 
    """
    sequence = list(iterator)
    type_mapping = get_type_mapping_from_list(iterator)
    # Is the compromise on speed worth supporting several types of iterators?

    default_response = type_mapping([])

    if minimum_length > maximum_length:
        warnings.warn("Bad input parameters: minimum_length > maximum_length")
        return default_response

    if (maximum_length <= 0) or (num_sub_iterators <= 0):
        warnings.warn("Bad input parameters: (maximum_length <= 0) or (num_sub_iterators <= 0)")
        return default_response

    iterator_length = len(sequence)
    if iterator_length < minimum_length:
        warnings.warn("Bad input parameters: iterator_length < minimum_length")
        return default_response

    if iterator_length < maximum_length:
        maximum_length = iterator_length

    if (len(avoid_value) > 1) or ((len(avoid_value) == 1) and (avoid_value[list(avoid_value.keys())[0]] <= 0)):
        avoid_value = dict()

    maximum_num_sub_iterators = int(0.5 * (maximum_length - minimum_length + 1) * (2 * iterator_length + 2 - minimum_length - maximum_length))

    if maximum_num_sub_iterators <= num_sub_iterators:
        # calculate all the possible sub-iterators
        all_possible_sub_iterators = []
        for length in range(minimum_length, maximum_length + 1):
            for i in range(iterator_length - length + 1):
                seq = sequence[i:length + i]
                if (seq not in undesired) and ((not avoid_duplicates) or (avoid_duplicates and (seq not in all_possible_sub_iterators))) and ((not bool(avoid_value)) or (bool(avoid_value) and (len([elt for elt in seq if elt != list(avoid_value.keys())[0]]) >= avoid_value[list(avoid_value.keys())[0]]))):
                    all_possible_sub_iterators.append(seq)

        return [type_mapping(elt) for elt in all_possible_sub_iterators]
    else:
        # proceed to random selection
        randomly_selected_sub_iterators = []
        already_selected_pairs = [] # cannot use sets (for faster lookup) because list of iterators (non hashable type)
        for i in range(num_sub_iterators):
            random_selection = True
            candidate = None
            while random_selection and len(already_selected_pairs) < maximum_num_sub_iterators:
                # randomly select length
                a = randint(minimum_length, maximum_length)
                # randomly select the start index
                b = randint(0, iterator_length - a)
                candidate = sequence[b:b + a]
                if (a, b) not in already_selected_pairs:
                    already_selected_pairs.append((a, b))
                    if (candidate not in undesired) and ((not avoid_duplicates) or (avoid_duplicates and (candidate not in randomly_selected_sub_iterators))) and ((not bool(avoid_value)) or (bool(avoid_value) and (len([elt for elt in candidate if elt != list(avoid_value.keys())[0]]) >= avoid_value[list(avoid_value.keys())[0]]))):
                        random_selection = False
                    else:
                        candidate = None
                else:
                    candidate = None
            if candidate is not None:
                # there shouldn't be duplicates at this point (not checking again for computational speed considerations)
                randomly_selected_sub_iterators.append(candidate)

        return [type_mapping(elt) for elt in randomly_selected_sub_iterators]


def get_data_sampling_rate(data, n_clusters=4, timestamp_field='timestamp'):
    """
    :param data: list of dictionaries containing at least a timestamp related key
    :param n_clusters: number of time_deltas clusters to create
    :return: dictionary of lists of time_deltas, organized in clusters whose ids are sorted in increasing cluster mean time_delta value
    
    Note: for Neura sensor data, n_clusters=4 is a good starting point to account 
    for the holes in the data of different kinds: normal operation, small offlines, 
    big offlines, long uninstallings/reinstallings periods
    """
    data = sorted(data, key=lambda x: x[timestamp_field], reverse=False)

    time_deltas = []
    for i in range(len(data) - 1):
        time_deltas.append(abs(data[i + 1][timestamp_field] - data[i][timestamp_field]))
    time_deltas = np.array(time_deltas).reshape(-1, 1)

    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(time_deltas)
    time_deltas_grouping_temp = dict((ind, []) for ind in range(n_clusters))
    for i in range(time_deltas.shape[0]):
        time_deltas_grouping_temp[kmeans.labels_[i]].append(time_deltas[i][0])

    time_deltas_grouping = dict((ind, []) for ind in range(n_clusters))
    group_id_mapping = [i[0] for i in sorted(enumerate([np.mean(time_deltas_grouping_temp[ind]) for ind in range(n_clusters)]), key=lambda x: x[1])]
    for i in range(n_clusters):
        time_deltas_grouping[i] = time_deltas_grouping_temp[group_id_mapping[i]]

    return time_deltas_grouping


def get_quantities_at_resolution(quantities, start, end, bin_resolution, aggregation_function, default_value=0, timestamp_field='timestamp', quantity_field='quantity'):
    """
    :param quantities: time series of a numbered quantity (list of dictionaries with at least the time and quantity keys, ordered by time)
    :param start: when to start (timestamp, integer, in seconds)
    :param end: when to stop (timestamp, integer, in seconds)
    :param bin_resolution: size of the bins on which the aggregation takes place (integer, in seconds)
    :param aggregation_function: takes the list of a bin's values as an input, produces an aggregation (e.g. sums the values)
    :param default_value: place holder value for bins that are not represented in the initial data
    :param timestamp_field: field of the times in the dictionaries
    :param quantity_field: field of the quantity in the dictionaries
    :return: list, with the aggregated value per bin
    """
    quantities_at_resolution = []
    time_range = time_utils.get_time_range_in_bins(start, end, bin_resolution)
    if len(quantities) == 0:
        quantities_at_resolution = [default_value] * (len(time_range) - 1)
    else:
        quantities_index = 0
        i = 0
        while i < len(time_range) - 1:
            low, up = time_range[i], time_range[i+1]
            stop = False
            aggregation = []
            while not stop:
                if quantities_index >= len(quantities):
                    stop = True
                else:
                    step = quantities[quantities_index]
                    if low <= step[timestamp_field] < up:
                        aggregation.append(step[quantity_field])
                        quantities_index += 1
                    else:
                        stop = True
            if len(aggregation) == 0:
                aggregation = default_value
            else:
                aggregation = aggregation_function(aggregation)
            quantities_at_resolution.append(aggregation)
            i += 1
        if quantities[-1][timestamp_field] == end:
            quantities_at_resolution[-1] += quantities[-1][quantity_field]
    return quantities_at_resolution


def compute_manhattan_distance(sequence_a, sequence_b):
    """
    :param sequence_a: first sequence (must be of same length as the second 
    sequence), list or numpy array
    :param sequence_b: second sequence (must be of same length as the first 
    sequence), list or numpy array
    :return: a number, the Manhattan distance, sum(abs(a_i-b_i) for i=0..N)
    """
    return cityblock(sequence_a, sequence_b)


def compute_euclidean_distance(sequence_a, sequence_b):
    """
    :param sequence_a: first sequence (must be of same length as the second 
    sequence), list or numpy array
    :param sequence_b: second sequence (must be of same length as the first 
    sequence), list or numpy array
    :return: a number, the Euclidean distance, sqrt(sum((a_i-b_i)**2 for 
    i=0..N))
    """
    return np.linalg.norm(np.array(sequence_a) - np.array(sequence_b))


def scale_number(number, min_a, max_a, min_b=0, max_b=1, limit_overflow=True):
    """
    :param number: number to scale
    :param min_a: lower bound of the interval
    :param max_a: upper bound of the interval
    :param min_b: lower bound of the destination interval (optional)
    :param max_b: upper bound of the destination interval (optional)
    :param limit_overflow: if number out of a's interval bounds, project it on the lower or upper bound
    :return: scaled number
    """
    scaled_number = (number - min_a) / (max_a - min_a)
    if limit_overflow:
        if scaled_number < 0:
            scaled_number = 0
        elif scaled_number > 1:
            scaled_number = 1
    return min_b + (max_b - min_b) * scaled_number


def check_for_special_characters(string):
    """
    :param string: str
    :return: boolean, checks if any special character
    """
    return any(ord(char) > 126 for char in string)


def string_matching(string_1, string_2, ratio=True, partial_ratio=False,
                    token_sort_ratio=False, token_set_ratio=False,
                    highest_ratio_augmentation=False):
    """
    :param string_1: first string
    :param string_2: second string
    :param ratio: fuzzywuzzy method
    :param partial_ratio: fuzzywuzzy method
    :param token_sort_ratio: fuzzywuzzy method
    :param token_set_ratio: fuzzywuzzy method
    :param highest_ratio_augmentation: increase the weight of the highest ratio
    :return: score, between 0 and 100
    
    It is letter case independent.
    
    More info: https://github.com/seatgeek/fuzzywuzzy
    """
    scores = list()

    if ratio:
        scores.append(fuzz.ratio(string_1.lower(), string_2.lower()))
    if partial_ratio:
        scores.append(fuzz.partial_ratio(string_1.lower(), string_2.lower()))
    if token_sort_ratio:
        scores.append(fuzz.token_sort_ratio(string_1.lower(), string_2.lower()))
    if token_set_ratio:
        scores.append(fuzz.token_set_ratio(string_1.lower(), string_2.lower()))

    if len(scores) == 0:
        return None
    else:
        if highest_ratio_augmentation:
            most_significant_score = scores.index(max(scores))
            scores += [scores[most_significant_score]] * 3
        return np.mean(scores)


def create_string_block(*args, sep=' '):
    """
    :param args: strings to be added (as part of the same 'block', could be 
    a line or something else depending on the chosen separator)
    :param sep: separator to use between each string argument
    :return: the string 'block', stripped of all whitespaces characters at 
    beginning and at the end
    """
    result = ""
    for arg in args:
        result += str(arg)
        result += sep
    return result.strip()


def add_block_to_logs(logs, *args, sep=' ', append_sep="\n"):
    """
    :param logs: logs string to append to
    :param args: strings to be added to the new block
    :param sep: separator for the block
    :param append_sep: separator between previous logs and new block
    :return: logs with appended block
    """
    if logs is None:
        return None
    else:
        if len(logs) > 0:
            logs += append_sep
        logs += create_string_block(*args, sep)
        return logs


def print_and_add_block_to_logs(logs, *args, sep=' ', append_sep="\n"):
    """
    :param logs: logs string to append to
    :param args: strings to be added to the new block
    :param sep: separator for the block
    :param append_sep: separator between previous logs and new block
    :return: logs with appended block, and prints the new block as well
    
    TODO: try to catch the warnings also (so have an always listening 
    logger...)
    """
    print(create_string_block(*args, sep))
    return add_block_to_logs(logs, *args, sep, append_sep)


def print_in_nice_columns(rows, padding=0):
    """
    :param rows: array of the lines to print, each line being an array of the 
    strings to put in each column
    :param padding: to enlarge space between columns
    :return: None
    """
    col_width = max(len(col) for row in rows for col in row) + padding
    for row in rows:
        print("".join(col.ljust(col_width) for col in row))


class ProgressBar:
    """
    Example usage:
    
        import time

        n = 10
        p = ProgressBar(n)
        count = 0
        for i in range(n):
            time.sleep(0.5)  # delays for 0.5 seconds
            count += 1
            p.animate(count)
            
    NOTE: will get stuck at 0 if n = 1...
    """
    def __init__(self, iterations):
        self.iterations = iterations
        self.prog_bar = '[]'
        self.fill_char = '*'
        self.width = 40
        self.__update_amount(0)
        self.animate = self.animate_ipython

    def animate_ipython(self, iter):
        print('\r', self, end='')
        sys.stdout.flush()
        self.update_iteration(iter + 1)

    def update_iteration(self, elapsed_iter):
        self.__update_amount((elapsed_iter / float(self.iterations)) * 100.0)
        self.prog_bar += '  %d of %s complete' % (elapsed_iter, self.iterations)

    def __update_amount(self, new_amount):
        percent_done = int(round((new_amount / 100.0) * 100.0))
        all_full = self.width - 2
        num_hashes = int(round((percent_done / 100.0) * all_full))
        self.prog_bar = '[' + self.fill_char * num_hashes + ' ' * (all_full - num_hashes) + ']'
        pct_place = (len(self.prog_bar) // 2) - len(str(percent_done))
        pct_string = '%d%%' % percent_done
        self.prog_bar = self.prog_bar[0:pct_place] + \
            (pct_string + self.prog_bar[pct_place + len(pct_string):])

    def __str__(self):
        return str(self.prog_bar)

if __name__ == "__main__":
    quantities = [{"caca": 1, "jour": 2},
                  {"caca": 2, "jour": 3},
                  {"caca": 1, "jour": 5},
                  {"caca": 3, "jour": 8}]

    start, end = 0, 10
    bin_resolution = 3

    aggregation_function = lambda x: sum(x)

    print(get_quantities_at_resolution(quantities, start, end, bin_resolution, aggregation_function, default_value=-1, timestamp_field='jour', quantity_field='caca'))
