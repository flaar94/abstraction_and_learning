import math
import numpy as np
import itertools

from matplotlib import colors, pyplot as plt


def ordinal(n):
    """Obtained from Stack overflow via user Gareth: https://codegolf.stackexchange.com/users/737/gareth"""
    return "%d%s" % (n, "tsnrhtdd"[(math.floor(n / 10) % 10 != 1) * (n % 10 < 4) * n % 10::4])


def flattener(pred):
    """Function provided by the challenge"""
    str_pred = str([row for row in pred])
    str_pred = str_pred.replace(', ', '')
    str_pred = str_pred.replace('[[', '|')
    str_pred = str_pred.replace('][', '|')
    str_pred = str_pred.replace(']]', '|')
    return str_pred


def multi_array_flattener(arrays):
    string_list = [flattener(array) for array in arrays]
    return " ".join(string_list)


def combine_sorted_queues(queues, max_nll):
    """First queue can be a generator"""
    if len(queues) == 2:
        first_queue, other_queue = queues
        for y in other_queue:
            for x in first_queue:
                if x.nll + y.nll > max_nll:
                    break
                else:
                    yield x, y
    elif len(queues) > 2:
        first_queue, *other_queues = queues
        for ys in combine_sorted_queues(other_queues, max_nll):
            for x in first_queue:
                if x.nll + sum((y.nll for y in ys)) > max_nll:
                    break
                else:
                    yield (x,) + ys


def plot_task(task, num=0):
    """
    Plots the first train and test pairs of a specified task,
    using same color scheme as the ARC app
    """
    cmap = colors.ListedColormap(
        ['#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
         '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25', '#ffffff'])
    norm = colors.Normalize(vmin=0, vmax=10)
    train_len = len(task['train'])
    fig, axs = plt.subplots(1, 2 * train_len, figsize=(15, 15))
    # print(len(task['train']))
    for i in range(train_len):
        axs[0 + 2 * i].imshow(task['train'][i]['input'], cmap=cmap, norm=norm)
        axs[0 + 2 * i].axis('off')
        axs[0 + 2 * i].set_title(f'Train Input {num}')
        axs[1 + 2 * i].imshow(task['train'][i]['output'], cmap=cmap, norm=norm)
        axs[1 + 2 * i].axis('off')
        axs[1 + 2 * i].set_title(f'Train Output {num}')
    plt.tight_layout()
    plt.show()


def to_tuple(lst):
    out = tuple((tuple((int(entry) for entry in row)) for row in lst))
    return out


def display_case(grid, title=''):
    if len(grid) == 0:
        return
    cmap = colors.ListedColormap(
        ['#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
         '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25', '#ffffff'])
    norm = colors.Normalize(vmin=0, vmax=10)
    fig, axs = plt.subplots(1, figsize=(12, 4))
    axs.imshow(grid, cmap=cmap, norm=norm)
    if title:
        plt.title(title)
    # plt.tight_layout()
    plt.show()


def add_one(properties):
    for prop in properties:
        prop.nll += 1


def list2d_to_set(lst):
    return {x for sublist in lst for x in sublist}


def filter_unlikelies(properties, max_nll):
    return [prop for prop in properties if prop.nll <= max_nll]


def extend_grid_via_extreme_coordinates(min_coords, max_coords, grid_arr):
    if min_coords == (0, 0) and all(
            max_coord + 1 == grid_length for max_coord, grid_length in zip(max_coords, grid_arr.shape)):
        return grid_arr
    shape = tuple((max_coord + 1 - min_coord for min_coord, max_coord in zip(min_coords, max_coords)))
    new_grid_arr = np.zeros(shape, dtype=grid_arr.dtype)
    for i, j in itertools.product(range(grid_arr.shape[0]), range(grid_arr.shape[1])):
        new_grid_arr[i + min_coords[0], j + min_coords[1]] = grid_arr[i, j]
    return new_grid_arr


def extend_entities_positions(min_coords, entities_positions):
    adjustments = [min(min_coord, 0) for min_coord in min_coords]
    new_entities_positions = []
    for entity_positions in entities_positions:
        new_entity_positions = {}
        for position, color in entity_positions.items():
            new_position = tuple((coord - adjustment for coord, adjustment in zip(position, adjustments)))
            new_entity_positions[new_position] = color
        new_entities_positions.append(new_entity_positions)
    return new_entities_positions


def stamp_entities_positions(entities_positions, grid_arr):
    for entity_positions in entities_positions:
        for position, color in entity_positions.items():
            grid_arr[position[0], position[1]] = color
    return grid_arr


def tuplefy_task(task, test=False):
    if test:
        tuplefied_task = {'train': [{'input': to_tuple(case['input']), 'output': to_tuple(case['output'])} for case in
                                    task['train']],
                          'test': [{'input': to_tuple(case['input'])} for case in
                                   task['test']]}
    else:
        tuplefied_task = {'train': [{'input': to_tuple(case['input']), 'output': to_tuple(case['output'])} for case in
                                    task['train']],
                          'test': [{'input': to_tuple(case['input']), 'output': to_tuple(case['output'])} for case in
                                   task['test']]}
    return tuplefied_task


class WindowsInhibitor:
    """Prevent OS sleep/hibernate in windows; code from:
    https://github.com/h3llrais3r/Deluge-PreventSuspendPlus/blob/master/preventsuspendplus/core.py
    API documentation:
    https://msdn.microsoft.com/en-us/library/windows/desktop/aa373208(v=vs.85).aspx"""
    ES_CONTINUOUS = 0x80000000
    ES_SYSTEM_REQUIRED = 0x00000001

    def __init__(self):
        pass

    def __enter__(self):
        self.inhibit()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.uninhibit()

    @staticmethod
    def inhibit():
        import ctypes
        print("Preventing Windows from going to sleep")
        ctypes.windll.kernel32.SetThreadExecutionState(
            WindowsInhibitor.ES_CONTINUOUS |
            WindowsInhibitor.ES_SYSTEM_REQUIRED)

    @staticmethod
    def uninhibit():
        import ctypes
        print("Allowing Windows to go to sleep")
        ctypes.windll.kernel32.SetThreadExecutionState(
            WindowsInhibitor.ES_CONTINUOUS)
