import os

import numpy as np
import plotly
import plotly.graph_objects as go
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from statsmodels.tsa.stattools import adfuller
from torch.utils.data import DataLoader

from util.constants import NUMBER_OF_CHANNELS
from util.constants import SUBJECT_FILE_PREFIX, SUBJECT_FILE_EXTENSION, \
    SUBJECT_FILE_CHANNEL
from reader.file_reader import read_chunk_from_binary_file
from recurrentGraphWavenet.ml.dataset import GraphWavenetDataset
from util.util import get_string_from_number, save_dictionary_to_file


def create_loader_window(dots_folder_path, subject_number, trial_index, window_index, trial_division_file_path,
                         input_length, output_length, batch_size, shuffle, output_path,
                         initial_train_percentage, increase_train_percentage, include_cross, plot_values=True):
    subject_directory = os.path.join(dots_folder_path, SUBJECT_FILE_PREFIX + get_string_from_number(subject_number))

    trial_division_file_path = os.path.join(trial_division_file_path, 'Window')
    trial_division_file = open(os.path.join(trial_division_file_path, f'{subject_number}_{trial_index}.txt'), 'r')
    trial_division_file_content = trial_division_file.readlines()

    window_start_timestamp = int(trial_division_file_content[window_index].split()[0])
    window_end_timestamp = int(trial_division_file_content[window_index].split()[1])

    window_length = window_end_timestamp - window_start_timestamp + 1

    channel_values = []
    for channel_index in range(1, NUMBER_OF_CHANNELS + 1):
        channel_file_path = SUBJECT_FILE_PREFIX + get_string_from_number(subject_number) + \
                            SUBJECT_FILE_CHANNEL + get_string_from_number(channel_index) + SUBJECT_FILE_EXTENSION
        channel_file_path = os.path.join(subject_directory, channel_file_path)
        channel_values.append(
            read_chunk_from_binary_file(channel_file_path, window_start_timestamp, window_length, 4, 'f'))

    channel_values = np.array(channel_values)

    if plot_values:
        plot_channel_values(channel_values, output_path, subject_number)

    loader_splits = []

    if include_cross:

        while initial_train_percentage < 100:

            train_len = int(len(channel_values[0]) * initial_train_percentage / 100)

            train_array = []
            cross_array = []

            for channel in range(NUMBER_OF_CHANNELS):
                train_array.append(channel_values[channel][:train_len])
                cross_array.append(channel_values[channel][train_len:])

            train_array = np.array(train_array)
            cross_array = np.array(cross_array)

            mean = train_array.mean()
            std = train_array.std()

            train_array = train_array - mean
            train_array = train_array / std

            cross_array = cross_array - mean
            cross_array = cross_array / std

            train_loader = create_loader_for_array(train_array, input_length, output_length, batch_size, shuffle)
            cross_loader = create_loader_for_array(cross_array, input_length, output_length, 1, False)

            loader_splits.append((train_loader, cross_loader, mean, std, train_array, cross_array))

            initial_train_percentage += increase_train_percentage
    else:
        mean = channel_values.mean()
        std = channel_values.std()

        channel_values = channel_values - mean
        channel_values = channel_values / std

        loader = create_loader_for_array(channel_values, input_length, output_length, batch_size, shuffle)
        loader_splits.append((loader, channel_values, mean, std))

    return loader_splits


def create_loader_trial(dots_folder_path, subject_number, trial_index, trial_division_file_path,
                        input_length, output_length, batch_size, shuffle, output_path,
                        initial_train_percentage, increase_train_percentage, include_cross, plot_values=True):
    subject_directory = os.path.join(dots_folder_path, SUBJECT_FILE_PREFIX + get_string_from_number(subject_number))

    trial_division_file_path = os.path.join(trial_division_file_path, 'Trial')

    trial_division_file = open(os.path.join(trial_division_file_path, f'{subject_number}_{trial_index}.txt'), 'r')
    trial_division_file_content = trial_division_file.readlines()

    channel_values = []
    if len(trial_division_file_content) == 1:

        window_start_timestamp = int(trial_division_file_content[0].split()[0])
        window_end_timestamp = int(trial_division_file_content[0].split()[1])

        window_length = window_end_timestamp - window_start_timestamp + 1

        for channel_index in range(1, NUMBER_OF_CHANNELS + 1):
            channel_file_path = SUBJECT_FILE_PREFIX + get_string_from_number(subject_number) + \
                                SUBJECT_FILE_CHANNEL + get_string_from_number(channel_index) + SUBJECT_FILE_EXTENSION
            channel_file_path = os.path.join(subject_directory, channel_file_path)
            channel_values.append(
                read_chunk_from_binary_file(channel_file_path, window_start_timestamp, window_length, 4, 'f'))

        channel_values = np.array(channel_values)

    else:
        first_start_timestamp = int(trial_division_file_content[0].split()[0])
        first_end_timestamp = int(trial_division_file_content[0].split()[1])

        second_start_timestamp = int(trial_division_file_content[1].split()[0])
        second_end_timestamp = int(trial_division_file_content[1].split()[1])

        first_length = first_end_timestamp - first_start_timestamp + 1
        second_length = second_end_timestamp - second_start_timestamp + 1

        for channel_index in range(1, NUMBER_OF_CHANNELS + 1):
            channel_file_path = SUBJECT_FILE_PREFIX + get_string_from_number(subject_number) + \
                                SUBJECT_FILE_CHANNEL + get_string_from_number(channel_index) + SUBJECT_FILE_EXTENSION
            channel_file_path = os.path.join(subject_directory, channel_file_path)
            channel_values.append(
                read_chunk_from_binary_file(channel_file_path, first_start_timestamp, first_length, 4, 'f'))
            channel_values[-1].extend(
                read_chunk_from_binary_file(channel_file_path, second_start_timestamp, second_length, 4, 'f')
            )

        channel_values = np.array(channel_values)

    if plot_values:
        plot_channel_values(channel_values, output_path, subject_number)

    loader_splits = []

    if include_cross:
        while initial_train_percentage < 100:

            train_len = int(len(channel_values[0]) * initial_train_percentage / 100)

            train_array = []
            cross_array = []

            for channel in range(NUMBER_OF_CHANNELS):
                train_array.append(channel_values[channel][:train_len])
                cross_array.append(channel_values[channel][train_len:])

            train_array = np.array(train_array)
            cross_array = np.array(cross_array)

            mean = train_array.mean()
            std = train_array.std()

            train_array = train_array - mean
            train_array = train_array / std

            cross_array = cross_array - mean
            cross_array = cross_array / std

            train_loader = create_loader_for_array(train_array, input_length, output_length, batch_size, shuffle)
            cross_loader = create_loader_for_array(cross_array, input_length, output_length, 1, False)

            loader_splits.append((train_loader, cross_loader, mean, std, train_array, cross_array))

            initial_train_percentage += increase_train_percentage

    else:
        mean = channel_values.mean()
        std = channel_values.std()

        channel_values = channel_values - mean
        channel_values = channel_values / std

        loader = create_loader_for_array(channel_values, input_length, output_length, batch_size, shuffle)
        loader_splits.append((loader, channel_values, mean, std))

    return loader_splits


def plot_channel_values(values, output_directory, subject_number):
    figure = go.Figure()

    for channel in range(NUMBER_OF_CHANNELS):
        figure.add_trace(
            go.Scatter(
                x=[i for i in range(len(values[channel]))],
                y=values[channel],
                mode='lines',
                name=f'Channel {channel}'
            )
        )

    plotly.offline.plot(figure, filename=os.path.join(output_directory, 'ChannelValues.html'), auto_open=False)

    check_for_stationarity(values, output_directory, subject_number)


def check_for_stationarity(values, output_directory, subject_number):
    dickey_fuller_log = open(os.path.join(output_directory, 'dickey_fuller.txt'), 'w+')

    for channel in range(NUMBER_OF_CHANNELS):
        if subject_number == 1 and channel == 6:
            continue
        print(f'Channel {channel}', file=dickey_fuller_log)
        result = adfuller(values[channel])
        print('\t ADF Statistic: %f' % result[0], file=dickey_fuller_log)
        print('\t p-value: %f' % result[1], file=dickey_fuller_log)
        print('\t Critical Values:', file=dickey_fuller_log)
        for key, value in result[4].items():
            print('\t\t%s: %.3f' % (key, value), file=dickey_fuller_log)

    dickey_fuller_log.close()


def create_loader_for_array(values, input_length, output_length, batch_size, shuffle):
    sequence_start = 0
    sequence_end = len(values[0]) - 1

    train_x = []
    train_y = []

    while (sequence_start + input_length + output_length - 1) <= sequence_end:
        train_x.append([])
        train_y.append([])

        for channel_index in range(NUMBER_OF_CHANNELS):
            train_x[-1].append(values[channel_index][sequence_start:(sequence_start + input_length)])
            train_y[-1].append(values[channel_index][
                               sequence_start + input_length:sequence_start + input_length + output_length])

        sequence_start += 1

    train_x = np.array(train_x)
    train_y = np.array(train_y)

    dataset = GraphWavenetDataset(train_x, train_y)
    dataset_loader = DataLoader(dataset, batch_size, shuffle)

    return dataset_loader


def save_running_parameters(batch_size, blocks, dots_folder_path, increase_train_percentage, initial_train_percentage,
                            input_length, layers, loader_option, number_of_epochs, output_length, output_path,
                            subject_number, trial_division_file_path, trial_index, window_index,
                            arguments_file_path, use_functional_network,
                            functional_network_path,
                            use_previous_weight_matrix, previous_weight_matrix_path, include_cross):
    arguments_dict = {}
    arguments_dict['dots_folder_path'] = dots_folder_path
    arguments_dict['trial_division_file_path'] = trial_division_file_path
    arguments_dict['output_path'] = output_path
    arguments_dict['subject_number'] = subject_number
    arguments_dict['trial_index'] = trial_index
    arguments_dict['window_index'] = window_index
    arguments_dict['input_length'] = input_length
    arguments_dict['output_length'] = output_length
    arguments_dict['batch_size'] = batch_size
    arguments_dict['loader_option'] = loader_option
    arguments_dict['blocks'] = blocks
    arguments_dict['layers'] = layers
    arguments_dict['number_of_epochs'] = number_of_epochs
    arguments_dict['initial_train_percentage'] = initial_train_percentage
    arguments_dict['increase_train_percentage'] = increase_train_percentage
    arguments_dict['use_functional_network'] = use_functional_network
    arguments_dict['functional_network_path'] = functional_network_path
    arguments_dict['use_previous_weight_matrix'] = use_previous_weight_matrix
    arguments_dict['previous_weight_matrix_path'] = previous_weight_matrix_path
    arguments_dict['include_cross'] = include_cross

    save_dictionary_to_file(arguments_dict, arguments_file_path)


def load_functional_network(functional_network_path, subject_number, trial_index, func_k=5):
    functional_network_path = os.path.join(functional_network_path, get_string_from_number(subject_number))
    functional_network_path = os.path.join(functional_network_path, f'wMatrix_K_trial{trial_index}')

    file = open(functional_network_path, 'r')
    content = file.readlines()

    start_line = int(func_k / 0.25 * (NUMBER_OF_CHANNELS + 1) + 1)

    mat = []
    for i in range(NUMBER_OF_CHANNELS):
        mat.append([])
        for j in range(NUMBER_OF_CHANNELS):
            mat[-1].append(float(content[start_line + i].split()[j]))
            if mat[-1][-1] == -1e+8:
                mat[-1][-1] = 0.0

    mat = np.array(mat)
    return [torch.tensor(asym_adj(mat)).double(), torch.tensor(asym_adj(np.transpose(mat))).double()]


def asym_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat = sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()


def load_previous_weight_matrix(previous_weight_matrix_path, loader_option, trial_index, window_index):
    if loader_option == "Trial":
        previous_weight_matrix_path = os.path.join(previous_weight_matrix_path, "Trial")
        previous_weight_matrix_path = os.path.join(previous_weight_matrix_path, str(trial_index - 1))
        previous_weight_matrix_path = os.path.join(previous_weight_matrix_path, "WeightMatrix.dat")

    else:
        if window_index != 0:
            previous_weight_matrix_path = os.path.join(previous_weight_matrix_path, "Window")
            previous_weight_matrix_path = os.path.join(previous_weight_matrix_path, str(trial_index))
            previous_weight_matrix_path = os.path.join(previous_weight_matrix_path, str(window_index - 1))
            previous_weight_matrix_path = os.path.join(previous_weight_matrix_path, "WeightMatrix.dat")
        else:
            previous_weight_matrix_path = os.path.join(previous_weight_matrix_path, "Window")
            previous_weight_matrix_path = os.path.join(previous_weight_matrix_path, str(trial_index - 1))

            folder_list = next(os.walk(previous_weight_matrix_path))[1]
            folder_list = [int(k) for k in folder_list]
            previous_weight_matrix_path = os.path.join(previous_weight_matrix_path, str(max(folder_list)))
            previous_weight_matrix_path = os.path.join(previous_weight_matrix_path, "WeightMatrix.dat")

    weight_matrix = np.fromfile(previous_weight_matrix_path, dtype=float)
    weight_matrix = weight_matrix.reshape(NUMBER_OF_CHANNELS, NUMBER_OF_CHANNELS)

    weight_matrix = torch.DoubleTensor(weight_matrix)
    weight_matrix_transpose = weight_matrix.transpose(0, 1)

    weight_matrix = F.softmax(weight_matrix, dim=1)
    weight_matrix_transpose = F.softmax(weight_matrix_transpose, dim=1)

    return [torch.tensor(asym_adj(weight_matrix.detach().numpy())).double(),
            torch.tensor(asym_adj(weight_matrix_transpose.detach().numpy())).double()]
