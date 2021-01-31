import datetime
import os
import platform

import torch

from recurrentGraphWavenet.ml.train_engine import TrainEngine
from recurrentGraphWavenet.recurrent_graph_wavenet_util import create_loader_window, create_loader_trial, \
    save_running_parameters, load_functional_network, load_previous_weight_matrix
from util.constants import NUMBER_OF_CHANNELS
from util.util import log


def recurrent_graph_wavenet(dots_folder_path, trial_division_file_path, output_path, subject_number, trial_index,
                            window_index, input_length, output_length, batch_size, loader_option, widget, blocks,
                            layers, number_of_epochs, initial_train_percentage, increase_train_percentage,
                            use_functional_network, functional_network_path, use_previous_weight_matrix,
                            previous_weight_matrix_path, include_cross, use_gpu, is_experiment = True):
    # set device
    if use_gpu:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    # set number of pytorch threads
    torch.set_num_threads(int(os.cpu_count() * 0.75))

    # set the highest priority to the process (if unix)
    if platform.uname().system == 'Linux':
        os.nice(-40)

    output_path = os.path.join(output_path, 'GraphWavenet')

    if is_experiment:
        if not os.path.exists(output_path):
            os.makedirs(output_path)
            output_path = os.path.join(output_path, 'Run 1')
            os.makedirs(output_path)
        else:
            folders = next(os.walk(output_path))[1]
            folder_numbers = [int(x.split()[1]) for x in folders]
            new_number = max(folder_numbers) + 1
            output_path = os.path.join(output_path, f'Run {new_number}')
            os.makedirs(output_path)
    else:
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        if loader_option == 'Window':
            output_path = os.path.join(output_path, f'{trial_index}_{window_index}')
        else:
            output_path = os.path.join(output_path, f'{trial_index}')
        os.makedirs(output_path)

    arguments_file_path = os.path.join(output_path, 'arguments.json')
    save_running_parameters(batch_size, blocks, dots_folder_path, increase_train_percentage, initial_train_percentage,
                            input_length, layers, loader_option, number_of_epochs, output_length, output_path,
                            subject_number, trial_division_file_path, trial_index, window_index,
                            arguments_file_path, use_functional_network,
                            functional_network_path,
                            use_previous_weight_matrix, previous_weight_matrix_path, include_cross)

    log_file = os.path.join(output_path, 'log.txt')
    log(f'Graph wavenet start: {str(datetime.datetime.now())}', log_file, widget)

    supports = None
    if use_functional_network:
        supports = load_functional_network(functional_network_path, subject_number, trial_index)

    if use_previous_weight_matrix:
        if supports is None:
            supports = []
        supports.extend(
            load_previous_weight_matrix(previous_weight_matrix_path, loader_option, trial_index, window_index))

    if supports is not None:
        supports = [x.to(device) for x in supports]

    loader_splits = None

    if loader_option == 'Window':
        loader_splits = create_loader_window(
            dots_folder_path = dots_folder_path,
            subject_number = subject_number,
            trial_index = trial_index,
            window_index = window_index,
            input_length = input_length,
            output_length = output_length,
            batch_size = batch_size,
            shuffle = True,
            trial_division_file_path = trial_division_file_path,
            output_path = output_path,
            initial_train_percentage = initial_train_percentage,
            increase_train_percentage = increase_train_percentage,
            include_cross = include_cross
        )

    if loader_option == 'Trial':
        loader_splits = create_loader_trial(
            dots_folder_path = dots_folder_path,
            subject_number = subject_number,
            trial_index = trial_index,
            input_length = input_length,
            output_length = output_length,
            batch_size = batch_size,
            shuffle = True,
            trial_division_file_path = trial_division_file_path,
            output_path = output_path,
            initial_train_percentage = initial_train_percentage,
            increase_train_percentage = increase_train_percentage,
            include_cross = include_cross
        )

    train_engine = TrainEngine(
        number_of_nodes = NUMBER_OF_CHANNELS,
        blocks = blocks,
        layers = layers,
        loader_splits = loader_splits,
        log_file = log_file,
        widget = widget,
        output_directory = output_path,
        number_of_epochs = number_of_epochs,
        use_previous_model = False,
        input_length = input_length,
        output_length = output_length,
        supports = supports,
        device = device
    )

    if include_cross:
        train_engine.train()
    else:
        train_engine.full_train()

        matrix_path = previous_weight_matrix_path
        if loader_option == 'Trial':
            matrix_path = os.path.join(matrix_path, 'Trial', f'{trial_index}')
            if not os.path.exists(matrix_path):
                os.makedirs(matrix_path)

        else:
            matrix_path = os.path.join(matrix_path, 'Window', f'{trial_index}')
            if not os.path.exists(matrix_path):
                os.makedirs(matrix_path)
            matrix_path = os.path.join(matrix_path, f'{window_index}')
            if not os.path.exists(matrix_path):
                os.makedirs(matrix_path)

        train_engine.save_weight_matrix(matrix_path)

    log(f'Graph wavenet end: {str(datetime.datetime.now())}', log_file, widget)
