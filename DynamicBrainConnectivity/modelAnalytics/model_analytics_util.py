import os

import matplotlib.pyplot as plt
import numpy as np
import plotly
import plotly.graph_objects as go
import seaborn as sns
import torch

from util.constants import DIVISION_LENGTH, STIMULUS_OUTPUT_SIZE, RESPONSE_OUTPUT_SIZE
from util.constants import NUMBER_OF_CHANNELS, COLOR_LIST_CHANNELS
from dualEnsembleClassifier.ml.DualEnsembleClassifierModel import DualEnsembleClassifierModel
from util.util import get_example_length, get_hidden_size, log


def plot_heatmaps(multiple_runs_path, plot_weight_heatmaps, plot_collapsed_weight_heatmaps,
                  plot_collapsed_weight_heatmaps_aligned, output_path, widget):
    # open multiple runs directory
    data_directory = multiple_runs_path

    # create output directory
    heatmap_output_directory = output_path
    heatmap_output_directory = os.path.join(heatmap_output_directory, 'ModelAnalytics')

    if not os.path.exists(heatmap_output_directory):
        os.makedirs(heatmap_output_directory)

    # find configurations
    division_directories = os.listdir(data_directory)

    # for each configuration
    for directory in division_directories:

        # open configuration
        directory = os.path.join(data_directory, directory)

        # get configuration name
        division_name = directory.split('\\')[-1:][0]

        # create configuration heatmap directory
        division_directory_heatmap = os.path.join(heatmap_output_directory, division_name)
        if not os.path.exists(division_directory_heatmap):
            os.makedirs(division_directory_heatmap)

        # compute example lengts
        example_length = get_example_length(DIVISION_LENGTH, int(division_name.split('_')[-2]),
                                            int(division_name.split('_')[-1]))

        # compute stimulus and response hidden size
        stimulus_hidden_size = get_hidden_size(example_length, STIMULUS_OUTPUT_SIZE)
        response_hidden_size = get_hidden_size(example_length, RESPONSE_OUTPUT_SIZE)

        # create a dummy model in which we will load the actual models
        model = DualEnsembleClassifierModel(
            (
                [
                    example_length,
                    stimulus_hidden_size,
                    STIMULUS_OUTPUT_SIZE
                ],
                [
                    example_length,
                    response_hidden_size,
                    RESPONSE_OUTPUT_SIZE
                ]
            ),

            NUMBER_OF_CHANNELS
        )

        # for each individual run
        runs_directories = [x[0] for x in os.walk(os.path.join(data_directory, directory))][1:]

        # find the number of individual runs
        number_of_models = len(runs_directories)

        model_tensor_stimulus = None
        model_tensor_response = None
        first_model = True

        # for each run
        for runs_directory in runs_directories:

            # load model
            model.load_model_from_file(os.path.join(runs_directory, division_name + '_DUAL.model'))

            first_tensor_response = True
            first_tensor_stimulus = True
            tensor_list_stimulus = None
            tensor_list_response = None

            # for each parameter of our model
            for name, param in model.named_parameters():

                # find input matrix for stimulus
                if name.find('0.0.weight') != -1:
                    if first_tensor_stimulus:
                        tensor_list_stimulus = param.data
                        tensor_list_stimulus = tensor_list_stimulus[None, :, :]
                        first_tensor_stimulus = False
                    else:
                        tensor = param.data
                        tensor = tensor[None, :, :]
                        tensor_list_stimulus = torch.cat((tensor_list_stimulus, tensor), 0)

                # find input matrix for response
                elif name.find('1.0.weight') != -1:
                    if first_tensor_response:
                        tensor_list_response = param.data
                        tensor_list_response = tensor_list_response[None, :, :]
                        first_tensor_response = False
                    else:
                        tensor = param.data
                        tensor = tensor[None, :, :]
                        tensor_list_response = torch.cat((tensor_list_response, tensor), 0)

            # if the first model, save the list of tensors (one tensor for each channel)
            if first_model:
                model_tensor_response = tensor_list_response
                model_tensor_stimulus = tensor_list_stimulus
                first_model = False

            # otherwise, add over the previous run
            else:
                model_tensor_response += tensor_list_response
                model_tensor_stimulus += tensor_list_stimulus

            log(f'Finished {runs_directory}', file = None, widget = widget)

        # average
        model_tensor_stimulus = model_tensor_stimulus / number_of_models
        model_tensor_response = model_tensor_response / number_of_models

        # compute std
        std_stimulus = torch.std(model_tensor_stimulus, unbiased = False).numpy().tolist()
        std_response = torch.std(model_tensor_response, unbiased = False).numpy().tolist()

        # compute mean
        mean_stimulus = torch.mean(model_tensor_stimulus).numpy().tolist()
        mean_response = torch.mean(model_tensor_response).numpy().tolist()

        # standardize response input
        response_array = model_tensor_response.numpy()
        response_array = response_array - mean_response
        response_array = response_array / std_response

        # standardize stimulus input
        stimulus_array = model_tensor_stimulus.numpy()
        stimulus_array = stimulus_array - mean_stimulus
        stimulus_array = stimulus_array / std_stimulus

        if plot_weight_heatmaps:
            # create a diverging pallete ( 0 - white, extremities - red)
            cmap = sns.diverging_palette(10, 10, as_cmap = True)

            """
    
            HEATMAP RESPONSE
    
            """

            # compute the heatmaps limits
            min_response = response_array[0][0][0]
            for channel in range(NUMBER_OF_CHANNELS):
                min_response = min(min_response, response_array[channel].min())

            max_response = response_array[0][0][0]
            for channel in range(NUMBER_OF_CHANNELS):
                max_response = max(max_response, response_array[channel].max())

            channel = 0

            # create 8 figs and plot response input
            for count in range(8):
                f, axes = plt.subplots(4, 4)

                # plot each channel
                for row in range(4):
                    for col in range(4):
                        sns.heatmap(response_array[channel], cmap = cmap, center = 0.0, ax = axes[row][col],
                                    cbar = False, vmin = min_response, vmax = max_response)
                        axes[row][col].set_ylabel('')
                        axes[row][col].set_xlabel('')
                        axes[row][col].set_xticks([])
                        axes[row][col].set_yticks([])
                        axes[row][col].set_title(f'{channel}', fontdict = { 'fontsize': 7 }, pad = 0)
                        log(f'Plotted channel {channel}', file = None, widget = widget)
                        channel += 1
                f.savefig(os.path.join(division_directory_heatmap, f"{division_name}_response_heatmap_{count}.png"))
                plt.close(f)

            channel = 0

            """
            HEATMAP STIMULUS
            """

            # compute the heatmaps limits
            min_stimulus = stimulus_array[0][0][0]
            for channel in range(NUMBER_OF_CHANNELS):
                min_stimulus = min(min_stimulus, stimulus_array[channel].min())

            max_stimulus = stimulus_array[0][0][0]
            for channel in range(NUMBER_OF_CHANNELS):
                max_stimulus = max(max_stimulus, stimulus_array[channel].max())

            channel = 0

            # create 8 figs and plot stimulus input
            for count in range(8):
                f, axes = plt.subplots(4, 4)

                # plot each channel
                for row in range(4):
                    for col in range(4):
                        sns.heatmap(stimulus_array[channel], cmap = cmap, center = 0.0, ax = axes[row][col],
                                    cbar = False, vmax = max_stimulus, vmin = min_stimulus)
                        axes[row][col].set_ylabel('')
                        axes[row][col].set_xlabel('')
                        axes[row][col].set_xticks([])
                        axes[row][col].set_yticks([])
                        axes[row][col].set_title(f'{channel}', fontdict = { 'fontsize': 7 }, pad = 0)
                        log(f'Plotted channel {channel}', file = None, widget = widget)
                        channel += 1
                f.savefig(os.path.join(division_directory_heatmap, f"{division_name}_stimulus_heatmap_{count}.png"))
                plt.close(f)

            channel = 0

        if plot_collapsed_weight_heatmaps:
            """
            HEATMAPS Y COLLAPSE
            """

            collapsed_folder = os.path.join(division_directory_heatmap, 'CollapsedHeatmaps')
            if not os.path.exists(collapsed_folder):
                os.makedirs(collapsed_folder)

            plot_collapsed_heatmaps(stimulus_array, collapsed_folder, 'stimulus_input',
                                    int(division_name.split('_')[-2]),
                                    int(division_name.split('_')[-1]), DIVISION_LENGTH, True, True, False)
            plot_collapsed_heatmaps(response_array, collapsed_folder, 'response_input',
                                    int(division_name.split('_')[-2]),
                                    int(division_name.split('_')[-1]), DIVISION_LENGTH, False, True, False)

        if plot_collapsed_weight_heatmaps_aligned:
            """
            HEATMAPS Y COLLAPSE
            """

            collapsed_folder = os.path.join(division_directory_heatmap, 'CollapsedHeatmapsAligned')
            if not os.path.exists(collapsed_folder):
                os.makedirs(collapsed_folder)

            plot_collapsed_heatmaps(stimulus_array, collapsed_folder, 'stimulus_input',
                                    int(division_name.split('_')[-2]),
                                    int(division_name.split('_')[-1]), DIVISION_LENGTH, True, True, True)
            plot_collapsed_heatmaps(response_array, collapsed_folder, 'response_input',
                                    int(division_name.split('_')[-2]),
                                    int(division_name.split('_')[-1]), DIVISION_LENGTH, False, True, True)


def plot_collapsed_heatmaps(array, collapsed_folder, label, window_size, window_offset, division_length, is_stimulus,
                            add_input_mapping, is_aligned):
    collapse_function = lambda x: sum(abs(x))

    array_collapsed = []
    for channel in range(NUMBER_OF_CHANNELS):
        array_collapsed.append(np.apply_along_axis(collapse_function, 0, array[channel]))

    if not is_aligned:

        figure = go.Figure()

        for channel in range(NUMBER_OF_CHANNELS):
            figure.add_trace(
                go.Scatter(
                    x = [x for x in range(len(array_collapsed[channel]))],
                    y = array_collapsed[channel],
                    name = f'Channel {channel}',
                    mode = 'lines'
                )
            )

        nr_windows = (division_length - window_size) // window_offset + 1
        if (division_length - window_size) % window_offset != 0:
            nr_windows += 1

        if add_input_mapping:
            for window_index in range(0, nr_windows):
                # shift the window to the right with window_offset starting from 0
                offset_stimulus = window_offset * window_index
                stimulus_end = min(division_length, offset_stimulus + window_size)

                if is_stimulus:
                    figure.add_trace(
                        go.Scatter(
                            x = [window_index * window_size,
                                 min(len(array_collapsed[0]), window_size * (window_index + 1))],
                            y = [0, 0],
                            name = f'Stimulus {offset_stimulus}:{stimulus_end}',
                            mode = 'lines'
                        )
                    )
                else:
                    figure.add_trace(
                        go.Scatter(
                            x = [window_index * window_size,
                                 min(len(array_collapsed[0]), window_size * (window_index + 1))],
                            y = [0, 0],
                            name = f'Response -{division_length - offset_stimulus}:-{division_length - stimulus_end}',
                            mode = 'lines'
                        )
                    )

        plotly.offline.plot(figure, filename = os.path.join(collapsed_folder,
                                                            f'Collapsed_channel_{label}.html'),
                            auto_open = False)

        array_collapsed = np.array(array_collapsed)
        array_collapsed_twice = np.apply_along_axis(collapse_function, 0, array_collapsed)

        figure = go.Figure()
        figure.add_trace(
            go.Scatter(
                x = [x for x in range(len(array_collapsed_twice))],
                y = array_collapsed_twice,
                name = f'Channel aggregated',
                mode = 'lines'
            )
        )

        nr_windows = (division_length - window_size) // window_offset + 1
        if (division_length - window_size) % window_offset != 0:
            nr_windows += 1

        if add_input_mapping:
            for window_index in range(0, nr_windows):
                # shift the window to the right with window_offset starting from 0
                offset_stimulus = window_offset * window_index
                stimulus_end = min(division_length, offset_stimulus + window_size)

                if is_stimulus:
                    figure.add_trace(
                        go.Scatter(
                            x = [window_size * window_index,
                                 min(len(array_collapsed_twice), window_size * (window_index + 1))],
                            y = [0, 0],
                            name = f'Stimulus {offset_stimulus}:{stimulus_end}',
                            mode = 'lines'
                        )
                    )
                else:
                    figure.add_trace(
                        go.Scatter(
                            x = [window_size * window_index,
                                 min(len(array_collapsed_twice), window_size * (window_index + 1))],
                            y = [0, 0],
                            name = f'Response -{division_length - offset_stimulus}:-{division_length - stimulus_end}',
                            mode = 'lines'
                        )
                    )

        plotly.offline.plot(figure, filename = os.path.join(collapsed_folder,
                                                            f'Collapsed_channel_{label}_agg.html'),
                            auto_open = False)
    else:

        figure = go.Figure()
        figure_channel_same_color = go.Figure()

        nr_windows = (division_length - window_size) // window_offset + 1
        if (division_length - window_size) % window_offset != 0:
            nr_windows += 1

        for channel in range(NUMBER_OF_CHANNELS):

            for window_index in range(0, nr_windows):

                # shift the window to the right with window_offset starting from 0
                offset_stimulus = window_offset * window_index
                stimulus_end = min(division_length, offset_stimulus + window_size)

                if is_stimulus:
                    figure.add_trace(
                        go.Scatter(
                            x = [x for x in range(offset_stimulus, stimulus_end)],
                            y = array_collapsed[channel][window_index * window_size: window_index * window_size + (
                                    stimulus_end - offset_stimulus + 1)],
                            name = f'Channel {channel} {offset_stimulus}:{stimulus_end}',
                            mode = 'lines'
                        )
                    )
                    figure_channel_same_color.add_trace(
                        go.Scatter(
                            x = [x for x in range(offset_stimulus, stimulus_end)],
                            y = array_collapsed[channel][window_index * window_size: window_index * window_size + (
                                    stimulus_end - offset_stimulus + 1)],
                            line = dict(
                                color = COLOR_LIST_CHANNELS[channel]
                            ),
                            name = f'Channel {channel} {offset_stimulus}:{stimulus_end}',
                            mode = 'lines'
                        )
                    )
                else:
                    figure.add_trace(
                        go.Scatter(
                            x = [x for x in range(offset_stimulus, stimulus_end)],
                            y = array_collapsed[channel][window_index * window_size: window_index * window_size + (
                                    stimulus_end - offset_stimulus + 1)],
                            name = f'Channel {channel} -{division_length - offset_stimulus}:-{division_length - stimulus_end}',
                            mode = 'lines'
                        )
                    )
                    figure_channel_same_color.add_trace(
                        go.Scatter(
                            x = [x for x in range(offset_stimulus, stimulus_end)],
                            y = array_collapsed[channel][window_index * window_size: window_index * window_size + (
                                    stimulus_end - offset_stimulus + 1)],
                            line = dict(
                                color = COLOR_LIST_CHANNELS[channel]
                            ),
                            name = f'Channel {channel} -{division_length - offset_stimulus}:-{division_length - stimulus_end}',
                            mode = 'lines'
                        )
                    )

        plotly.offline.plot(figure, filename = os.path.join(collapsed_folder,
                                                            f'Collapsed_channel_{label}.html'),
                            auto_open = False)
        plotly.offline.plot(figure_channel_same_color, filename = os.path.join(collapsed_folder,
                                                                               f'Collapsed_channel_{label}_same_color.html'),
                            auto_open = False)

        array_collapsed = np.array(array_collapsed)
        array_collapsed_twice = np.apply_along_axis(collapse_function, 0, array_collapsed)

        figure = go.Figure()

        for window_index in range(0, nr_windows):

            # shift the window to the right with window_offset starting from 0
            offset_stimulus = window_offset * window_index
            stimulus_end = min(division_length, offset_stimulus + window_size)

            if is_stimulus:
                figure.add_trace(
                    go.Scatter(
                        x = [x for x in range(offset_stimulus, stimulus_end)],
                        y = array_collapsed_twice[window_index * window_size: window_index * window_size + (
                                stimulus_end - offset_stimulus + 1)],
                        name = f'Stimulus {offset_stimulus}:{stimulus_end}',
                        mode = 'lines'
                    )
                )
            else:
                figure.add_trace(
                    go.Scatter(
                        x = [x for x in range(offset_stimulus, stimulus_end)],
                        y = array_collapsed_twice[window_index * window_size: window_index * window_size + (
                                stimulus_end - offset_stimulus + 1)],
                        name = f'Response -{division_length - offset_stimulus}:-{division_length - stimulus_end}',
                        mode = 'lines'
                    )
                )

        plotly.offline.plot(figure, filename = os.path.join(collapsed_folder,
                                                            f'Collapsed_channel_{label}_agg.html'),
                            auto_open = False)
