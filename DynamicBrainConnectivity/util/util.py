import json
import math
import os

import numpy as np
import plotly
import plotly.express as px
import plotly.graph_objects as go


def get_string_from_number(number):
    """
    Gets the string from a number:
    - X => 00X
    - XY => 0XY
    - XYZ => XYZ
    """
    if number < 10:
        return '00' + str(number)
    else:
        if number < 100:
            return '0' + str(number)
        else:
            return str(number)


def get_hidden_layer_size(example_length):
    """
    Computes the hidden layer size as the first 2's power greater than the input
    """
    two_power = int(math.log2(example_length)) + 1
    return 2 ** two_power


def log(text, file, widget = None):
    if file:
        with open(file, 'a+') as f:
            print(text, file = f)

    if widget:
        widget.emit(text)


def get_example_length(division_length, window_size, window_offset):
    example_length = 0

    nr_windows = (division_length - window_size) // window_offset + 1
    if (division_length - window_size) % window_offset != 0:
        nr_windows += 1

    for window_index in range(0, nr_windows):
        # shift the window to the right with window_offset starting from 0
        offset_stimulus = window_offset * window_index
        example_length += min(division_length, offset_stimulus + window_size) - offset_stimulus

    return example_length


# compute hidden layer size
def get_hidden_size(example_length, output_size):
    return int(example_length * 2 / 3 + output_size)


def plot_histogram(data1, data2, name1, name2, output_directory, file_name, median_value):
    fig = go.Figure()
    max_val = max(max(data1), max(data2))

    fig.add_trace(go.Histogram(x = np.array(data1), name = name1, xbins = {
        'start': 0,
        'end': max_val,
        'size': 500
    }))

    fig.add_trace(go.Histogram(x = np.array(data2), name = name2, xbins = {
        'start': 0,
        'end': max_val,
        'size': 500
    }))

    fig.add_trace(
        go.Scatter(
            x = [median_value, median_value],
            y = [0, 50],
            mode = 'lines+markers',
            name = 'Median threshold'
        )
    )

    # Overlay both histograms
    fig.update_layout(barmode = 'overlay')
    # Reduce opacity to see both histograms
    fig.update_traces(opacity = 0.75)
    file_name = os.path.join(output_directory, file_name)
    plotly.offline.plot(fig, filename = file_name + ".html", auto_open = False)

    number_of_bins = max_val // 500
    data1_bin = []
    data2_bin = []

    percent_missclassified = []
    percent_missclassified_labels = []

    for bin in range(number_of_bins):
        data1_bin.append(
            len(list(filter(lambda value: value >= ((bin + 1) * 500) and value < ((bin + 2) * 500), data1))))
        data2_bin.append(
            len(list(filter(lambda value: value >= ((bin + 1) * 500) and value < ((bin + 2) * 500), data2))))

        if (data1_bin[bin] + data2_bin[bin]) > 0:
            percent_missclassified.append(data2_bin[bin] / (data1_bin[bin] + data2_bin[bin]) * 100)
        else:
            percent_missclassified.append(0)

        percent_missclassified_labels.append(((bin + 1) * 500))

    fig = px.bar(x = percent_missclassified_labels, y = percent_missclassified,
                 labels = { 'x': 'trial_length', 'y': 'percent' })
    plotly.offline.plot(fig, filename = file_name + "_percent.html", auto_open = False)


def save_dictionary_to_file(dict, path):
    json_dump = json.dumps(dict)
    output_file = open(path, "w+")
    output_file.write(json_dump)
    output_file.close()


def load_dictionary_from_file(path):
    with open(path) as json_file:
        return json.load(json_file)


def exponential_moving_average(values, period = 5):
    new_values = []

    smoothing_factor = 2 / (period + 1)

    for index, value in enumerate(values):
        if index == 0:
            new_values.append(value)
        else:
            new_values.append(smoothing_factor * value + (1 - smoothing_factor) * new_values[-1])

    return new_values
