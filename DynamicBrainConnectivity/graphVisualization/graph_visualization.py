import datetime
import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from util.constants import INTERVAL_START, INTERVAL_END
from graphVisualization.graph_visualization_util import aggregate_channels, plot_graph
from util.util import log


def graph_regions_plot_individual(matrices_directory, output_directory, trial_index, window_index, is_trial = False,
                                  widget = None, normalize = True, should_filter = True):
    # create output directory
    output_directory = os.path.join(output_directory, 'GraphWavenetAdjacency')
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    if is_trial:
        output_directory = os.path.join(output_directory, 'Trial')
    else:
        output_directory = os.path.join(output_directory, 'Window')
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    output_directory = os.path.join(output_directory, f'{trial_index}')
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    output_directory = os.path.join(output_directory, 'Individual')
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    log("Started graph regions plot individual: " + str(datetime.datetime.now()), file = None, widget = widget)

    # find input matrix
    if is_trial:
        matrices_directory = os.path.join(matrices_directory, 'Trial')
    else:
        matrices_directory = os.path.join(matrices_directory, 'Window')

    matrices_directory = os.path.join(matrices_directory, f'{trial_index}')

    node_size, node_edges = aggregate_channels(matrices_directory, trial_index, window_index, is_trial, normalize,
                                               should_filter)

    title = f'{trial_index}'
    if not is_trial:
        title += f' {window_index}'

    plot_graph(node_edges, node_size, title, output_directory, plt.cm.Blues, INTERVAL_START, INTERVAL_END)

    log("Finished graph regions plot individual: " + str(datetime.datetime.now()), file = None, widget = widget)


def graph_regions_plot_window_difference(matrices_directory, output_directory, trial_index,
                                         widget = None, normalize = True, should_filter = True):
    # create output directory
    output_directory = os.path.join(output_directory, 'GraphWavenetAdjacency')
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    output_directory = os.path.join(output_directory, 'Window')
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    output_directory = os.path.join(output_directory, f'{trial_index}')
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    output_directory = os.path.join(output_directory, 'Differences')
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    log("Started graph regions plot window difference: " + str(datetime.datetime.now()), file = None, widget = widget)

    # find input matrix
    matrices_directory = os.path.join(matrices_directory, 'Window')
    matrices_directory = os.path.join(matrices_directory, f'{trial_index}')

    node_sizes_list = []
    node_edges_list = []

    folder_list = [int(x) for x in next(os.walk(matrices_directory))[1]]
    folder_list = sorted(folder_list)

    for window_index in folder_list:
        node_size, node_edges = aggregate_channels(matrices_directory, trial_index, window_index, False, normalize,
                                                   should_filter)

        node_sizes_list.append(node_size)
        node_edges_list.append(node_edges)

    node_sizes_list = [np.array(x) for x in node_sizes_list]
    node_edges_list = [np.array(x) for x in node_edges_list]

    node_sizes_differences = []
    node_edges_differences = []

    node_similarity = []
    edge_similarity = []

    for index in range(1, len(node_sizes_list)):
        node_size = node_sizes_list[index] - node_sizes_list[index - 1]
        node_edges = node_edges_list[index] - node_edges_list[index - 1]

        node_sizes_differences.append(node_size)
        node_edges_differences.append(node_edges)

        node_similarity.append(1 - abs(node_size).sum() / node_size.size)
        edge_similarity.append(1 - abs(node_edges).sum() / node_edges.size)

    similarity_file = os.path.join(output_directory, "similarity.txt")

    if os.path.exists(similarity_file):
        os.remove(similarity_file)

    for index in range(len(node_similarity)):
        log(f'Window {index + 1}-{index}', file = similarity_file, widget = None)
        log(f'\tNode similarity: {node_similarity[index]}', file = similarity_file, widget = None)
        log(f'\tEdge similarity: {edge_similarity[index]}', file = similarity_file, widget = None)

    maximum = max(
        max([x.max() for x in node_sizes_differences]),
        max([x.max() for x in node_edges_differences])
    )
    minimum = min(
        min([x.min() for x in node_sizes_differences]),
        min([x.min() for x in node_edges_differences])
    )
    maximum = max(maximum, abs(minimum))
    minimum = -maximum

    for index in range(len(node_sizes_differences)):
        title = f'{trial_index} {index + 1}-{index}'

        plot_graph(node_edges_differences[index], node_sizes_differences[index], title, output_directory, plt.cm.bwr,
                   minimum, maximum)

    log("Finished graph regions plot window difference: " + str(datetime.datetime.now()), file = None, widget = widget)


def concatenate_images_trial(directory, trial, folder_type, images_per_row, name_difference = False):
    trial_directory = os.path.join(directory, f'{trial}')
    image_directory = os.path.join(trial_directory, folder_type)

    output_directory = os.path.join(trial_directory, folder_type + '_Concatenated')
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    if not name_difference:
        images = [(file, int(file.split(' ')[1].split('.')[0])) for file in os.listdir(image_directory)]
    else:
        images = [(file, int(file.split(' ')[1].split('-')[0])) for file in os.listdir(image_directory)]

    images = [tup[0] for tup in sorted(images, key = lambda x: x[1])]

    images = [Image.open(os.path.join(image_directory, image)) for image in images]

    number_of_images = len(images)
    number_of_rows = number_of_images // images_per_row + 1
    if number_of_images % images_per_row == 0:
        number_of_rows -= 1

    image_width, image_height = images[0].size

    new_im = Image.new('RGB', (image_width * images_per_row, image_height * number_of_rows))
    dummy_image = Image.new('RGB', (image_width, image_height))

    for row in range(image_width):
        for column in range(image_height):
            dummy_image.putpixel((row, column), (255, 255, 255))

    count = 0
    for row in range(number_of_rows):
        for column in range(images_per_row):
            if count < number_of_images:
                new_im.paste(images[count], (column * image_width, row * image_height))
                count += 1
            else:
                new_im.paste(dummy_image, (column * image_width, row * image_height))

    new_im.save(os.path.join(output_directory, f'{trial}.png'))


