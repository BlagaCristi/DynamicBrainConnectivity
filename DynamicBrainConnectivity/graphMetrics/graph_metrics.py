import datetime
import os

import networkx as nx

from graphMetrics.graph_metrics_util import compute_histogram, centrality, clustering, graph_shortest_path, \
    graph_minimum_spanning_arborescence, graph_strongly_connected_components, graph_clique, read_and_normalize_matrix
from util.constants import NUMBER_OF_CHANNELS, CHANNELS_DICT
from util.util import log, save_dictionary_to_file, load_dictionary_from_file


def graph_metrics(matrices_directory, output_directory, trial_index, is_trial = False,
                  widget = None, histogram = False, percentage = 0.05):
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

    # find input matrix
    if is_trial:
        matrices_directory = os.path.join(matrices_directory, 'Trial')
    else:
        matrices_directory = os.path.join(matrices_directory, 'Window')

    matrices_directory = os.path.join(matrices_directory, f'{trial_index}')

    log_file = os.path.join(output_directory, 'log.txt')

    log("Started graph analysis: " + str(datetime.datetime.now()), file = log_file, widget = widget)

    properties_dict_file_path = os.path.join(output_directory, 'property_dict.json')
    if os.path.exists(properties_dict_file_path):
        properties_dict = load_dictionary_from_file(properties_dict_file_path)
        properties_dict = { int(k): v for k, v in properties_dict.items() }
        os.remove(properties_dict_file_path)
        init_properties_dict = False
    else:
        properties_dict = { }
        init_properties_dict = True

    input_matrix = []
    if not is_trial:
        folder_list = [int(x) for x in next(os.walk(matrices_directory))[1]]
        folder_list = sorted(folder_list)
        for folder in folder_list:
            matrix_directory = os.path.join(matrices_directory, f'{folder}')
            input_matrix.append(read_and_normalize_matrix(matrix_directory))
            if init_properties_dict:
                properties_dict[folder] = { }
    else:
        input_matrix.append(read_and_normalize_matrix(matrices_directory))

    if histogram:
        compute_histogram(input_matrix, widget, output_directory, is_trial)

    inversed_filtered_graphs = []
    filtered_graphs = []
    inversed_unfiltered_graphs = []
    adjacency_matrices = []
    for matrix in input_matrix:
        inversed_unfiltered_graphs.append(
            nx.DiGraph()
        )
        inversed_filtered_graphs.append(
            nx.DiGraph()
        )
        filtered_graphs.append(
            nx.DiGraph()
        )
        values = sorted(matrix, reverse = True)
        threshold = values[int(percentage * len(values))]
        matrix = matrix.reshape(NUMBER_OF_CHANNELS, NUMBER_OF_CHANNELS)
        adjacency_matrices.append(matrix)
        for start in range(NUMBER_OF_CHANNELS):
            for end in range(NUMBER_OF_CHANNELS):
                if matrix[start][end] >= threshold:
                    inversed_filtered_graphs[-1].add_edge(
                        CHANNELS_DICT[start],
                        CHANNELS_DICT[end],
                        weight = 1.0 - matrix[start][end]
                    )
                    filtered_graphs[-1].add_edge(
                        CHANNELS_DICT[start],
                        CHANNELS_DICT[end],
                        weight = matrix[start][end]
                    )
                inversed_unfiltered_graphs[-1].add_edge(
                    CHANNELS_DICT[start],
                    CHANNELS_DICT[end],
                    weight = 1.0 - matrix[start][end]
                )

    log("Started graph clique: " + str(datetime.datetime.now()), file = log_file, widget = widget)
    graph_clique(output_directory, is_trial, inversed_filtered_graphs, widget, properties_dict)

    log("Started graph strongly connected components: " + str(datetime.datetime.now()), file = log_file,
        widget = widget)
    graph_strongly_connected_components(output_directory, is_trial, inversed_filtered_graphs, widget, properties_dict)

    log("Started graph MSA: " + str(datetime.datetime.now()), file = log_file, widget = widget)
    graph_minimum_spanning_arborescence(output_directory, is_trial, inversed_unfiltered_graphs, adjacency_matrices,
                                        widget, properties_dict)

    log("Started graph shortest path: " + str(datetime.datetime.now()), file = log_file, widget = widget)
    graph_shortest_path(output_directory, is_trial, inversed_filtered_graphs, widget, properties_dict)

    log("Started graph clustering: " + str(datetime.datetime.now()), file = log_file, widget = widget)
    clustering(output_directory, is_trial, filtered_graphs, widget, properties_dict)

    log("Started graph centrality: " + str(datetime.datetime.now()), file = log_file, widget = widget)
    centrality(output_directory, is_trial, inversed_filtered_graphs, widget, properties_dict)

    save_dictionary_to_file(properties_dict, os.path.join(output_directory, 'property_dict.json'))

    log("Finished graph analysis: " + str(datetime.datetime.now()), file = log_file, widget = widget)

