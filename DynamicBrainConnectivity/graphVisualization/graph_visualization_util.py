import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from util.constants import NUMBER_OF_CHANNELS, CHANNELS_PLACEMENT, INTERVAL_START, INTERVAL_END, CHANNELS_PLACEMENT_LABEL, \
    CHANNELS_PLACEMENT_COORD, CHANNELS_PLACEMENT_COORD_LABELS


def plot_graph(node_edges, node_size, title, output_directory, cmap, interval_start, interval_end):
    # sort edge weights
    edge_pairs = []
    for start in range(len(CHANNELS_PLACEMENT)):
        for end in range(len(CHANNELS_PLACEMENT)):
            edge_pairs.append([(start, end), node_edges[start][end]])

    edge_pairs = sorted(edge_pairs, key = lambda x: abs(x[1]))

    # plot graph
    graph = nx.OrderedDiGraph()

    for pair in edge_pairs:
        graph.add_edge(
            CHANNELS_PLACEMENT_LABEL[pair[0][0]],
            CHANNELS_PLACEMENT_LABEL[pair[0][1]],
            weight = pair[1]
        )

    # find the placement of the nodes
    node_placement = { }
    for index, label in enumerate(CHANNELS_PLACEMENT_LABEL):
        node_placement[label] = CHANNELS_PLACEMENT_COORD[index]

    graph_position = nx.spring_layout(graph, pos = node_placement, fixed = node_placement)

    # find plotted edge weights
    edge_weights = []
    for pair in edge_pairs:
        edge_weights.append(pair[1])

    node_weights = []
    for node in graph.nodes:
        node_weights.append(
            node_size[CHANNELS_PLACEMENT_LABEL.index(node)]
        )

    # plot nodes
    nx.draw_networkx_nodes(graph, graph_position, node_size = [100 for _ in range(len(CHANNELS_PLACEMENT))],
                           node_color = node_weights, cmap = cmap, vmin = interval_start, vmax = interval_end)

    edges_order = []
    for pair in edge_pairs:
        edges_order.append((CHANNELS_PLACEMENT_LABEL[pair[0][0]], CHANNELS_PLACEMENT_LABEL[pair[0][1]]))

    # plot edges
    nx.draw_networkx_edges(graph, graph_position, node_size = [100 for _ in range(len(CHANNELS_PLACEMENT))],
                           arrowstyle = '->', arrowsize = 10, edge_color = edge_weights, edge_vmin = interval_start,
                           edge_vmax = interval_end, edge_cmap = cmap, width = abs(np.array(edge_weights) * 5),
                           connectionstyle = 'arc3, rad=0.2', edgelist = edges_order)

    # find the placement of the labels
    label_placement = { }
    labels = { }
    for index, label in enumerate(CHANNELS_PLACEMENT_LABEL):
        label_placement[label] = CHANNELS_PLACEMENT_COORD_LABELS[index]
        labels[label] = label

    graph_label_position = nx.spring_layout(graph, pos = label_placement, fixed = label_placement)

    # plot labels
    nx.draw_networkx_labels(graph, graph_label_position, labels, font_size = 8)

    # add colorbar
    sm = plt.cm.ScalarMappable(cmap = cmap, norm = plt.Normalize(vmin = interval_start, vmax = interval_end))
    sm._A = []
    plt.colorbar(sm)

    # disable axis
    ax = plt.gca()
    ax.set_axis_off()

    plt.title(title)

    # plt.show()
    plt.savefig(fname = os.path.join(output_directory, title + ".png"))
    plt.close()


def aggregate_channels(matrices_directory, trial_index, window_index, is_trial, normalize, should_filter):
    if not is_trial:
        matrices_directory = os.path.join(matrices_directory, f'{window_index}')

    # read input matrix
    matrix_path = os.path.join(matrices_directory, 'WeightMatrix.dat')
    input_matrix = np.fromfile(matrix_path, dtype = float)

    # reshape input matrix
    input_matrix = input_matrix.reshape(NUMBER_OF_CHANNELS, NUMBER_OF_CHANNELS)

    # aggregate channels
    channel_dict = { }

    for index in range(len(CHANNELS_PLACEMENT)):
        for channel in CHANNELS_PLACEMENT[index]:
            channel_dict[channel] = index

    node_size = [0.0 for x in range(len(CHANNELS_PLACEMENT))]
    node_edges = [[0 for x in range(len(CHANNELS_PLACEMENT))] for y in range(len(CHANNELS_PLACEMENT))]

    if normalize:
        maximum = input_matrix.max()
        minimum = input_matrix.min()

        input_matrix = (input_matrix - minimum) / (maximum - minimum)

    threshold = 0.0
    if should_filter:
        # threshold = list(filter(lambda x: x >= 0.00001, input_matrix.reshape(NUMBER_OF_CHANNELS * NUMBER_OF_CHANNELS)))
        # threshold = sorted(threshold)[0]
        # threshold = 0.4
        threshold = sorted(input_matrix.reshape(NUMBER_OF_CHANNELS * NUMBER_OF_CHANNELS), reverse = True)[
            int(0.05 * NUMBER_OF_CHANNELS * NUMBER_OF_CHANNELS)]

    for start in range(NUMBER_OF_CHANNELS):
        for end in range(NUMBER_OF_CHANNELS):
            if start != end:
                if input_matrix[start][end] >= threshold:
                    start_pos = channel_dict[start]
                    end_pos = channel_dict[end]

                    if start_pos == end_pos:
                        node_size[start_pos] += input_matrix[start][end]
                    else:
                        node_edges[start_pos][end_pos] += input_matrix[start][end]

    # rescale values
    interval_minimum = min(min(node_size), min(min(node_edges)))
    interval_maximum = max(max(node_size), max(max(node_edges)))

    for start in range(len(CHANNELS_PLACEMENT)):
        node_size[start] = rescale_value(node_size[start], interval_minimum, interval_maximum, INTERVAL_START,
                                         INTERVAL_END)
        for end in range(len(CHANNELS_PLACEMENT)):
            node_edges[start][end] = rescale_value(node_edges[start][end], interval_minimum, interval_maximum,
                                                   INTERVAL_START, INTERVAL_END)

    return node_size, node_edges


def rescale_value(value, old_start, old_end, new_start, new_end):
    return new_start + (new_end - new_start) / (old_end - old_start) * (value - old_start)
