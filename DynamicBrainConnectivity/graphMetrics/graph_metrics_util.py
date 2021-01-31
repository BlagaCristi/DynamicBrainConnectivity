import os
import pickle

import networkx as nx
import numpy as np
import plotly
import plotly.graph_objects as go
from networkx.algorithms.clique import find_cliques
from networkx.algorithms.components import strongly_connected_components

from util.constants import NUMBER_OF_CHANNELS, CHANNELS_DICT, CHANNELS_PLACEMENT, CHANNELS_PLACEMENT_LABEL, \
    AVG_SHORTEST_PATH, AVG_MAX_WEIGHT_PATH, MSA_WEIGHT, STRONGLY_CONNECTED_COMPONENTS, LONGEST_COMPONET_LENGTH, \
    MAX_CLIQUE_LENGTH, NUMBER_OF_MAX_CLIQUES, AVG_NR_OF_TRIANGLES, TRANSITIVITY, UNDIRECTED_AVG_CLUSTERING, \
    DIRECTED_AVG_CLUSTERING, DIRECTED_WEIGHTED_AVG_CLUSTERING, UNDIRECTED_AVG_SQ_CLUSTERING, DIRECTED_AVG_SQ_CLUSTERING, \
    AVG_DEGREE_CENTRALITY, AVG_IN_DEGREE_CENTRALITY, AVG_OUT_DEGREE_CENTRALITY, AVG_UNWEIGHTED_BETWEENES_CENTRALITY, \
    AVG_WEIGHTED_BETWEENES_CENTRALITY
from util.util import log


def centrality(output_directory, is_trial, graphs, widget, properties_dict):
    output_directory = os.path.join(output_directory, 'Centrality')
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    log_file = os.path.join(output_directory, 'Centrality.txt')

    for window, graph in enumerate(graphs):

        if not is_trial:
            log(f'Window {window}:', file = log_file)

        # compute degree centrality
        node_degree_centrality = nx.degree_centrality(graph)

        avg_degree_centrality = 0
        for node in list(node_degree_centrality.keys()):
            avg_degree_centrality += node_degree_centrality[node]

        avg_degree_centrality /= len(node_degree_centrality.keys())

        node_degree_centrality = nx.in_degree_centrality(graph)

        avg_in_degree_centrality = 0
        for node in list(node_degree_centrality.keys()):
            avg_in_degree_centrality += node_degree_centrality[node]

        avg_in_degree_centrality /= len(node_degree_centrality.keys())

        node_degree_centrality = nx.out_degree_centrality(graph)

        avg_out_degree_centrality = 0
        for node in list(node_degree_centrality.keys()):
            avg_out_degree_centrality += node_degree_centrality[node]

        avg_out_degree_centrality /= len(node_degree_centrality.keys())

        log(f'\tAverage Degree Centrality: {avg_degree_centrality}', file = log_file)
        log(f'\tAverage In Degree Centrality: {avg_in_degree_centrality}', file = log_file)
        log(f'\tAverage Out Degree Centrality: {avg_out_degree_centrality}', file = log_file)

        # compute betweenes centrality
        unweighted_betweenes_centrality = nx.betweenness_centrality(graph)

        avg_unweighted_betweenes_centrality = 0
        for node in list(unweighted_betweenes_centrality.keys()):
            avg_unweighted_betweenes_centrality += unweighted_betweenes_centrality[node]

        avg_unweighted_betweenes_centrality /= len(unweighted_betweenes_centrality.keys())

        weighted_betweenes_centrality = nx.betweenness_centrality(graph, weight = "weight")

        avg_weighted_betweenes_centrality = 0
        for node in list(weighted_betweenes_centrality.keys()):
            avg_weighted_betweenes_centrality += weighted_betweenes_centrality[node]

        avg_weighted_betweenes_centrality /= len(weighted_betweenes_centrality.keys())

        log(f'\tAverage Unweighted Betweenes Centrality: {avg_unweighted_betweenes_centrality}', file = log_file)
        log(f'\tAverage Weighted Betweenes Centrality: {avg_weighted_betweenes_centrality}', file = log_file)

        if not is_trial:
            properties_dict[window][AVG_DEGREE_CENTRALITY] = avg_degree_centrality
            properties_dict[window][AVG_IN_DEGREE_CENTRALITY] = avg_in_degree_centrality
            properties_dict[window][AVG_OUT_DEGREE_CENTRALITY] = avg_out_degree_centrality
            properties_dict[window][AVG_UNWEIGHTED_BETWEENES_CENTRALITY] = avg_unweighted_betweenes_centrality
            properties_dict[window][AVG_WEIGHTED_BETWEENES_CENTRALITY] = avg_weighted_betweenes_centrality
        else:
            properties_dict[AVG_DEGREE_CENTRALITY] = avg_degree_centrality
            properties_dict[AVG_IN_DEGREE_CENTRALITY] = avg_in_degree_centrality
            properties_dict[AVG_OUT_DEGREE_CENTRALITY] = avg_out_degree_centrality
            properties_dict[AVG_UNWEIGHTED_BETWEENES_CENTRALITY] = avg_unweighted_betweenes_centrality
            properties_dict[AVG_WEIGHTED_BETWEENES_CENTRALITY] = avg_weighted_betweenes_centrality


def clustering(output_directory, is_trial, graphs, widget, properties_dict):
    output_directory = os.path.join(output_directory, 'Clustering')
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    log_file = os.path.join(output_directory, 'Clustering.txt')

    for window, graph in enumerate(graphs):

        if not is_trial:
            log(f'Window {window}:', file = log_file)

        # compute transitivity
        transitivity = nx.transitivity(graph.to_undirected())
        log(f'\tTransitivity: {transitivity}', file = log_file)

        # compute average number of triangles
        node_triangles = nx.triangles(graph.to_undirected())

        average_triangles = 0
        for node in list(node_triangles.keys()):
            average_triangles += node_triangles[node]

        average_triangles /= len(list(node_triangles.keys()))

        log(f'\t Average number of triangles: {average_triangles}', file = log_file)

        # compute average clustering
        undirected_average_clustering = nx.average_clustering(graph.to_undirected())
        directed_average_clustering = nx.average_clustering(graph)
        directed_weighted_average_clustering = nx.average_clustering(graph, weight = 'weight')

        log(f'\tUndirected average clustering: {undirected_average_clustering}', file = log_file)
        log(f'\tDirected average clustering: {directed_average_clustering}', file = log_file)
        log(f'\tDirected weighted average clustering: {directed_weighted_average_clustering}', file = log_file)

        # compute square clustering

        # undirected
        nodes_square_clustering = nx.square_clustering(graph.to_undirected())

        undirected_average_square_clustering = 0
        for node in list(nodes_square_clustering.keys()):
            undirected_average_square_clustering += nodes_square_clustering[node]

        undirected_average_square_clustering /= len(nodes_square_clustering.keys())

        # directed
        nodes_square_clustering = nx.square_clustering(graph)

        directed_average_square_clustering = 0
        for node in list(nodes_square_clustering.keys()):
            directed_average_square_clustering += nodes_square_clustering[node]

        directed_average_square_clustering /= len(nodes_square_clustering.keys())

        log(f'\tUndirected average square clustering: {undirected_average_square_clustering}', file = log_file)
        log(f'\tDirected average square clustering: {directed_average_square_clustering}', file = log_file)

        if not is_trial:
            properties_dict[window][AVG_NR_OF_TRIANGLES] = average_triangles
            properties_dict[window][TRANSITIVITY] = transitivity
            properties_dict[window][UNDIRECTED_AVG_CLUSTERING] = undirected_average_clustering
            properties_dict[window][DIRECTED_AVG_CLUSTERING] = directed_average_clustering
            properties_dict[window][DIRECTED_WEIGHTED_AVG_CLUSTERING] = directed_weighted_average_clustering
            properties_dict[window][UNDIRECTED_AVG_SQ_CLUSTERING] = undirected_average_square_clustering
            properties_dict[window][DIRECTED_AVG_SQ_CLUSTERING] = directed_average_square_clustering
        else:
            properties_dict[AVG_NR_OF_TRIANGLES] = average_triangles
            properties_dict[TRANSITIVITY] = transitivity
            properties_dict[UNDIRECTED_AVG_CLUSTERING] = undirected_average_clustering
            properties_dict[DIRECTED_AVG_CLUSTERING] = directed_average_clustering
            properties_dict[DIRECTED_WEIGHTED_AVG_CLUSTERING] = directed_weighted_average_clustering
            properties_dict[UNDIRECTED_AVG_SQ_CLUSTERING] = undirected_average_square_clustering
            properties_dict[DIRECTED_AVG_SQ_CLUSTERING] = directed_average_square_clustering


def graph_shortest_path(output_directory, is_trial, graphs, widget, properties_dict):
    output_directory = os.path.join(output_directory, 'ShortestPath')
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    log_file = os.path.join(output_directory, 'ShortestPath.txt')

    for window, graph in enumerate(graphs):
        shortest_path_dict = nx.shortest_path(graph, weight = 'weight')

        average = 0
        for start in list(shortest_path_dict.keys()):
            for end in list(shortest_path_dict[start].keys()):
                if start != end:
                    path = shortest_path_dict[start][end]

                    path_weight = 0
                    for index in range(1, len(path)):
                        path_weight = path_weight + 1 - graph.get_edge_data(path[index - 1], path[index])['weight']

                    average += path_weight
        average /= (NUMBER_OF_CHANNELS * (NUMBER_OF_CHANNELS - 1.0))

        shortest_path = nx.average_shortest_path_length(graph)

        if not is_trial:
            log(f'Window {window}: ', file = log_file)
        log(f'\tAverage shortest path length: {shortest_path}', file = log_file)
        log(f'\tAverage maximum weight path: {average}', file = log_file)

        if not is_trial:
            properties_dict[window][AVG_SHORTEST_PATH] = shortest_path
            properties_dict[window][AVG_MAX_WEIGHT_PATH] = average
        else:
            properties_dict[AVG_SHORTEST_PATH] = shortest_path
            properties_dict[AVG_MAX_WEIGHT_PATH] = average


def graph_minimum_spanning_arborescence(output_directory, is_trial, graphs, adjacency_matrices, widget,
                                        properties_dict):
    output_directory = os.path.join(output_directory, 'MaximumSpanningArborescence')
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    log_file = os.path.join(output_directory, 'MaximumSpanningArborescence.txt')

    msa_list = []
    for window, graph in enumerate(graphs):
        msa_graph = nx.minimum_spanning_arborescence(graph)

        msa_list.append(msa_graph.edges())

        count = 0
        weight = 0

        for edge in msa_graph.edges():
            node_index_1 = 0
            node_index_2 = 0

            for key in list(CHANNELS_DICT.keys()):
                if CHANNELS_DICT[key] == edge[0]:
                    node_index_1 = key
                    break

            for key in list(CHANNELS_DICT.keys()):
                if CHANNELS_DICT[key] == edge[1]:
                    node_index_2 = key
                    break

            weight += adjacency_matrices[window][node_index_1][node_index_2]
            count += 1

        if not is_trial:
            log(f'Window {window}', file = log_file)
        log(f'\t Weight: {weight}', file = log_file)
        log(f'\t Weight average: {weight / (count * 1.0)}', file = log_file)

        if not is_trial:
            properties_dict[window][MSA_WEIGHT] = weight
        else:
            properties_dict[MSA_WEIGHT] = weight

    with open(os.path.join(output_directory, 'MSAList.bin'), 'wb+') as f:
        pickle.dump(msa_list, f)


def graph_strongly_connected_components(output_directory, is_trial, graphs, widget, properties_dict):
    output_directory = os.path.join(output_directory, 'StronglyConnectedComponents')
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    log_file = os.path.join(output_directory, 'StronglyConnectedComponents.txt')

    largest_components = []

    for window, graph in enumerate(graphs):
        components = strongly_connected_components(graph)

        if is_trial:
            log('Strongly connected components: ', file = log_file, widget = None)
        else:
            log(f'Strongly Connected Components for window {window}:', file = log_file, widget = None)

        components = [component for component in components]
        components = sorted(components, key = lambda x: len(x))

        largest_components.append(components[-1])

        longest_component_length = len(components[-1])
        components_greater_than_one = 0

        for component in components:

            if len(component) > 1:
                components_greater_than_one += 1

            component_regions = []

            for node in component:
                node_index = 0

                for key in list(CHANNELS_DICT.keys()):
                    if CHANNELS_DICT[key] == node:
                        node_index = key
                        break

                region_index = 0

                for index, region in enumerate(CHANNELS_PLACEMENT):
                    if node_index in region:
                        region_index = index
                        break

                component_regions.append(CHANNELS_PLACEMENT_LABEL[region_index].replace('\n', '_'))

            if is_trial:
                log(f'\t Length {len(component)} : {component}', file = log_file, widget = None)
                log(f'\t Length {len(component_regions)} : {component_regions}', file = log_file, widget = None)
            else:
                log(f'\t Length {len(component)} : {component}', file = log_file, widget = None)
                log(f'\t Length {len(component_regions)} : {component_regions}', file = log_file, widget = None)

        if not is_trial:
            properties_dict[window][STRONGLY_CONNECTED_COMPONENTS] = components_greater_than_one
            properties_dict[window][LONGEST_COMPONET_LENGTH] = longest_component_length
        else:
            properties_dict[STRONGLY_CONNECTED_COMPONENTS] = components_greater_than_one
            properties_dict[LONGEST_COMPONET_LENGTH] = longest_component_length

    with open(os.path.join(output_directory, 'LargeComponent.bin'), 'wb+') as f:
        pickle.dump(largest_components, f)

    components_intersection = largest_components[0]

    for index in range(1, len(largest_components)):
        aux = []
        for node in components_intersection:
            if node in largest_components[index]:
                aux.append(node)
        components_intersection = aux

    log(f'Largest component intersection; {len(components_intersection)} nodes: ', file = log_file, widget = None)
    for node in components_intersection:
        log(f'\tNode {node}', file = log_file, widget = None)

    concatenated_largest_component = []
    for component in largest_components:
        concatenated_largest_component.extend(component)

    freq_dict = { }
    for key in list(CHANNELS_DICT.keys()):
        node = CHANNELS_DICT[key]
        freq_dict[node] = concatenated_largest_component.count(node) / (len(largest_components))

    sorted_keys = sorted(freq_dict, key = lambda x: freq_dict[x], reverse = True)

    log(f'Largest component frequencies: ', file = log_file, widget = None)
    for key in sorted_keys:
        log(f'\t{key} : {freq_dict[key]}', file = log_file, widget = None)


def graph_clique(output_directory, is_trial, graphs, widget, properties_dict):
    output_directory = os.path.join(output_directory, 'MaxClique')
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    log_file = os.path.join(output_directory, 'MaxClique.txt')
    for window, graph in enumerate(graphs):
        max_cliques = find_cliques(graph.to_undirected())

        if is_trial:
            log('Max clique: ', file = log_file, widget = None)
        else:
            log(f'Max clique for window {window}:', file = log_file, widget = None)

        max_cliques = [clique for clique in max_cliques]
        max_cliques = sorted(max_cliques, key = lambda x: len(x), reverse = True)
        max_length = len(max_cliques[0])

        number_of_cliques = 0

        for clique in max_cliques:

            if len(clique) != max_length:
                break

            number_of_cliques += 1

            max_clique_regions = []

            for node in clique:
                node_index = 0

                for key in list(CHANNELS_DICT.keys()):
                    if CHANNELS_DICT[key] == node:
                        node_index = key
                        break

                region_index = 0

                for index, region in enumerate(CHANNELS_PLACEMENT):
                    if node_index in region:
                        region_index = index
                        break

                max_clique_regions.append(CHANNELS_PLACEMENT_LABEL[region_index].replace('\n', '_'))

            if is_trial:
                log(f'\t {clique}', file = log_file, widget = None)
                log(f'\t {max_clique_regions}', file = log_file, widget = None)
            else:
                log(f'\t {clique}', file = log_file, widget = None)
                log(f'\t {max_clique_regions}', file = log_file, widget = None)

        if not is_trial:
            properties_dict[window][MAX_CLIQUE_LENGTH] = max_length
            properties_dict[window][NUMBER_OF_MAX_CLIQUES] = number_of_cliques
        else:
            properties_dict[MAX_CLIQUE_LENGTH] = max_length
            properties_dict[NUMBER_OF_MAX_CLIQUES] = number_of_cliques


def compute_histogram(input_matrix, widget, output_directory, is_trial):
    output_directory = os.path.join(output_directory, 'Histogram')
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    log_file = os.path.join(output_directory, 'histogram_details.txt')

    if is_trial:
        current = 0.0
        while current <= 1.0:
            count = len(list(filter(lambda x: x >= current, input_matrix)))
            log(f'Number connections for threshold {current}: {count}', file = log_file, widget = None)
            current += 0.1

        fig = go.Figure(
            data = [
                go.Histogram(
                    x = input_matrix,
                    xbins = dict(
                        start = 0.0,
                        end = 1.0,
                        size = 0.1
                    ),

                )
            ]
        )
        plotly.offline.plot(fig, filename = os.path.join(output_directory, 'WeightHistogram.html'), auto_open = False)
    else:

        fig = go.Figure()

        for window, matrix in enumerate(input_matrix):
            log(f'Window {window}', file = log_file, widget = None)
            current = 0.0
            while current <= 1.0:
                count = len(list(filter(lambda x: x >= current, matrix)))
                log(f'\tNumber connections for threshold {current}: {count}', file = log_file, widget = None)
                current += 0.1

            fig.add_trace(
                go.Histogram(
                    x = matrix,
                    xbins = dict(
                        start = 0.0,
                        end = 1.0,
                        size = 0.1
                    ),
                    name = f'Window {window}'
                )
            )
        fig.update_layout(barmode = 'overlay')
        # Reduce opacity to see both histograms
        fig.update_traces(opacity = 0.75)
        plotly.offline.plot(fig, filename = os.path.join(output_directory, 'WeightHistogram.html'),
                            auto_open = False)


def read_and_normalize_matrix(matrices_directory):
    # read input matrix
    matrix_path = os.path.join(matrices_directory, 'WeightMatrix.dat')
    input_matrix = np.fromfile(matrix_path, dtype = float)

    # normalize matrix
    minimum = input_matrix.min()
    maximum = input_matrix.max()

    input_matrix = (input_matrix - minimum) / (maximum - minimum)

    return input_matrix
