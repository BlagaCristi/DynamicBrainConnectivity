import datetime
import os

from graphAnalysis.graph_analysis_util import dynamic_time_warping, trend_analysis
from util.constants import NUMBER_OF_MAX_CLIQUES, MAX_CLIQUE_LENGTH, \
    LONGEST_COMPONET_LENGTH, STRONGLY_CONNECTED_COMPONENTS, MSA_WEIGHT, AVG_MAX_WEIGHT_PATH, AVG_SHORTEST_PATH, \
    AVG_NR_OF_TRIANGLES, TRANSITIVITY, UNDIRECTED_AVG_CLUSTERING, DIRECTED_AVG_CLUSTERING, \
    DIRECTED_WEIGHTED_AVG_CLUSTERING, UNDIRECTED_AVG_SQ_CLUSTERING, DIRECTED_AVG_SQ_CLUSTERING, AVG_DEGREE_CENTRALITY, \
    AVG_IN_DEGREE_CENTRALITY, AVG_OUT_DEGREE_CENTRALITY, AVG_UNWEIGHTED_BETWEENES_CENTRALITY, \
    AVG_WEIGHTED_BETWEENES_CENTRALITY
from util.util import log, load_dictionary_from_file


def graph_analysis(directory, output_path, stimulus_list, period, widget):
    output_directory = os.path.join(output_path, 'GraphAnalysis')
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    log_file = os.path.join(output_directory, 'log.txt')
    log("Started graph analysis: " + str(datetime.datetime.now()), file = log_file, widget = widget)

    input_directory = os.path.join(directory, 'Window')
    if not os.path.exists(input_directory):
        log("Input window directory does not exist", file = log_file, widget = widget)
        return

    metrics = [
        NUMBER_OF_MAX_CLIQUES,
        MAX_CLIQUE_LENGTH,
        LONGEST_COMPONET_LENGTH,
        STRONGLY_CONNECTED_COMPONENTS,
        MSA_WEIGHT,
        AVG_MAX_WEIGHT_PATH,
        AVG_SHORTEST_PATH,
        AVG_NR_OF_TRIANGLES,
        TRANSITIVITY,
        UNDIRECTED_AVG_CLUSTERING,
        DIRECTED_AVG_CLUSTERING,
        DIRECTED_WEIGHTED_AVG_CLUSTERING,
        UNDIRECTED_AVG_SQ_CLUSTERING,
        DIRECTED_AVG_SQ_CLUSTERING,
        AVG_DEGREE_CENTRALITY,
        AVG_IN_DEGREE_CENTRALITY,
        AVG_OUT_DEGREE_CENTRALITY,
        AVG_UNWEIGHTED_BETWEENES_CENTRALITY,
        AVG_WEIGHTED_BETWEENES_CENTRALITY
    ]

    trial_folder_list = next(os.walk(input_directory))[1]
    trials = [int(trial) for trial in trial_folder_list]
    trials = sorted(trials)

    trial_dictionary = { }

    for trial in trials:
        trial_folder = os.path.join(input_directory, str(trial), 'property_dict.json')
        dict = load_dictionary_from_file(trial_folder)

        trial_dictionary[trial] = dict

    stimulus_pairs = []

    for first_stimulus in stimulus_list:
        for second_stimulus in stimulus_list:
            if first_stimulus != second_stimulus:
                if (first_stimulus, second_stimulus) in stimulus_pairs or (
                        second_stimulus, first_stimulus) in stimulus_pairs:
                    pass
                else:
                    stimulus_pairs.append((first_stimulus, second_stimulus))

    pairs_dict = { }
    for metric in metrics:
        pairs_dict[metric] = { }
        for stimulus in stimulus_list:
            pairs_dict[metric][stimulus] = { }

    log("Started dtw: " + str(datetime.datetime.now()), file = log_file, widget = widget)
    dynamic_time_warping(
        metrics = metrics,
        output_path = output_directory,
        stimulus_pairs = stimulus_pairs,
        trial_dictionary = trial_dictionary
    )

    log("Started trend: " + str(datetime.datetime.now()), file = log_file, widget = widget)
    trend_analysis(
        output_path = output_directory,
        metrics = metrics,
        trial_dictionary = trial_dictionary,
        period = period
    )

    log("Finished graph analysis: " + str(datetime.datetime.now()), file = log_file, widget = widget)
