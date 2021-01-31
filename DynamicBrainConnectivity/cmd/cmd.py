import sys

from classificationStatistics.model_classification_statistics import model_classification_statistics
from datasetStatistics.dataset_statistics import dataset_statistics
from dualEnsembleClassifier.dual_ensemble_classifier import dual_ensemble_classifier
from dualEnsembleClassifierPerformanceStatistics.dual_ensemble_classifier_performance_statistics import \
    dual_ensemble_classifier_performance_statistics
from graphAnalysis.graph_analysis import graph_analysis
from graphMetrics.graph_metrics import graph_metrics
from graphVisualization.graph_visualization import graph_regions_plot_individual, graph_regions_plot_window_difference
from gui.validator import check_if_path_valid, check_if_int, check_if_T_F
from modelAnalytics.model_analytics import model_analytics
from rawDataFilter.raw_data_filter import raw_data_filter
from recurrentGraphWavenet.recurrent_graph_wavenet import recurrent_graph_wavenet
from trialWindowConfiguration.trial_window_configuration import trial_window_configuration
from util.constants import GRAPH_WAVENET_LOADER_OPTIONS, TRIALS_FOR_STIMULUS


def cmd_interface():
    arguments = sys.argv
    arguments_len = len(sys.argv)

    if arguments_len < 2:
        print('Error! At least one argument needed!')
        exit(-1)
    else:
        if arguments[1] == 'DualEnsembleClassifier':
            if arguments_len != 11:
                print(f'{arguments[1]} requires 9 arguments!')
                exit(-1)
            else:
                eeg_raw_data_filtered = arguments[2]
                eeg_dots_folder = arguments[3]
                output_path = arguments[4]
                window_size = arguments[5]
                window_offset = arguments[6]
                division_size = arguments[7]
                only_two_subjects = arguments[8]
                with_visdom = arguments[9]
                save_loaders = arguments[10]

                if not check_if_path_valid(eeg_raw_data_filtered):
                    print("Data path is not valid!")
                    exit(-1)

                if not check_if_path_valid(eeg_dots_folder):
                    print("EEG dots path is not valid!")
                    exit(-1)

                if not check_if_path_valid(output_path):
                    print("Output path is not valid!")
                    exit(-1)

                if not check_if_int(window_size):
                    print("Window size must be an integer!")
                    exit(-1)

                if not check_if_int(window_offset):
                    print("Window offset must be an integer!")
                    exit(-1)

                if not check_if_int(division_size):
                    print("Division size must be an integer!")
                    exit(-1)

                if not check_if_T_F(only_two_subjects):
                    print("Only two subjects must be either 'T'(true) of 'F'(false)!")
                    exit(-1)

                if not check_if_T_F(with_visdom):
                    print("With visdom must be either 'T'(true) of 'F'(false)!")
                    exit(-1)

                if not check_if_T_F(save_loaders):
                    print("Save loaders must be either 'T'(true) of 'F'(false)!")
                    exit(-1)

                dual_ensemble_classifier(
                    eeg_raw_data_filtered,
                    eeg_dots_folder,
                    output_path,
                    int(window_size),
                    int(window_offset),
                    int(division_size),
                    only_two_subjects == 'T',
                    with_visdom == 'T',
                    save_loaders == 'T',
                    None
                )

        elif arguments[1] == 'DatasetStatistics':
            if arguments_len != 4:
                print(f'{arguments[1]} requires 2 arguments!')
                exit(-1)
            else:
                eeg_dots_folder = arguments[2]
                output_path = arguments[3]

                if not check_if_path_valid(eeg_dots_folder):
                    print("EEG dots path is not valid!")
                    exit(-1)

                if not check_if_path_valid(output_path):
                    print("Output path is not valid!")
                    exit(-1)

                dataset_statistics(
                    eeg_dots_folder,
                    output_path,
                    None
                )

        elif arguments[1] == 'RawDataFilter':
            if arguments_len != 6:
                print(f'{arguments[1]} requires 4 arguments!')
                exit(-1)
            else:
                eeg_dots_folder = arguments[2]
                output_path = arguments[3]
                degree_of_parallelism = arguments[4]
                trial_filter_length = arguments[5]

                if not check_if_path_valid(eeg_dots_folder):
                    print("EEG dots path is not valid!")
                    exit(-1)

                if not check_if_path_valid(output_path):
                    print("Output path is not valid!")
                    exit(-1)

                if not check_if_int(degree_of_parallelism):
                    print("Degree of parallelism is not an integer!")
                    exit(-1)

                if not check_if_int(trial_filter_length):
                    print("Trial filter length is not an integer!")
                    exit(-1)

                raw_data_filter(
                    eeg_dots_folder,
                    output_path,
                    int(degree_of_parallelism),
                    int(trial_filter_length),
                    None
                )

        elif arguments[1] == 'DualEnsembleClassifierPerformanceStatistics':
            if arguments_len != 7:
                print(f'{arguments[1]} requires 5 arguments')
                exit(-1)
            else:
                initial_runs_folder = arguments[2]
                multiple_runs_folder = arguments[3]
                output_path = arguments[4]
                simple_plots = arguments[5]
                distribution_plots = arguments[6]

                if not check_if_path_valid(initial_runs_folder):
                    print("Initial runs folder path is not valid!")
                    exit(-1)

                if not check_if_path_valid(multiple_runs_folder):
                    print("Multiple runs folder path is not valid!")
                    exit(-1)

                if not check_if_path_valid(output_path):
                    print("The output path is not valid!")
                    exit(-1)

                if not check_if_T_F(simple_plots):
                    print("Simple plots must be either 'T'(true) of 'F'(false)!")
                    exit(-1)

                if not check_if_T_F(distribution_plots):
                    print("Distribution plots must be either 'T'(true) of 'F'(false)!")
                    exit(-1)

                dual_ensemble_classifier_performance_statistics(
                    initial_runs_folder,
                    multiple_runs_folder,
                    output_path,
                    simple_plots == 'T',
                    distribution_plots == 'T',
                    None
                )

        elif arguments[1] == 'ModelAnalytics':
            if arguments_len != 7:
                print(f'{arguments[1]} requires 5 arguments')
                exit(-1)
            else:
                multiple_runs_folder = arguments[2]
                output_path = arguments[3]
                weight_heatmaps = arguments[4]
                collapsed_heatmaps = arguments[5]
                collapsed_heatmaps_aligned = arguments[6]

                if not check_if_path_valid(multiple_runs_folder):
                    print("Multiple runs path is not valid!")
                    exit(-1)

                if not check_if_path_valid(output_path):
                    print("Output path is not valid!")
                    exit(-1)

                if not check_if_T_F(collapsed_heatmaps):
                    print("Plot collapsed weight heatmaps must be either 'T'(true) of 'F'(false)!")
                    exit(-1)

                if not check_if_T_F(collapsed_heatmaps_aligned):
                    print("Plot collapsed weight heatmaps aligned must be either 'T'(true) of 'F'(false)!")
                    exit(-1)

                if not check_if_T_F(weight_heatmaps):
                    print("Plot weight heatmaps must be either 'T'(true) of 'F'(false)!")
                    exit(-1)

                # start model analytics
                model_analytics(
                    multiple_runs_folder,
                    output_path,
                    weight_heatmaps == 'T',
                    collapsed_heatmaps == 'T',
                    collapsed_heatmaps_aligned == 'T',
                    None
                )

        elif arguments[1] == 'ModelClassificationStatistics':
            if arguments_len != 9:
                print(f'{arguments[1]} requires 7 arguments')
                exit(-1)
            else:
                model_with_loaders = arguments[2]
                trial_lengths_path = arguments[3]
                output_path = arguments[4]
                median_value = arguments[5]
                generate_from_train = arguments[6]
                generate_from_cross = arguments[7]
                generate_from_test = arguments[8]

                if not check_if_path_valid(model_with_loaders):
                    print("Model path is not valid!")
                    exit(-1)

                if not check_if_path_valid(output_path):
                    print("Output path is not valid!")
                    exit(-1)

                if not check_if_path_valid(trial_lengths_path):
                    print("Trial lengths path is not valid!")
                    exit(-1)

                if not check_if_int(median_value):
                    print("Median value must be an integer!")
                    exit(-1)

                if not check_if_T_F(generate_from_train):
                    print("Generate from train must be either 'T'(true) of 'F'(false)!")
                    exit(-1)

                if not check_if_T_F(generate_from_cross):
                    print("Generate from cross must be either 'T'(true) of 'F'(false)!")
                    exit(-1)

                if not check_if_T_F(generate_from_test):
                    print("Generate from test must be either 'T'(true) of 'F'(false)!")
                    exit(-1)

                # start model classification statistics
                model_classification_statistics(
                    model_with_loaders,
                    trial_lengths_path,
                    output_path,
                    int(median_value),
                    generate_from_train == 'T',
                    generate_from_cross == 'T',
                    generate_from_test == 'T',
                    None
                )
        elif arguments[1] == "TrialWindowConfiguration":
            if arguments_len != 7:
                print(f'{arguments[1]} requires 5 arguments')
                exit(-1)
            else:
                eeg_dots_folder = arguments[2]
                output_folder = arguments[3]
                window_size = arguments[4]
                window_offset = arguments[5]
                threshold_value = arguments[6]

                if not check_if_path_valid(eeg_dots_folder):
                    print("Dots path is not valid!")
                    exit(-1)

                if not check_if_path_valid(output_folder):
                    print("Output path is not valid!")
                    exit(-1)

                if not check_if_int(window_size):
                    print("Window size must be an integer!")
                    exit(-1)

                if not check_if_int(window_offset):
                    print("Window offset must be an integer!")
                    exit(-1)

                if not check_if_int(threshold_value):
                    print("Threshold must be an integer!")
                    exit(-1)

                trial_window_configuration(
                    eeg_dots_folder,
                    output_folder,
                    int(window_size),
                    int(window_offset),
                    int(threshold_value),
                    None
                )

        elif arguments[1] == 'RecurrentGraphWavenet':
            if arguments_len != 23:
                print(f'{arguments[1]} requires 21 arguments')
                exit(-1)
            else:
                eeg_dots_folder = arguments[2]
                output_folder = arguments[3]
                trial_division_file_path = arguments[4]
                subject_number = arguments[5]
                trial_index = arguments[6]
                window_index = arguments[7]
                input_length = arguments[8]
                output_length = arguments[9]
                batch_size = arguments[10]
                loader_option = arguments[11]
                blocks = arguments[12]
                layers = arguments[13]
                number_of_epochs = arguments[14]
                initial_train_percentage = arguments[15]
                increase_train_percentage = arguments[16]
                use_functional_network = arguments[17]
                functional_network_path = arguments[18]
                use_previous_weight_matrix = arguments[19]
                previous_weight_matrix_path = arguments[20]
                include_cross = arguments[21]
                use_gpu = arguments[22]

                if not check_if_path_valid(eeg_dots_folder):
                    print("Dots path is not valid!")
                    exit(-1)

                if not check_if_path_valid(output_folder):
                    print("Output path is not valid!")
                    exit(-1)

                if not check_if_path_valid(trial_division_file_path):
                    print("Trial division file path is not valid!")
                    exit(-1)

                if loader_option not in GRAPH_WAVENET_LOADER_OPTIONS:
                    print('Loader option is not valid!')
                    exit(-1)

                if not check_if_int(subject_number):
                    print("Subject number must be an integer!")
                    exit(-1)

                if not check_if_int(trial_index):
                    print("Trial index must be an integer!")
                    exit(-1)

                if not check_if_int(window_index):
                    print("Window index must be an integer!")
                    exit(-1)

                if not check_if_int(input_length):
                    print("Input length must be an integer!")
                    exit(-1)

                if not check_if_int(subject_number):
                    print("Output length must be an integer!")
                    exit(-1)

                if not check_if_int(batch_size):
                    print("Batch size must be an integer!")
                    exit(-1)

                if not check_if_int(blocks):
                    print("Blocks must be an integer!")
                    exit(-1)

                if not check_if_int(layers):
                    print("Layers must be an integer!")
                    exit(-1)

                if not check_if_int(number_of_epochs):
                    print("Number of epochs must be an integer!")
                    exit(-1)

                if not check_if_int(initial_train_percentage):
                    print("Initial train percentage must be an int!")
                    exit(-1)

                if not check_if_int(increase_train_percentage):
                    print("Increase train percentage must be an int!")
                    exit(-1)

                if not check_if_T_F(use_functional_network):
                    print("Use functional networks be T/F!")
                    exit(-1)

                if not check_if_path_valid(functional_network_path):
                    print("Functional network path is not valid!")
                    exit(-1)

                if not check_if_T_F(use_previous_weight_matrix):
                    print("Use previous weight matrix must be T/F!")
                    exit(-1)

                if not check_if_path_valid(previous_weight_matrix_path):
                    print("Previous weight matrix path is not valid!")
                    exit(-1)

                if not check_if_T_F(include_cross):
                    print("Include cross should be T/F!")
                    exit(-1)

                if not check_if_T_F(use_gpu):
                    print("Use gpu should be T/F!")

                recurrent_graph_wavenet(
                    dots_folder_path = eeg_dots_folder,
                    trial_division_file_path = trial_division_file_path,
                    output_path = output_folder,
                    subject_number = int(subject_number),
                    trial_index = int(trial_index),
                    window_index = int(window_index),
                    input_length = int(input_length),
                    output_length = int(output_length),
                    batch_size = int(batch_size),
                    loader_option = loader_option,
                    widget = None,
                    blocks = int(blocks),
                    layers = int(layers),
                    number_of_epochs = int(number_of_epochs),
                    initial_train_percentage = int(initial_train_percentage),
                    increase_train_percentage = int(increase_train_percentage),
                    use_functional_network = (use_functional_network == 'T'),
                    functional_network_path = functional_network_path,
                    use_previous_weight_matrix = (use_previous_weight_matrix == 'T'),
                    previous_weight_matrix_path = previous_weight_matrix_path,
                    include_cross = (include_cross == 'T'),
                    use_gpu = (use_gpu == 'T')
                )

        elif arguments[1] == 'GraphWindowVisualization':
            if arguments_len != 6:
                print(f'{arguments[1]} requires 4 arguments')
                exit(-1)
            else:
                matrices_directory = arguments[2]
                output_directory = arguments[3]
                trial_index = arguments[4]
                window_index = arguments[5]

                if not check_if_path_valid(matrices_directory):
                    print("Matrices directory path is not valid!")
                    exit(-1)

                if not check_if_path_valid(output_directory):
                    print("Output directory path is not valid!")
                    exit(-1)

                if not check_if_int(trial_index):
                    print("Trial index must be an integer!")
                    exit(-1)

                if not check_if_int(window_index):
                    print("Window index must be an integer!")
                    exit(-1)

                graph_regions_plot_individual(
                    matrices_directory = matrices_directory,
                    output_directory = output_directory,
                    trial_index = int(trial_index),
                    window_index = int(window_index)
                )

        elif arguments[1] == 'GraphDifferenceVisualization':
            if arguments_len != 5:
                print(f'{arguments[1]} requires 3 arguments')
                exit(-1)
            else:
                matrices_directory = arguments[2]
                output_directory = arguments[3]
                trial_index = arguments[4]

                if not check_if_path_valid(matrices_directory):
                    print("Matrices directory path is not valid!")
                    exit(-1)

                if not check_if_path_valid(output_directory):
                    print("Output directory path is not valid!")
                    exit(-1)

                if not check_if_int(trial_index):
                    print("Trial index must be an integer!")
                    exit(-1)

                graph_regions_plot_window_difference(
                    matrices_directory = matrices_directory,
                    output_directory = output_directory,
                    trial_index = int(trial_index)
                )

        elif arguments[1] == 'GraphMetrics':
            if arguments_len != 5:
                print(f'{arguments[1]} requires 3 arguments')
                exit(-1)
            else:
                matrices_directory = arguments[2]
                output_directory = arguments[3]
                trial_index = arguments[4]

                if not check_if_path_valid(matrices_directory):
                    print("Matrices directory path is not valid!")
                    exit(-1)

                if not check_if_path_valid(output_directory):
                    print("Output directory path is not valid!")
                    exit(-1)

                if not check_if_int(trial_index):
                    print("Trial index must be an integer!")
                    exit(-1)

                graph_metrics(
                    matrices_directory = matrices_directory,
                    output_directory = output_directory,
                    trial_index = int(trial_index)
                )

        elif arguments[1] == 'GraphAnalysis':
            if arguments_len != 5:
                print(f'{arguments[1]} requires 3 arguments')
                exit(-1)
            else:
                metrics_directory = arguments[2]
                output_directory = arguments[3]
                trend_period = arguments[4]

                if not check_if_path_valid(metrics_directory):
                    print("Matrices directory path is not valid!")
                    exit(-1)

                if not check_if_path_valid(output_directory):
                    print("Output directory path is not valid!")
                    exit(-1)

                if not check_if_int(trend_period):
                    print("Trend period must be an integer!")
                    exit(-1)

                graph_analysis(
                    directory = metrics_directory,
                    output_path = output_directory,
                    period = int(trend_period),
                    stimulus_list = list(TRIALS_FOR_STIMULUS.keys()),
                    widget = None
                )

        elif arguments[1] == 'help':
            print("If option is: DualEnsembleClassifier")
            print("\t 1. Path to filtered raw data")
            print("\t 2. Path to the Dots folders")
            print("\t 3. The output path")
            print("\t 4. Window size")
            print("\t 5. Window offset")
            print("\t 6. Division size")
            print("\t 7. Only two subjects T/F")
            print("\t 8. With visdom T/F")
            print("\t 9. Save loaders T/F")
            print()

            print("If option is: DatasetStatistics")
            print("\t 1. Path to the Dots folders")
            print("\t 2. The output path")
            print()

            print("If option is: RawDataFilter")
            print("\t 1. Path to the Dots folders")
            print("\t 2. The output path")
            print("\t 3. The degree of parallelism")
            print("\t 4. Trial filter length - a choice would be 627")
            print()

            print("If option is: DualEnsembleClassifierPerformanceStatistics")
            print("\t 1. Path to the initial runs folder")
            print("\t 2. Path to the multiple runs folder")
            print("\t 3. The output path")
            print("\t 4. Simple plots T/F")
            print("\t 5. Distribution plots T/F")
            print()

            print("If option is: ModelAnalytics")
            print("\t 1. Path to the multiple runs folder")
            print("\t 2. Path to the model (with loaders)")
            print("\t 3. The output path")
            print("\t 4. Plot collapsed weight heatmaps T/F")
            print("\t 5. Plot collapsed weight heatmaps alligned T/F")
            print()

            print("If option is: ModelClassificationStatistics")
            print("\t 1. Path to the model (with loaders)")
            print("\t 2. Path to the trial lengths folder")
            print("\t 3. The output path")
            print("\t 4. The median value of the trial lengths")
            print("\t 5. Generate from train T/F")
            print("\t 6. Generate from cross T/F")
            print("\t 7. Generate from test T/F")
            print()

            print("If option is: TrialWindowConfiguration")
            print("\t 1. Path to the Dots folder")
            print("\t 2. The output path")
            print("\t 3. The window size")
            print("\t 4. The window offset")
            print("\t 5. The threshold value")
            print()

            print("If option is: RecurrentGraphWavenet")
            print("\t 1. Path to the Dots folder")
            print("\t 2. The output path")
            print("\t 3. Trial window configuration path")
            print("\t 4. Subject number")
            print("\t 5. Trial index")
            print("\t 6. Window index (must be given even if loader option is not 'Window'")
            print("\t 7. Example input length")
            print("\t 8. Example output length")
            print("\t 9. The batch size")
            print("\t 10. Loader option; must be: " + str(GRAPH_WAVENET_LOADER_OPTIONS))
            print("\t 11. Number of blocks")
            print("\t 12. Number of layers")
            print("\t 13. Number of Epochs")
            print("\t 14. Initial train percentage (expressed as an int)")
            print("\t 15. Increase train percentage (expressed as an int)")
            print("\t 16. Use functional networks as support matrices")
            print("\t 17. Path to the functional networks folder")
            print("\t 18. Use previous weight matrix (T/F)")
            print("\t 19. Previous weight matrix path")
            print("\t 20. Include cross set (T/F)")
            print("\t 21. Use GPU (T/F)")

            print("If option is: GraphWindowVisualization")
            print("\t 1. Path to the RGW matrices")
            print("\t 2. The output path")
            print("\t 3. Trial index")
            print("\t 4. Window index")

            print("If option is: GraphDifferenceVisualization")
            print("\t 1. Path to the RGW matrices")
            print("\t 2. The output path")
            print("\t 3. Trial index")

            print("If option is: GraphMetrics")
            print("\t 1. Path to the RGW matrices")
            print("\t 2. The output path")
            print("\t 3. Trial index")

            print("If option is: GraphAnalysis")
            print("\t 1. Path to the metric results")
            print("\t 2. The output path")
            print("\t 3. Period for trend computation")


        else:
            print('No valid option offered!')
            print('Run the program with the option "help" in order to see which options are available.')
            exit(-1)
