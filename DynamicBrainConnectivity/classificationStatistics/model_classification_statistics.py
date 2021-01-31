import datetime
import os

import numpy as np
from torch.utils.data import DataLoader

from util.constants import DIVISION_LENGTH, STIMULUS_OUTPUT_SIZE, RESPONSE_OUTPUT_SIZE, NUMBER_OF_CHANNELS
from dualEnsembleClassifier.ml.Dataset import DatasetForClassificationStatistics
from dualEnsembleClassifier.ml.DualEnsembleClassifierModel import DualEnsembleClassifierModel
from util.util import get_example_length, get_hidden_size, plot_histogram, log


def model_classification_statistics(model_path, trial_lengths_path, output_path, median_value, generate_from_train,
                                    generate_from_cross, generate_from_test, widget):
    log("Started classification statistics: " + str(datetime.datetime.now()), file = None, widget = widget)

    model_directory = model_path
    trial_lengths_directory = trial_lengths_path
    output_directory = output_path

    output_directory = os.path.join(output_directory, "ClassificationStatistics")
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    window_size = int(model_directory.split('_')[-2])
    window_offset = int(model_directory.split('_')[-1])

    example_length = get_example_length(DIVISION_LENGTH, window_size, window_offset)
    stimulus_hidden_size = get_hidden_size(example_length, STIMULUS_OUTPUT_SIZE)
    response_hidden_size = get_hidden_size(example_length, RESPONSE_OUTPUT_SIZE)

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

    model.load_model_from_file(os.path.join(model_directory, f"Training_with_{window_size}_{window_offset}_DUAL.model"))

    response_labels_train = np.fromfile(os.path.join(model_directory, "response_labels_train.dat"), dtype = int)
    response_labels_cross = np.fromfile(os.path.join(model_directory, "response_labels_cross.dat"), dtype = int)
    response_labels_test = np.fromfile(os.path.join(model_directory, "response_labels_test.dat"), dtype = int)

    stimulus_labels_train = np.fromfile(os.path.join(model_directory, "stimulus_labels_train.dat"), dtype = int)
    stimulus_labels_cross = np.fromfile(os.path.join(model_directory, "stimulus_labels_cross.dat"), dtype = int)
    stimulus_labels_test = np.fromfile(os.path.join(model_directory, "stimulus_labels_test.dat"), dtype = int)

    subjects_train = np.fromfile(os.path.join(model_directory, "subjects_train.dat"), dtype = int)
    subjects_cross = np.fromfile(os.path.join(model_directory, "subjects_cross.dat"), dtype = int)
    subjects_test = np.fromfile(os.path.join(model_directory, "subjects_test.dat"), dtype = int)

    trial_index_train = np.fromfile(os.path.join(model_directory, "trial_index_train.dat"), dtype = int)
    trial_index_cross = np.fromfile(os.path.join(model_directory, "trial_index_cross.dat"), dtype = int)
    trial_index_test = np.fromfile(os.path.join(model_directory, "trial_index_test.dat"), dtype = int)

    files_list = []
    for (dirpath, dirnames, filenames) in os.walk(model_directory):
        files_list.extend(filenames)

    stimulus_train_examples = None
    stimulus_cross_examples = None
    stimulus_test_examples = None

    response_train_examples = None
    response_cross_examples = None
    response_test_examples = None
    for file in files_list:
        if "channel_stimulus_train" in file:
            stimulus_train_examples = int(file.split('_')[-2])

        if "channel_stimulus_cross" in file:
            stimulus_cross_examples = int(file.split('_')[-2])

        if "channel_stimulus_test" in file:
            stimulus_test_examples = int(file.split('_')[-2])

        if "channel_response_train" in file:
            response_train_examples = int(file.split('_')[-2])

        if "channel_response_cross" in file:
            response_cross_examples = int(file.split('_')[-2])

        if "channel_response_test" in file:
            response_test_examples = int(file.split('_')[-2])

    channel_stimulus_train = np.fromfile(os.path.join(model_directory,
                                                      f"channel_stimulus_train_{NUMBER_OF_CHANNELS}_{stimulus_train_examples}_{example_length}.dat"))
    channel_stimulus_cross = np.fromfile(os.path.join(model_directory,
                                                      f"channel_stimulus_cross_{NUMBER_OF_CHANNELS}_{stimulus_cross_examples}_{example_length}.dat"))
    channel_stimulus_test = np.fromfile(os.path.join(model_directory,
                                                     f"channel_stimulus_test_{NUMBER_OF_CHANNELS}_{stimulus_test_examples}_{example_length}.dat"))

    channel_response_train = np.fromfile(os.path.join(model_directory,
                                                      f"channel_response_train_{NUMBER_OF_CHANNELS}_{response_train_examples}_{example_length}.dat"))
    channel_response_cross = np.fromfile(os.path.join(model_directory,
                                                      f"channel_response_cross_{NUMBER_OF_CHANNELS}_{response_cross_examples}_{example_length}.dat"))
    channel_response_test = np.fromfile(os.path.join(model_directory,
                                                     f"channel_response_cross_{NUMBER_OF_CHANNELS}_{response_test_examples}_{example_length}.dat"))

    channel_stimulus_train = np.reshape(channel_stimulus_train,
                                        (NUMBER_OF_CHANNELS, stimulus_train_examples, example_length))
    channel_stimulus_cross = np.reshape(channel_stimulus_cross,
                                        (NUMBER_OF_CHANNELS, stimulus_cross_examples, example_length))
    channel_stimulus_test = np.reshape(channel_stimulus_test,
                                       (NUMBER_OF_CHANNELS, stimulus_test_examples, example_length))

    channel_response_train = np.reshape(channel_response_train,
                                        (NUMBER_OF_CHANNELS, response_train_examples, example_length))
    channel_response_cross = np.reshape(channel_response_cross,
                                        (NUMBER_OF_CHANNELS, response_cross_examples, example_length))
    channel_response_test = np.reshape(channel_response_test,
                                       (NUMBER_OF_CHANNELS, response_test_examples, example_length))

    trial_lengths = np.fromfile(os.path.join(trial_lengths_directory, "trial_lengths.dat"), dtype = int)
    trial_lengths = np.reshape(trial_lengths, (11, 180))

    # dual dataset creation
    dual_dataset_train = DatasetForClassificationStatistics(channel_stimulus_train, channel_response_train,
                                                            stimulus_labels_train,
                                                            response_labels_train, subjects_train, trial_index_train)
    dual_dataset_train_loader = DataLoader(dual_dataset_train, batch_size = 1, shuffle = True)

    dual_dataset_cross = DatasetForClassificationStatistics(channel_stimulus_cross, channel_response_cross,
                                                            stimulus_labels_cross,
                                                            response_labels_cross, subjects_cross, trial_index_cross)
    dual_dataset_cross_loader = DataLoader(dual_dataset_cross, batch_size = 1, shuffle = True)

    dual_dataset_test = DatasetForClassificationStatistics(channel_stimulus_test, channel_response_test,
                                                           stimulus_labels_test,
                                                           response_labels_test, subjects_test, trial_index_test)
    dual_dataset_test_loader = DataLoader(dual_dataset_test, batch_size = 1, shuffle = True)

    stimulus_classified = []
    stimulus_misclassified = []
    response_classified = []
    response_misclassified = []

    if generate_from_train:
        train_output_directory = os.path.join(output_directory, 'Train')
        if not os.path.exists(train_output_directory):
            os.makedirs(train_output_directory)
        stimulus_classified_train, stimulus_misclasified_train, response_classified_train, response_misclasified_train = model.predict_for_classification_statistics(
            dual_dataset_train_loader, trial_lengths, STIMULUS_OUTPUT_SIZE, RESPONSE_OUTPUT_SIZE,
            train_output_directory, "train",
            median_value)
        stimulus_classified.extend(stimulus_classified_train)
        stimulus_misclassified.extend(stimulus_misclasified_train)
        response_classified.extend(response_classified_train)
        response_misclassified.extend(response_misclasified_train)

    if generate_from_cross:
        cross_output_directory = os.path.join(output_directory, 'Cross')
        if not os.path.exists(cross_output_directory):
            os.makedirs(cross_output_directory)
        stimulus_classified_cross, stimulus_misclasified_cross, response_classified_cross, response_misclasified_cross = model.predict_for_classification_statistics(
            dual_dataset_cross_loader, trial_lengths, STIMULUS_OUTPUT_SIZE, RESPONSE_OUTPUT_SIZE,
            cross_output_directory, "cross",
            median_value)
        stimulus_classified.extend(stimulus_classified_cross)
        stimulus_misclassified.extend(stimulus_misclasified_cross)
        response_classified.extend(response_classified_cross)
        response_misclassified.extend(response_misclasified_cross)

    if generate_from_test:
        test_output_directory = os.path.join(output_directory, 'Test')
        if not os.path.exists(test_output_directory):
            os.makedirs(test_output_directory)
        stimulus_classified_test, stimulus_misclasified_test, response_classified_test, response_misclasified_test = model.predict_for_classification_statistics(
            dual_dataset_test_loader, trial_lengths, STIMULUS_OUTPUT_SIZE, RESPONSE_OUTPUT_SIZE, test_output_directory,
            "test",
            median_value)
        stimulus_classified.extend(stimulus_classified_test)
        stimulus_misclassified.extend(stimulus_misclasified_test)
        response_classified.extend(response_classified_test)
        response_misclassified.extend(response_misclasified_test)

    aggregated_output_directory = os.path.join(output_directory, 'Aggregated')
    if not os.path.exists(aggregated_output_directory):
        os.makedirs(aggregated_output_directory)
    plot_histogram(stimulus_classified, stimulus_misclassified, 'Stimulus Correctly Classified',
                   'Stimulus Incorrectly Classified', aggregated_output_directory,
                   f"stimulus_classified", median_value)
    plot_histogram(response_classified, response_misclassified, 'Response Correctly Classified',
                   'Response Incorrectly Classified', aggregated_output_directory,
                   f"response_classified", median_value)

    log("Finished classification statistics: " + str(datetime.datetime.now()), file = None, widget = widget)
