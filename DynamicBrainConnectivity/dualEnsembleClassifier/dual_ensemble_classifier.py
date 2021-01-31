import datetime
import os
import random
from math import sqrt

import numpy as np
import pandas
from torch.utils.data import DataLoader
from visdom import Visdom

from util.constants import NUMBER_OF_CHANNELS, NUMBER_OF_TRIALS, SUBJECT_FILE_PREFIX, TRIAL_DATA, CSV_FILE_EXTENSION, \
    TEXT_FILE_EXTENSION, HTML_FILE_EXTENSION, STIMULUS_OUTPUT_SIZE, \
    RESPONSE_OUTPUT_SIZE
from dualEnsembleClassifier.dual_ensemble_classifier_util import convert_trial_dictionary_to_example
from dualEnsembleClassifier.ml.DualEnsembleClassifierModel import DualEnsembleClassifierModel
from dualEnsembleClassifier.ml.Dataset import TrialDataset
from dualEnsembleClassifier.ml.WeightInitializer import WeightInitializer
from reader.file_reader import read_value_from_binary_file, read_array_from_binary_file
from util.util import get_string_from_number, log

'''
*******************************************************************************************

FILTERED TRIAL DATA PARSING

*******************************************************************************************
'''


def dual_ensemble_classifier(data_directory, csv_path, output_path, window_size, window_offset, division_length,
                             only_two_subjects, with_visdom, should_save_datasets, widget):
    # define paths
    output_folder = os.path.join(output_path, 'DualEnsembleClassifier')
    training_name = "Training_with_" + str(window_size) + "_" + str(window_offset)
    training_path = os.path.join(output_folder,
                                 training_name)
    dual_training_name = training_name + "_DUAL"
    dual_training_name_response = dual_training_name + "_RESPONSE"
    dual_training_name_stimulus = dual_training_name + "_STIMULUS"
    dual_training_name_channels = dual_training_name + "_CHANNELS"

    # define channel list
    channel_list = [x for x in range(NUMBER_OF_CHANNELS)]

    # define stimulus dict
    stimulus_dict = {
        0.05: 0,
        0.1: 1,
        0.15: 2,
        0.2: 3,
        0.25: 4,
        0.3: 5,
    }

    # create output directories
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    if not os.path.exists(training_path):
        os.makedirs(training_path)

    log_file = os.path.join(training_path, training_name + TEXT_FILE_EXTENSION)
    dual_log_file_stimulus_csv = os.path.join(training_path, dual_training_name_stimulus + CSV_FILE_EXTENSION)

    dual_log_file_stimulus_txt = os.path.join(training_path, dual_training_name_stimulus + TEXT_FILE_EXTENSION)

    dual_log_file_response_csv = os.path.join(training_path, dual_training_name_response + CSV_FILE_EXTENSION)

    dual_log_file_response_txt = os.path.join(training_path, dual_training_name_response + TEXT_FILE_EXTENSION)

    dual_log_file_channels_txt = os.path.join(training_path, dual_training_name_channels + TEXT_FILE_EXTENSION)

    dual_log_file_html = os.path.join(training_path, dual_training_name + HTML_FILE_EXTENSION)

    dual_model_path = os.path.join(training_path, dual_training_name + ".model")

    subjects_data = read_trial_data(channel_list, data_directory, log_file, only_two_subjects, widget)

    channel_response_cross, channel_response_test, channel_response_train, channel_stimulus_cross, channel_stimulus_test, \
    channel_stimulus_train, number_of_subjects, response_labels_cross, response_labels_test, response_labels_train, \
    stimulus_labels_cross, stimulus_labels_test, stimulus_labels_train, subjects_cross, subjects_test, subjects_train, \
    trial_index_cross, trial_index_test, trial_index_train = create_dataset(
        channel_list, csv_path, division_length, log_file, only_two_subjects, stimulus_dict, subjects_data, widget,
        window_offset, window_size
    )

    log("Start dataset creation: " + str(datetime.datetime.now()), file = log_file, widget = widget)

    response_classes, stimulus_classes, channel_response_train, channel_stimulus_train, channel_response_cross, \
    channel_stimulus_cross, channel_response_test, channel_stimulus_test = standardize_dataset(channel_list,
                                                                                               channel_response_cross,
                                                                                               channel_response_test,
                                                                                               channel_response_train,
                                                                                               channel_stimulus_cross,
                                                                                               channel_stimulus_test,
                                                                                               channel_stimulus_train,
                                                                                               log_file,
                                                                                               number_of_subjects,
                                                                                               only_two_subjects,
                                                                                               stimulus_dict,
                                                                                               subjects_cross,
                                                                                               subjects_test,
                                                                                               subjects_train, widget)

    # set example length
    example_length = len(channel_stimulus_train[0][0])

    dual_dataset_cross_loader, dual_dataset_test_loader, dual_dataset_train_loader = create_loaders(
        channel_response_cross, channel_response_test, channel_response_train, channel_stimulus_cross,
        channel_stimulus_test, channel_stimulus_train, response_labels_cross, response_labels_test,
        response_labels_train, stimulus_labels_cross, stimulus_labels_test, stimulus_labels_train, subjects_cross,
        subjects_test, subjects_train)

    save_datasets(channel_response_cross, channel_response_test, channel_response_train,
                  channel_stimulus_cross, channel_stimulus_test, channel_stimulus_train,
                  response_labels_cross, response_labels_test, response_labels_train,
                  should_save_datasets, stimulus_labels_cross, stimulus_labels_test,
                  stimulus_labels_train, subjects_cross, subjects_test, subjects_train,
                  training_path, trial_index_cross, trial_index_test, trial_index_train)

    log("End dataset creation: " + str(datetime.datetime.now()), file = log_file, widget = widget)

    initialize_model(channel_list, dual_dataset_cross_loader, dual_dataset_test_loader, dual_dataset_train_loader,
                     dual_log_file_channels_txt, dual_log_file_html, dual_log_file_response_csv,
                     dual_log_file_response_txt, dual_log_file_stimulus_csv, dual_log_file_stimulus_txt,
                     dual_model_path, example_length, log_file, number_of_subjects, response_classes, stimulus_classes,
                     widget, with_visdom)


def initialize_model(channel_list, dual_dataset_cross_loader, dual_dataset_test_loader, dual_dataset_train_loader,
                     dual_log_file_channels_txt, dual_log_file_html, dual_log_file_response_csv,
                     dual_log_file_response_txt, dual_log_file_stimulus_csv, dual_log_file_stimulus_txt,
                     dual_model_path, example_length, log_file, number_of_subjects, response_classes, stimulus_classes,
                     widget, with_visdom):
    # plot to VISDOM if enabled
    viz = None
    if with_visdom:
        viz = Visdom(port = 8097, server = 'http://localhost', base_url = '/')
    # create model
    dual_model = DualEnsembleClassifierModel(
        (
            [example_length,
             int(example_length * 2 / 3 + STIMULUS_OUTPUT_SIZE), STIMULUS_OUTPUT_SIZE],
            [example_length,
             int(example_length * 2 / 3 + RESPONSE_OUTPUT_SIZE), RESPONSE_OUTPUT_SIZE]
        ),
        len(channel_list)
    )
    log(dual_model, file = log_file, widget = None)
    # initialize weights
    weightInit = WeightInitializer()
    weightInit.init_weights(dual_model, 'xavier_normal_', { 'gain': 0.02 })
    log("Started dual training: " + str(datetime.datetime.now()), file = log_file, widget = widget)
    # fit the model
    dual_model.fit(viz, "dual", dual_dataset_train_loader, dual_dataset_cross_loader, dual_log_file_html,
                   number_epochs = 5,
                   learning_rate = 0.001,
                   widget = widget)
    # make the prediction
    dual_model.predict(dual_dataset_test_loader, dual_log_file_stimulus_csv, dual_log_file_stimulus_txt,
                       dual_log_file_response_csv, dual_log_file_response_txt, dual_log_file_channels_txt,
                       stimulus_classes,
                       response_classes, number_of_subjects)
    # save the model to a file
    dual_model.save_model_to_file(dual_model_path)
    log("End dual training: " + str(datetime.datetime.now()), file = log_file, widget = widget)


def save_datasets(channel_response_cross, channel_response_test, channel_response_train, channel_stimulus_cross,
                  channel_stimulus_test, channel_stimulus_train, response_labels_cross, response_labels_test,
                  response_labels_train, should_save_datasets, stimulus_labels_cross, stimulus_labels_test,
                  stimulus_labels_train, subjects_cross, subjects_test, subjects_train, training_path,
                  trial_index_cross, trial_index_test, trial_index_train):
    if should_save_datasets:
        channel_stimulus_train = np.array(channel_stimulus_train)
        channel_stimulus_train.tofile(os.path.join(training_path,
                                                   f"channel_stimulus_train"
                                                   f"_{channel_stimulus_train.shape[0]}"
                                                   f"_{channel_stimulus_train.shape[1]}"
                                                   f"_{channel_stimulus_train.shape[2]}.dat"))
        channel_response_train = np.array(channel_response_train)
        channel_response_train.tofile(os.path.join(training_path,
                                                   f"channel_response_train"
                                                   f"_{channel_response_train.shape[0]}"
                                                   f"_{channel_response_train.shape[1]}"
                                                   f"_{channel_response_train.shape[2]}.dat"))

        stimulus_labels_train = np.array(stimulus_labels_train)
        stimulus_labels_train.tofile(os.path.join(training_path, "stimulus_labels_train.dat"))

        response_labels_train = np.array(response_labels_train)
        response_labels_train.tofile(os.path.join(training_path, "response_labels_train.dat"))

        subjects_train = np.array(subjects_train)
        subjects_train.tofile(os.path.join(training_path, "subjects_train.dat"))

        trial_index_train = np.array(trial_index_train)
        trial_index_train.tofile(os.path.join(training_path, "trial_index_train.dat"))

        channel_stimulus_cross = np.array(channel_stimulus_cross)
        channel_stimulus_cross.tofile(os.path.join(training_path,
                                                   f"channel_stimulus_cross"
                                                   f"_{channel_stimulus_cross.shape[0]}"
                                                   f"_{channel_stimulus_cross.shape[1]}"
                                                   f"_{channel_stimulus_cross.shape[2]}.dat"))
        channel_response_cross = np.array(channel_response_cross)
        channel_response_cross.tofile(os.path.join(training_path,
                                                   f"channel_response_cross"
                                                   f"_{channel_response_cross.shape[0]}"
                                                   f"_{channel_response_cross.shape[1]}"
                                                   f"_{channel_response_cross.shape[2]}.dat"))

        stimulus_labels_cross = np.array(stimulus_labels_cross)
        stimulus_labels_cross.tofile(os.path.join(training_path, "stimulus_labels_cross.dat"))

        response_labels_cross = np.array(response_labels_cross)
        response_labels_cross.tofile(os.path.join(training_path, "response_labels_cross.dat"))

        subjects_cross = np.array(subjects_cross)
        subjects_cross.tofile(os.path.join(training_path, "subjects_cross.dat"))

        trial_index_cross = np.array(trial_index_cross)
        trial_index_cross.tofile(os.path.join(training_path, "trial_index_cross.dat"))

        channel_stimulus_test = np.array(channel_stimulus_test)
        channel_stimulus_test.tofile(os.path.join(training_path,
                                                  f"channel_stimulus_test"
                                                  f"_{channel_stimulus_test.shape[0]}"
                                                  f"_{channel_stimulus_test.shape[1]}"
                                                  f"_{channel_stimulus_test.shape[2]}.dat"))
        channel_response_test = np.array(channel_response_test)
        channel_response_test.tofile(os.path.join(training_path,
                                                  f"channel_response_test"
                                                  f"_{channel_response_test.shape[0]}"
                                                  f"_{channel_response_test.shape[1]}"
                                                  f"_{channel_response_test.shape[2]}.dat"))

        stimulus_labels_test = np.array(stimulus_labels_test)
        stimulus_labels_test.tofile(os.path.join(training_path, "stimulus_labels_test.dat"))

        response_labels_test = np.array(response_labels_test)
        response_labels_test.tofile(os.path.join(training_path, "response_labels_test.dat"))

        subjects_test = np.array(subjects_test)
        subjects_test.tofile(os.path.join(training_path, "subjects_test.dat"))

        trial_index_test = np.array(trial_index_test)
        trial_index_test.tofile(os.path.join(training_path, "trial_index_test.dat"))


def create_loaders(channel_response_cross, channel_response_test, channel_response_train, channel_stimulus_cross,
                   channel_stimulus_test, channel_stimulus_train, response_labels_cross, response_labels_test,
                   response_labels_train, stimulus_labels_cross, stimulus_labels_test, stimulus_labels_train,
                   subjects_cross, subjects_test, subjects_train):
    '''
    *******************************************************************************************

    DATASET CREATION

    *******************************************************************************************
    '''
    # dual dataset creation
    dual_dataset_train = TrialDataset(channel_stimulus_train, channel_response_train, stimulus_labels_train,
                                      response_labels_train, subjects_train)
    dual_dataset_train_loader = DataLoader(dual_dataset_train, batch_size = 32, shuffle = True)
    dual_dataset_cross = TrialDataset(channel_stimulus_cross, channel_response_cross, stimulus_labels_cross,
                                      response_labels_cross, subjects_cross)
    dual_dataset_cross_loader = DataLoader(dual_dataset_cross, batch_size = 32, shuffle = True)
    dual_dataset_test = TrialDataset(channel_stimulus_test, channel_response_test, stimulus_labels_test,
                                     response_labels_test, subjects_test)
    dual_dataset_test_loader = DataLoader(dual_dataset_test, batch_size = 32, shuffle = True)
    return dual_dataset_cross_loader, dual_dataset_test_loader, dual_dataset_train_loader


def standardize_dataset(channel_list, channel_response_cross, channel_response_test, channel_response_train,
                        channel_stimulus_cross, channel_stimulus_test, channel_stimulus_train, log_file,
                        number_of_subjects, only_two_subjects, stimulus_dict, subjects_cross, subjects_test,
                        subjects_train, widget):
    '''
    *******************************************************************************************

    DATASET STANDARDIZATION (done on each channel and on each subject)

    *******************************************************************************************
    '''
    stimulus_classes = list(map(lambda x: str(x), list(stimulus_dict.keys())))
    response_classes = ['seen', 'uncertain', 'not_seen']
    # for each channel
    for channel_index in range(len(channel_list)):

        log(f"Started standardization for channel {channel_index}", file = log_file, widget = widget)

        count = 0
        # for each subject
        for subject in range(1, number_of_subjects + 1):

            data = []

            # concatenate all the values from this channel and subject (only the train)
            for index in range(len(channel_response_train[channel_index])):
                if subjects_train[index] == subject:  # if the example is for the current subject
                    data.extend(channel_response_train[channel_index][index])
                    data.extend(channel_stimulus_train[channel_index][index])

            # compute mean and std
            mean = sum(data) / len(data)
            std = sqrt(sum(list(map(lambda elem: (elem - mean) ** 2, data))) / len(data))

            if std != 0:  # for faulty channels where all the values are 0

                # for all train examples
                for index in range(len(channel_response_train[channel_index])):
                    if subjects_train[index] == subject:  # if the example is for the current subject
                        channel_response_train[channel_index][index] = list(
                            map(lambda elem: (elem - mean) / std, channel_response_train[channel_index][index]))
                        channel_stimulus_train[channel_index][index] = list(
                            map(lambda elem: (elem - mean) / std, channel_stimulus_train[channel_index][index]))

                # for all cross examples
                for index in range(len(channel_response_cross[channel_index])):
                    if subjects_cross[index] == subject:  # if the example is for the current subject
                        channel_response_cross[channel_index][index] = list(
                            map(lambda elem: (elem - mean) / std, channel_response_cross[channel_index][index]))
                        channel_stimulus_cross[channel_index][index] = list(
                            map(lambda elem: (elem - mean) / std, channel_stimulus_cross[channel_index][index]))

                # for all test examples
                for index in range(len(channel_response_test[channel_index])):
                    if subjects_test[index] == subject:  # if the example is for the current subject
                        channel_response_test[channel_index][index] = list(
                            map(lambda elem: (elem - mean) / std, channel_response_test[channel_index][index]))
                        channel_stimulus_test[channel_index][index] = list(
                            map(lambda elem: (elem - mean) / std, channel_stimulus_test[channel_index][index]))

            count += 1
            if count == 2 and only_two_subjects:
                break

        # convert the lists to numpy arrays
        channel_response_train[channel_index] = np.array(channel_response_train[channel_index])
        channel_stimulus_train[channel_index] = np.array(channel_stimulus_train[channel_index])
        channel_response_cross[channel_index] = np.array(channel_response_cross[channel_index])
        channel_stimulus_cross[channel_index] = np.array(channel_stimulus_cross[channel_index])
        channel_response_test[channel_index] = np.array(channel_response_test[channel_index])
        channel_stimulus_test[channel_index] = np.array(channel_stimulus_test[channel_index])
    return response_classes, stimulus_classes, channel_response_train, channel_stimulus_train, channel_response_cross, \
           channel_stimulus_cross, channel_response_test, channel_stimulus_test


def create_dataset(channel_list, csv_path, division_length, log_file, only_two_subjects, stimulus_dict, subjects_data,
                   widget, window_offset, window_size):
    '''
    *******************************************************************************************

    CSV PARSING

    *******************************************************************************************
    '''
    subjects_directories = [x[0] for x in os.walk(csv_path)]
    # eliminate current directory
    subjects_directories = subjects_directories[1:]
    number_of_subjects = len(subjects_directories)
    log("Start time for processing csv: " + str(datetime.datetime.now()), file = log_file, widget = widget)
    # split the trials into train, cross and test
    trial_partition_list_train = []
    trial_partition_list_cross = []
    number_of_trials = NUMBER_OF_TRIALS - 30  # - 30 = without g = 0
    train_length = int(0.6 * number_of_trials)
    cross_length = int(0.2 * number_of_trials)
    for subject in range(number_of_subjects):
        trial_index_list = [(x + 30) for x in range(number_of_trials)]  # - 30 = without g = 0
        random.shuffle(trial_index_list)

        trial_partition_list_train.append(trial_index_list[:train_length])
        trial_partition_list_cross.append(trial_index_list[train_length:cross_length + train_length])
    # define lists where the examples are stored
    channel_stimulus_train = [[] for x in range(len(channel_list))]
    channel_response_train = [[] for x in range(len(channel_list))]
    stimulus_labels_train = []
    response_labels_train = []
    subjects_train = []
    channel_stimulus_cross = [[] for x in range(len(channel_list))]
    channel_response_cross = [[] for x in range(len(channel_list))]
    stimulus_labels_cross = []
    response_labels_cross = []
    subjects_cross = []
    channel_stimulus_test = [[] for x in range(len(channel_list))]
    channel_response_test = [[] for x in range(len(channel_list))]
    stimulus_labels_test = []
    response_labels_test = []
    subjects_test = []
    # create a channel dict which will be used if a subset of channels is given
    channel_dict = { }
    for channel_index in range(len(channel_list)):
        channel_dict[channel_list[channel_index]] = channel_index
    trial_index_train = []
    trial_index_cross = []
    trial_index_test = []
    count = 0
    for subject_directory in subjects_directories:
        log(subject_directory, file = log_file, widget = widget)

        # compute subject number
        subject_number = subjects_directories.index(subject_directory) + 1

        csv_path = SUBJECT_FILE_PREFIX + get_string_from_number(subject_number) + \
                   TRIAL_DATA + CSV_FILE_EXTENSION

        csv_path = os.path.join(subject_directory, csv_path)

        # open csv file
        csvfile = pandas.read_csv(csv_path, encoding = 'utf-8')

        # keep only columns we are interested in
        csvfile = csvfile[['Trial', 'G', 'CorrectResp', 'ResponseID']]

        log("Started csv parsing for subject " + str(subject_number), file = log_file, widget = widget)

        # iterate over trials
        for trial_index in range(0, NUMBER_OF_TRIALS):

            # find the trial from the csv file
            csv_line = csvfile[(csvfile.Trial == trial_index + 1)]

            if csv_line['G'].values[0] == 0:
                continue

            # we will process the same trial for all the channels
            for channel_index in range(0, NUMBER_OF_CHANNELS):

                # use only a channel subset (specified in the channel list)
                if channel_index not in channel_list:
                    continue

                # get trial values from subjects_data
                trial_values = subjects_data[subject_number - 1][channel_index][trial_index]

                # convert data to learning example
                example = convert_trial_dictionary_to_example({
                    'trial_number': trial_index + 1,
                    'g': csv_line['G'].values[0],
                    'stimulus': csv_line['CorrectResp'].values[0],
                    'response': csv_line['ResponseID'].values[0],
                    'values': trial_values,
                    'subject_number': subject_number
                },
                    window_size,
                    window_offset,
                    division_length
                )

                # add channel to example
                example['channel'] = channel_index
                example['subject_number'] = subject_number
                example['trial_number'] = trial_index

                # add example to the correct set
                if trial_index in trial_partition_list_train[subject_number - 1]:  # train
                    channel_stimulus_train[channel_dict[channel_index]].append(example['stimulus_values'])
                    channel_response_train[channel_dict[channel_index]].append(example['response_values'])
                else:
                    if trial_index in trial_partition_list_cross[subject_number - 1]:  # cross
                        channel_stimulus_cross[channel_dict[channel_index]].append(example['stimulus_values'])
                        channel_response_cross[channel_dict[channel_index]].append(example['response_values'])
                    else:  # test
                        channel_stimulus_test[channel_dict[channel_index]].append(example['stimulus_values'])
                        channel_response_test[channel_dict[channel_index]].append(example['response_values'])

            # add example label to the correct set
            if trial_index in trial_partition_list_train[subject_number - 1]:  # train
                stimulus_labels_train.append(stimulus_dict[csv_line['G'].values[0]])
                response_labels_train.append(csv_line['ResponseID'].values[0] - 1)
                subjects_train.append(subject_number)
                trial_index_train.append(trial_index)
            else:
                if trial_index in trial_partition_list_cross[subject_number - 1]:  # cross
                    stimulus_labels_cross.append(stimulus_dict[csv_line['G'].values[0]])
                    response_labels_cross.append(csv_line['ResponseID'].values[0] - 1)
                    subjects_cross.append(subject_number)
                    trial_index_cross.append(trial_index)
                else:  # test
                    stimulus_labels_test.append(stimulus_dict[csv_line['G'].values[0]])
                    response_labels_test.append(csv_line['ResponseID'].values[0] - 1)
                    subjects_test.append(subject_number)
                    trial_index_test.append(trial_index)

        subjects_data[subject_number - 1] = []
        log("Finished csv parsing for subject " + str(subject_number) + " at time " + str(datetime.datetime.now()),
            file = log_file, widget = widget)

        count += 1
        if only_two_subjects and count == 2:
            break
    log("End time for processing csv: " + str(datetime.datetime.now()), file = log_file, widget = widget)
    return channel_response_cross, channel_response_test, channel_response_train, channel_stimulus_cross, channel_stimulus_test, channel_stimulus_train, number_of_subjects, response_labels_cross, response_labels_test, response_labels_train, stimulus_labels_cross, stimulus_labels_test, stimulus_labels_train, subjects_cross, subjects_test, subjects_train, trial_index_cross, trial_index_test, trial_index_train


def read_trial_data(channel_list, data_directory, log_file, only_two_subjects, widget):
    count = 0
    file_names = []
    for (dirpath, dirnames, filenames) in os.walk(data_directory):
        file_names.extend(filenames)
        count += 1
        if only_two_subjects and count == 2:
            break
    # compute number of subjects
    subject_number = len(file_names)
    subjects_data = []
    for x in range(0, subject_number):
        subjects_data.append([])
    log('Start time of parsing: ' + str(datetime.datetime.now()), file = log_file, widget = widget)
    count = 0
    # parse the file for each subject
    for file_name in file_names:

        # get subject's file
        subject_file = os.path.join(data_directory, file_name)

        # compute subject's number
        subject_number = int(file_name.split('.')[0]) - 1

        log('Started reading filtered data for subject ' + str(subject_number + 1), file = log_file, widget = widget)

        # read a subject's data trial by trial
        # differentiate between channels knowing that there are 210 trials per channel
        with open(subject_file, 'rb') as file:

            # iterate over channels
            for channel_index in range(0, NUMBER_OF_CHANNELS):
                subjects_data[subject_number].append([])

                # iterate over trials
                for trial_index in range(0, NUMBER_OF_TRIALS):
                    # read trial's length
                    trial_length = read_value_from_binary_file(file, 'f', 4)

                    # read trial's values
                    trial_values = read_array_from_binary_file(file, 'f', 4, int(trial_length))

                    # keep only channel A23
                    if channel_index in channel_list:
                        subjects_data[subject_number][channel_index].append(list(trial_values))
        log('Finished reading filtered data for subject ' + str(subject_number + 1), file = log_file, widget = widget)
        count += 1
        if only_two_subjects and count == 2:
            break
    log('End time of parsing: ' + str(datetime.datetime.now()), file = log_file, widget = widget)
    return subjects_data
