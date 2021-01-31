import datetime
import os

import numpy as np
import pandas as pd

from datasetStatistics.dataset_statistics_util import show_histogram_dataset_statistics
from reader.file_reader import read_values_from_binary_file_one_by_one
from util.constants import SUBJECT_FILE_PREFIX, SUBJECT_FILE_EVENT_TIMESTAMPS, SUBJECT_FILE_EXTENSION, \
    SUBJECT_FILE_EVENT_CODES, EVENT_CODES_FILTER
from util.util import get_string_from_number, log


def dataset_statistics(data_directory, output_directory, widget):
    # create output directory
    output_directory = os.path.join(output_directory, 'DatasetStatistics')
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # create output log file
    output_file = os.path.join(output_directory, 'DatasetStatistic.txt')

    log("Started dataset statistics: " + str(datetime.datetime.now()), file = output_file, widget = widget)

    number_of_subjects, subjects_trial_lengths, subjects_trial_response, trial_lengths = read_trial_metadata(
        data_directory)

    compute_statistics(number_of_subjects, output_directory, output_file, subjects_trial_lengths,
                       subjects_trial_response, trial_lengths, widget)

    log("Finished dataset statistics: " + str(datetime.datetime.now()), file = output_file, widget = widget)


def compute_statistics(number_of_subjects, output_directory, output_file, subjects_trial_lengths,
                       subjects_trial_response, trial_lengths, widget):
    log("Number of trials: ", output_file, widget)
    log(len(trial_lengths), output_file, widget)
    log("Minimum value: ", output_file, widget)
    log(min(trial_lengths), output_file, widget)
    log("Maximum value: ", output_file, widget)
    log(max(trial_lengths), output_file, widget)
    # show unfiltered histogram
    show_histogram_dataset_statistics(trial_lengths, "Initial_histogram", output_directory, bin_size = 500,
                                      median_value = sum(trial_lengths) / len(trial_lengths))
    # show filtered histogram
    trial_lengths_filtered = list(filter(lambda value: value < 1000, trial_lengths))
    show_histogram_dataset_statistics(trial_lengths_filtered, "Filtered_histogram", output_directory, bin_size = 100)
    log(f'Trial count smaller than 1000ms: {len(trial_lengths_filtered)}', output_file, widget)
    trial_lengths_np = np.array(trial_lengths)
    trial_lengths_np.tofile(os.path.join(output_directory, "trial_lengths.dat"))
    trial_lengths_mean = sum(trial_lengths) / len(trial_lengths)
    trial_lengths_smaller_than_mean = len(list(filter(lambda value: value <= trial_lengths_mean, trial_lengths)))
    trial_lengths_greater_than_mean = len(trial_lengths) - trial_lengths_smaller_than_mean
    log(f'Trial lengths mean: {trial_lengths_mean}', output_file, widget)
    log(f'Trial lengths count smaller than mean: {trial_lengths_smaller_than_mean}', output_file, widget)
    log(f'Trial lengths count greater than mean: {trial_lengths_greater_than_mean}', output_file, widget)
    trial_lengths_sorted = sorted(trial_lengths)
    trial_lengths_median = trial_lengths_sorted[len(trial_lengths) // 2]
    log(f'Trial lengths median: {trial_lengths_median}', output_file, widget)
    stimulus_dict = {
        0: '0.05',
        1: '0.10',
        2: '0.15',
        3: '0.20',
        4: '0.25',
        5: '0.30'
    }
    response_dict = {
        1: 'seen',
        2: 'uncertain',
        3: 'not_seen'
    }
    subject_plotted = []
    subjects_folder = []
    for subject_index in range(number_of_subjects):
        subject_plotted.append(False)
        subject_folder_name = os.path.join(output_directory, f'Subject_{subject_index + 1}')

        if not os.path.exists(subject_folder_name):
            os.makedirs(subject_folder_name)

        subjects_folder.append(subject_folder_name)

        subject_response_folder_name = os.path.join(subject_folder_name, 'Response')
        subject_stimulus_folder_name = os.path.join(subject_folder_name, 'Stimulus')

        if not os.path.exists(subject_response_folder_name):
            os.makedirs(subject_response_folder_name)

        if not os.path.exists(subject_stimulus_folder_name):
            os.makedirs(subject_stimulus_folder_name)
    generic_stimulus_folder = os.path.join(output_directory, 'Stimulus')
    if not os.path.exists(generic_stimulus_folder):
        os.makedirs(generic_stimulus_folder)
    generic_response_folder = os.path.join(output_directory, 'Response')
    if not os.path.exists(generic_response_folder):
        os.makedirs(generic_response_folder)
    log("Started statistics on stimulus: " + str(datetime.datetime.now()), file = output_file, widget = widget)
    for key in list(stimulus_dict.keys()):
        stimulus_class = []
        for subject_index, subject_trial_lengths in enumerate(subjects_trial_lengths):
            stimulus_class.extend(subject_trial_lengths[key * 30: (key + 1) * 30])
            show_histogram_dataset_statistics(subject_trial_lengths[key * 30: (key + 1) * 30],
                                              f'Subject_{subject_index + 1}_Stimulus_{stimulus_dict[key]}_histogram',
                                              os.path.join(subjects_folder[subject_index], 'Stimulus'),
                                              median_value = trial_lengths_mean)
            if not subject_plotted[subject_index]:
                show_histogram_dataset_statistics(subject_trial_lengths, f'Subject_{subject_index + 1}_histogram',
                                                  subjects_folder[subject_index],
                                                  median_value = trial_lengths_mean)
                subject_plotted[subject_index] = True

        show_histogram_dataset_statistics(stimulus_class, f'Stimulus_{stimulus_dict[key]}_histogram',
                                          generic_stimulus_folder,
                                          median_value = trial_lengths_mean)
    log("Started statistics on response: " + str(datetime.datetime.now()), file = output_file, widget = widget)
    for key in list(response_dict.keys()):
        response_class = []
        for subject_index, subject_trial_lengths in enumerate(subjects_trial_lengths):
            subject_response = []
            for index in range(len(subject_trial_lengths)):
                if subjects_trial_response[subject_index][index] == key:
                    subject_response.append(subject_trial_lengths[index])
            show_histogram_dataset_statistics(subject_response,
                                              f'Subject_{subject_index + 1}_Response_{response_dict[key]}_histogram',
                                              os.path.join(subjects_folder[subject_index], 'Response'),
                                              median_value = trial_lengths_mean)
            response_class.extend(subject_response)

        show_histogram_dataset_statistics(response_class, f'Response_{response_dict[key]}_histogram',
                                          generic_response_folder,
                                          median_value = trial_lengths_mean)

        log(f'Class {response_dict[key]}: {len(response_class)} trials', file = output_file, widget = widget)


def read_trial_metadata(data_directory):
    # get the list of root directories for each subject
    subjects_directories = [x[0] for x in os.walk(data_directory)]
    # eliminate current directory
    subjects_directories = subjects_directories[1:]
    number_of_subjects = len(subjects_directories)
    trial_lengths = []
    subjects_trial_lengths = []
    subjects_trial_response = []
    for subject_directory in subjects_directories:
        # compute subject number
        subject_number = subjects_directories.index(subject_directory) + 1

        # construct event timestamp file name for the current subject
        event_timestamp_file_path = SUBJECT_FILE_PREFIX + get_string_from_number(subject_number) + \
                                    SUBJECT_FILE_EVENT_TIMESTAMPS + SUBJECT_FILE_EXTENSION

        # construct full path for the current student's event timestamp file
        event_timestamp_file_path = os.path.join(subject_directory, event_timestamp_file_path)

        # read the timestamps for the current subject
        timestamps = read_values_from_binary_file_one_by_one(event_timestamp_file_path, 'i', 4)

        # construct event code file name for the current subject
        event_codes_file_path = SUBJECT_FILE_PREFIX + get_string_from_number(subject_number) + \
                                SUBJECT_FILE_EVENT_CODES + SUBJECT_FILE_EXTENSION

        # construct full path for the current subject's event codes file
        event_codes_file_path = os.path.join(subject_directory, event_codes_file_path)

        # read the event codes for the current subject
        event_codes = read_values_from_binary_file_one_by_one(event_codes_file_path, 'i', 4)

        # create a list of tuples where we attach to each event code its corresponding timestamp
        # structure: [...,(timestamp, event), ...]
        event_code_timestamps = list(zip(timestamps, event_codes))

        # filter out the events we don't need
        event_code_timestamps = list(filter(lambda event_code_timestamp: event_code_timestamp[1] in EVENT_CODES_FILTER,
                                            event_code_timestamps))

        csv_file_name = SUBJECT_FILE_PREFIX + get_string_from_number(subject_number) + '-Trial-Data.csv'
        csv_file_path = os.path.join(subject_directory, csv_file_name)
        csv_file = pd.read_csv(csv_file_path)

        subjects_trial_response.append(csv_file['ResponseID'].values[30:])

        subject_trial_lengths = []
        # construct the list containing the length of each trial for the current subject
        for event_code_timestamp_index in range(0, len(event_code_timestamps), 2):
            subject_trial_lengths.append(
                event_code_timestamps[event_code_timestamp_index + 1][0] -
                event_code_timestamps[event_code_timestamp_index][0] + 1
            )

        # eliminate g = 0
        subject_trial_lengths = subject_trial_lengths[30:]

        trial_lengths.extend(subject_trial_lengths)
        subjects_trial_lengths.append(subject_trial_lengths)
    return number_of_subjects, subjects_trial_lengths, subjects_trial_response, trial_lengths
