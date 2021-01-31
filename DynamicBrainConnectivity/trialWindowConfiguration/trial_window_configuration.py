import os
import datetime

from util.constants import NUMBER_OF_TRIALS, SUBJECT_FILE_PREFIX, SUBJECT_FILE_EVENT_TIMESTAMPS, SUBJECT_FILE_EXTENSION, \
    SUBJECT_FILE_EVENT_CODES, EVENT_CODES_FILTER
from reader.file_reader import read_values_from_binary_file_one_by_one
from trialWindowConfiguration.trial_window_configuration_util import split_trial
from util.util import get_string_from_number, log


def trial_window_configuration(dots_folder_path, output_directory, window_size, window_offset, threshold, widget):
    output_directory = os.path.join(output_directory, 'TrialWindowConfiguration')
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    window_output_directory = os.path.join(output_directory, 'Window')
    if not os.path.exists(window_output_directory):
        os.makedirs(window_output_directory)

    trial_output_directory = os.path.join(output_directory, 'Trial')
    if not os.path.exists(trial_output_directory):
        os.makedirs(trial_output_directory)

    log_file = os.path.join(output_directory, "log.txt")

    log("Started creating files for split configuration for each subject and trial: " + str(datetime.datetime.now()),
        log_file, widget)
    number_of_subjects = 0
    for _, dirnames, filenames in os.walk(dots_folder_path):
        number_of_subjects += len(dirnames)

    for subject_number in range(1, number_of_subjects + 1):
        subject_directory = os.path.join(dots_folder_path, SUBJECT_FILE_PREFIX + get_string_from_number(subject_number))

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

        for trial_number in range(NUMBER_OF_TRIALS):
            trial_start_timestamp = event_code_timestamps[2 * trial_number][0]
            trial_end_timestamp = event_code_timestamps[2 * trial_number + 1][0]

            trial_length = trial_end_timestamp - trial_start_timestamp + 1

            window_file = open(os.path.join(window_output_directory, f'{subject_number}_{trial_number + 1}.txt'), 'w+')
            trial_file = open(os.path.join(trial_output_directory, f'{subject_number}_{trial_number + 1}.txt'), 'w+')

            if trial_length <= threshold:
                split_trial(window_file, trial_start_timestamp, trial_end_timestamp, window_size, window_offset)
                print(f'{trial_start_timestamp} {trial_end_timestamp}', file = trial_file)
            else:
                split_trial(window_file, trial_start_timestamp, trial_start_timestamp + threshold // 2 - 1, window_size,
                            window_offset)
                split_trial(window_file, trial_end_timestamp - threshold // 2 + 1, trial_end_timestamp, window_size,
                            window_offset)

                print(f'{trial_start_timestamp} {trial_start_timestamp + threshold // 2 - 1}', file = trial_file)
                print(f'{trial_end_timestamp - threshold // 2 + 1} {trial_end_timestamp}', file = trial_file)

            window_file.close()
            trial_file.close()
    log("Finished creating files for split configuration for each subject and trial: " + str(datetime.datetime.now()),
        log_file, widget)
