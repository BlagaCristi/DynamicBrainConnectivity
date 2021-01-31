# class used to parse the raw data file of a subject
import os
import threading

from util.constants import SUBJECT_FILE_PREFIX, SUBJECT_FILE_EVENT_TIMESTAMPS, SUBJECT_FILE_EXTENSION, \
    SUBJECT_FILE_EVENT_CODES, EVENT_CODES_FILTER, NUMBER_OF_CHANNELS, SUBJECT_FILE_CHANNEL
from reader.file_reader import read_values_from_binary_file_one_by_one, read_array_from_unopened_binary_file, \
    write_to_binary_file
from util.util import log, get_string_from_number


class ParsingThread(threading.Thread):
    def __init__(self, threadID, name, subject_directory, subject_number, output_directory, trial_filter_length,
                 widget):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.subject_directory = subject_directory
        self.subject_number = subject_number
        self.output_directory = output_directory
        self.trial_filter_length = trial_filter_length
        self.widget = widget

    def run(self):
        log("Starting " + self.name, None, self.widget)
        self.thread_function()
        log("Exiting " + self.name, None, self.widget)

    def thread_function(self):
        # parse event timestamps file for current subject
        event_timestamp_file_path = SUBJECT_FILE_PREFIX + get_string_from_number(self.subject_number) + \
                                    SUBJECT_FILE_EVENT_TIMESTAMPS + SUBJECT_FILE_EXTENSION

        event_timestamp_file_path = os.path.join(self.subject_directory, event_timestamp_file_path)

        timestamps = read_values_from_binary_file_one_by_one(event_timestamp_file_path, 'i', 4)

        # parse event codes file for current subject
        event_codes_file_path = SUBJECT_FILE_PREFIX + get_string_from_number(self.subject_number) + \
                                SUBJECT_FILE_EVENT_CODES + SUBJECT_FILE_EXTENSION

        event_codes_file_path = os.path.join(self.subject_directory, event_codes_file_path)

        event_codes = read_values_from_binary_file_one_by_one(event_codes_file_path, 'i', 4)

        # create a list of tuples where we attach to each event code its corresponding timestamp
        # structure: [...,(timestamp, event), ...]
        event_code_timestamps = list(zip(timestamps, event_codes))

        # filter out the events we don't need
        event_code_timestamps = list(filter(lambda event_code_timestamp: event_code_timestamp[1] in EVENT_CODES_FILTER,
                                            event_code_timestamps))

        # get the minimum length of a trial specified using an environment variable
        trial_minimum_length = self.trial_filter_length

        channel_filtered_values = []
        channel_filtered_values_length = 0

        # split channels for subject
        for channel_index in range(1, NUMBER_OF_CHANNELS + 1):
            channel_file_path = SUBJECT_FILE_PREFIX + get_string_from_number(self.subject_number) + \
                                SUBJECT_FILE_CHANNEL + get_string_from_number(channel_index) + SUBJECT_FILE_EXTENSION
            channel_file_path = os.path.join(self.subject_directory, channel_file_path)

            # get file size that indicates how many values are in the file
            file_size_bytes = os.stat(channel_file_path).st_size

            channel_values = read_array_from_unopened_binary_file(channel_file_path, 'f', 4, file_size_bytes // 4)

            # divide channels based on trials using the event_code_timestamps
            index = 0
            while index < len(event_code_timestamps):
                # a trial is represented by 2 consecutive events
                # in order to extract a trial's value from the channel between stimulus and response
                # we must use the timestamps of the two consecutive events
                # these timestamps can be used as indices in the channel_values list

                trial_start = event_code_timestamps[index][0]
                trial_end = event_code_timestamps[index + 1][0]

                # extract the filtered trial values from the channel between the two consecutive events:
                # stimulus and response
                trial_values = channel_values[
                               trial_start: (trial_end + 1)]
                trial_length = trial_end - trial_start + 1

                # the minimum length is specified twice because the windows are computed both from left and right
                if trial_length >= 2 * trial_minimum_length:

                    channel_filtered_values_length += 2 * trial_minimum_length + 1

                    # save the trial's filtered values by first saving the length of the trial
                    channel_filtered_values.append(float(2 * trial_minimum_length))

                    # afterwards, save the values of the trial
                    channel_filtered_values.extend(
                        trial_values[:trial_minimum_length] +
                        trial_values[-trial_minimum_length:]
                    )
                else:
                    channel_filtered_values_length += trial_length + 1

                    # save the trial's filtered values by first saving the length of the trial
                    channel_filtered_values.append(float(trial_length))

                    # afterwards, save the values of the trial
                    channel_filtered_values.extend(trial_values)

                # index is incremented with 2 because we process 2 events at once
                index += 2

        # write the data of the subject to a file
        # there will be 128 channels, each channel being formed of 210 trials, each trial having its length
        # and its values in the file
        write_to_binary_file(self.subject_number, channel_filtered_values, channel_filtered_values_length,
                             self.output_directory)
