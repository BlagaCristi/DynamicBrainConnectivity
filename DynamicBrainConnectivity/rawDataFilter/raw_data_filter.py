import datetime
import os
import time

from rawDataFilter.parsing_thread import ParsingThread
from util.util import log


def raw_data_filter(data_directory, output_directory, degree_of_parallelism, trial_filter_length, widget):
    # find the subject's directories
    subjects_directories = [x[0] for x in os.walk(data_directory)]

    # eliminate current directory
    subjects_directories = subjects_directories[1:]

    subject_threads = []

    log("Start time: " + str(datetime.datetime.now()), None, widget)
    start_time = time.time()

    for subject_directory in subjects_directories:
        # compute subject number
        subject_number = subjects_directories.index(subject_directory) + 1

        # create thread
        # specify the subject directory and subject number
        subject_thread = ParsingThread(subject_number, "thread-" + str(subject_number), subject_directory,
                                       subject_number, output_directory, trial_filter_length, widget)

        # start thread
        subject_thread.start()
        subject_threads.append(subject_thread)

        # create a number of threads equal to the degree of parallelism
        if subject_number % degree_of_parallelism == 0:
            # wait for threads to finish
            for subject_thread in subject_threads:
                subject_thread.join()
            subject_threads = []

    log("--- %s seconds ---" % (time.time() - start_time), None, widget)
    log("End time: " + str(datetime.datetime.now()), None, widget)
