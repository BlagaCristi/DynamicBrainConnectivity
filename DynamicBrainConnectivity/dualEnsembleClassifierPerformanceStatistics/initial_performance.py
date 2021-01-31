import os

import matplotlib.pyplot as plt
import pandas as pd

from util.util import log


def plot_initial_performance(data_directory, output_directory, widget):
    # open data directory
    division_directories = [x[0] for x in os.walk(data_directory)][1:]

    # create log file
    log_file = os.path.join(output_directory, 'Trial_Division_Statistics.txt')

    """

    RESPONSE PERFORMANCE

    """

    # dictionary for response performance
    division_response = { }

    # for each division
    for division_directory in division_directories:
        division_name = division_directory.split('\\')[-1:][0]

        response_csv_file = os.path.join(division_directory, division_name + '_DUAL_RESPONSE.csv')

        # open csv
        response_df = pd.read_csv(response_csv_file)

        # keep only the f1-score
        response_df = response_df['f1-score']

        # drop the last 3 rows because they are no use for us
        response_df = response_df.drop([3, 4, 5], axis = 0)

        # save division performance
        division_response[division_name] = { }
        division_response[division_name]['seen'] = response_df[0]
        division_response[division_name]['uncertain'] = response_df[1]
        division_response[division_name]['not_seen'] = response_df[2]

    # plot response
    count = 1
    division_list_response = []
    points_response = []

    # compute points to be plotted (one point = average over classes)
    for division in list(division_response.keys()):
        points_response.append(
            [count,
             (division_response[division]['seen'] + division_response[division]['uncertain'] +
              division_response[division][
                  'not_seen']) / 3])
        division_list_response.append(division.split('Training_with_')[1])
        count += 1

    # get minimum and maximum for this point
    minimum = min(points_response, key = lambda value: value[1])
    maximum = max(points_response, key = lambda value: value[1])
    index_max = points_response.index(maximum)

    # print response performance statistic
    log("***********************************************************", log_file, widget)
    log("Response statistic", log_file, widget)
    log("Best configuration:", log_file, widget)
    log("Window size: " + str(division_list_response[index_max].split('_')[0]), log_file, widget)
    log("Window offset: " + str(division_list_response[index_max].split('_')[1]), log_file, widget)
    log('', log_file, widget)
    log("Average f1-score: " + str(points_response[index_max][1]), log_file, widget)
    log('', log_file, widget)
    log('', log_file, widget)

    # plot the points
    for i in range(len(points_response)):
        x = points_response[i][0]
        y = points_response[i][1]
        plt.plot(x, y, 'bo')
        plt.text(x * 1.01, y * 1.01, division_list_response[i], fontsize = 8)

    plt.xlim((0, len(points_response) + 1))
    plt.ylim((minimum[1] - 0.005, maximum[1] + 0.02))
    plt.title('Response f1-score mean per classes')
    plt.xlabel('Division')
    plt.ylabel('f1-score')
    plt.tight_layout()
    plt.savefig(os.path.join(output_directory, 'Response_statistics' + ".png"), bbox_inches = "tight")
    plt.close()

    """

    STIMULUS PERFORMANCE

    """

    # dictionary for stimulus division
    division_stimulus = { }

    # for each division
    for division_directory in division_directories:

        division_name = division_directory.split('\\')[-1:][0]
        stimulus_csv_file = os.path.join(division_directory, division_name + '_DUAL_STIMULUS.csv')

        # open stimulus csv
        stimulus_df = pd.read_csv(stimulus_csv_file)
        stimulus_df = stimulus_df['f1-score']

        number_of_rows = stimulus_df.shape[0]

        # drop the last 3 rows because they are no use for us
        stimulus_df = stimulus_df.drop([
            number_of_rows - 1,
            number_of_rows - 2,
            number_of_rows - 3
        ], axis = 0)

        number_of_rows = number_of_rows - 3

        division_stimulus[division_name] = 0

        # compute average of f1- score amongst divisions
        for row in stimulus_df:
            division_stimulus[division_name] += row

        division_stimulus[division_name] /= number_of_rows

    # plot stimulus
    count = 1
    division_list_stimulus = []
    points_stimulus = []

    # for each division
    for division in list(division_stimulus.keys()):
        # a point is represented by a count and stimulus f1-score
        points_stimulus.append([count, division_stimulus[division]])
        division_list_stimulus.append(division.split('Training_with_')[1])
        count += 1

    # get minimum and maximum
    minimum = min(points_stimulus, key = lambda value: value[1])
    maximum = max(points_stimulus, key = lambda value: value[1])
    index_max = points_stimulus.index(maximum)

    # print statistics
    log("***********************************************************", log_file, widget)
    log("Stimulus statistic", log_file, widget)
    log("Best configuration:", log_file, widget)
    log("Window size: " + str(division_list_stimulus[index_max].split('_')[0]), log_file, widget)
    log("Window offset: " + str(division_list_stimulus[index_max].split('_')[1]), log_file, widget)
    log('', log_file, widget)
    log("Average f1-score: " + str(points_stimulus[index_max][1]), log_file, widget)

    # plot points
    for i in range(len(points_stimulus)):
        x = points_stimulus[i][0]
        y = points_stimulus[i][1]
        plt.plot(x, y, 'bo')
        plt.text(x * 1.01, y * 1.01, division_list_stimulus[i], fontsize = 8)

    plt.xlim((0, len(points_stimulus) + 1))
    plt.ylim((minimum[1] - 0.005, maximum[1] + 0.02))
    plt.title('Stimulus f1-score mean per classes')
    plt.xlabel('Division')
    plt.ylabel('f1-score')
    plt.savefig(os.path.join(output_directory, 'Stimulus_statistics' + ".png"), bbox_inches = "tight")
    plt.close()

    """

    COMBINED PERFORMANCE

    """

    # the two performances are in the same order

    points_combined = []
    for count in range(len(points_response)):
        # a point has x as response f1-score and y as stimulus f1-score
        points_combined.append([
            points_response[count][1],
            points_stimulus[count][1]
        ])

    # get minimum and maximum for both axes
    minimum_x = min(points_combined, key = lambda x: x[0])
    maximum_x = max(points_combined, key = lambda x: x[0])
    minimum_y = min(points_combined, key = lambda x: x[1])
    maximum_y = max(points_combined, key = lambda x: x[1])

    # plot the points
    for i in range(len(points_combined)):
        x = points_combined[i][0]
        y = points_combined[i][1]
        plt.plot(x, y, 'bo')
        plt.text(x * 1.005, y * 1.005, division_list_stimulus[i], fontsize = 8)

    plt.xlim((minimum_x[0] - 0.005, maximum_x[0] + 0.005))
    plt.ylim((minimum_y[1] - 0.005, maximum_y[1] + 0.005))
    plt.title('Stimulus/Response f1-score mean per classes')
    plt.xlabel('Response f1-score')
    plt.ylabel('Stimulus f1-score')
    plt.savefig(os.path.join(output_directory, 'Stimulus_response_statistics' + ".png"), bbox_inches = "tight")
    plt.close()

    """

    UPPER LEFT CORNER FILTERING FOR COMBINED PERFORMANCE

    """

    # filtered based on the response mean and stimulus mean
    average_response = sum(list(map(lambda x: x[0], points_combined))) / len(points_combined)
    average_stimulus = sum(list(map(lambda x: x[1], points_combined))) / len(points_combined)

    filtered_list = list(filter(lambda x: x[0][0] > average_response and x[0][1] > average_stimulus,
                                zip(points_combined, division_list_stimulus)))

    # get minimum and maximum for both axes
    minimum_x = min(filtered_list, key = lambda x: x[0][0])
    maximum_x = max(filtered_list, key = lambda x: x[0][0])
    minimum_y = min(filtered_list, key = lambda x: x[0][1])
    maximum_y = max(filtered_list, key = lambda x: x[0][1])

    # plot the points
    for i in range(len(filtered_list)):
        x = filtered_list[i][0][0]
        y = filtered_list[i][0][1]
        plt.plot(x, y, 'bo')
        plt.text(x * 1.005, y * 1.005, filtered_list[i][1], fontsize = 8)

    plt.xlim((minimum_x[0][0] - 0.005, maximum_x[0][0] + 0.005))
    plt.ylim((minimum_y[0][1] - 0.005, maximum_y[0][1] + 0.005))
    plt.title('Stimulus/Response f1-score mean per classes ~ filtered')
    plt.xlabel('Response f1-score')
    plt.ylabel('Stimulus f1-score')
    plt.savefig(os.path.join(output_directory, 'Stimulus_response_statistics_filtered' + ".png"), bbox_inches = "tight")
    plt.close()
