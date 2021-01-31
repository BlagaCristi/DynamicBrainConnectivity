import os
from math import sqrt

import pandas as pd
import plotly
import plotly.figure_factory as ff
import plotly.graph_objs as go
from plotly.subplots import make_subplots

from util.constants import COLOR_LIST_DISTRIBUTION_PLOTS
from dualEnsembleClassifierPerformanceStatistics.dual_ensemble_classifier_performance_statistics_util import \
    generate_distribution
from util.util import log


def plot_distribution_for_multiple_runs(multiple_runs_directory, output_directory, widget):
    # open multiple runs path
    division_directories = os.listdir(multiple_runs_directory)

    # create log file
    log_file = os.path.join(output_directory, 'Multiple_runs_statistics.txt.txt')

    multiple_runs_dict = { }

    # for each configuration
    for directory in division_directories:
        directory = os.path.join(multiple_runs_directory, directory)
        division_name = directory.split('\\')[-1:][0]
        multiple_runs_dict[division_name] = { }
        multiple_runs_dict[division_name]['avg_response_list'] = []
        multiple_runs_dict[division_name]['avg_stimulus_list'] = []

        # for each run
        runs_directories = [x[0] for x in os.walk(os.path.join(multiple_runs_directory, directory))][1:]
        for runs_directory in runs_directories:
            response_csv_file = os.path.join(runs_directory, division_name + '_DUAL_RESPONSE.csv')
            stimulus_csv_file = os.path.join(runs_directory, division_name + '_DUAL_STIMULUS.csv')

            # open csv
            response_df = pd.read_csv(response_csv_file)

            # keep only the f1-score
            response_df = response_df['f1-score']

            # drop the last 3 rows because they are no use for us
            response_df = response_df.drop([3, 4, 5], axis = 0)

            # add response average
            multiple_runs_dict[division_name]['avg_response_list'].append(
                (response_df[0] + response_df[1] + response_df[2]) / 3)

            # open csv
            stimulus_df = pd.read_csv(stimulus_csv_file)

            # keep only the f1-score
            stimulus_df = stimulus_df['f1-score']

            # find number of rows
            number_of_rows = stimulus_df.shape[0]

            # drop the last 3 rows because they are no use for us
            stimulus_df = stimulus_df.drop([
                number_of_rows - 1,
                number_of_rows - 2,
                number_of_rows - 3
            ], axis = 0)

            # number of classes
            number_of_rows = number_of_rows - 3

            average = 0

            # compute average of f1- score amongst divisions
            for row in stimulus_df:
                average += row

            average /= number_of_rows

            # add stimulus average
            multiple_runs_dict[division_name]['avg_stimulus_list'].append(average)

        # compute response mean for current configuration
        mean_response = sum(multiple_runs_dict[division_name]['avg_response_list']) / len(
            multiple_runs_dict[division_name]['avg_response_list'])

        # compute response std for current configuration
        std_response = sqrt(sum(list(
            map(lambda x: (x - mean_response) ** 2, multiple_runs_dict[division_name]['avg_response_list']))) / len(
            multiple_runs_dict[division_name]['avg_response_list']))

        # compute stimulus mean for current configuration
        mean_stimulus = sum(multiple_runs_dict[division_name]['avg_stimulus_list']) / len(
            multiple_runs_dict[division_name]['avg_stimulus_list'])

        # compute stimulus std for current configuration
        std_stimulus = sqrt(sum(list(
            map(lambda x: (x - mean_stimulus) ** 2, multiple_runs_dict[division_name]['avg_stimulus_list']))) / len(
            multiple_runs_dict[division_name]['avg_stimulus_list']))

        # save parameters
        multiple_runs_dict[division_name]['mean_response'] = mean_response
        multiple_runs_dict[division_name]['std_response'] = std_response

        multiple_runs_dict[division_name]['mean_stimulus'] = mean_stimulus
        multiple_runs_dict[division_name]['std_stimulus'] = std_stimulus

    # generate distributions to be plotted
    response_distributions = []
    stimulus_distributions = []
    for key in multiple_runs_dict.keys():
        response_distributions.append(
            generate_distribution(multiple_runs_dict[key]['std_response'], multiple_runs_dict[key]['mean_response']))
        stimulus_distributions.append(
            generate_distribution(multiple_runs_dict[key]['std_stimulus'], multiple_runs_dict[key]['mean_stimulus']))

    # create a figure
    fig = make_subplots(rows = 2, cols = 1, subplot_titles = ('Stimulus', 'Response'))

    # create distribution plots for response
    distribution_plot = ff.create_distplot(stimulus_distributions, list(multiple_runs_dict.keys()), show_hist = False)

    # for each configuration (for STIMULUS)
    count = 0
    for name in list(multiple_runs_dict.keys()):
        # plot distribution
        fig.add_trace(
            go.Scatter(
                distribution_plot['data'][count],
                name = name.split('_')[-2] + '_' + name.split('_')[-1],
                line = dict(
                    color = COLOR_LIST_DISTRIBUTION_PLOTS[count]
                )
            ),
            row = 1,
            col = 1
        )

        # plot distribution mean
        fig.add_trace(
            go.Scatter(
                x = [multiple_runs_dict[name]['mean_stimulus'], multiple_runs_dict[name]['mean_stimulus']],
                y = [0, max(distribution_plot['data'][count].y)],
                mode = 'lines+markers',
                name = 'Mean ' + name.split('_')[-2] + '_' + name.split('_')[-1],
                line = dict(
                    color = COLOR_LIST_DISTRIBUTION_PLOTS[count]
                )
            ),
            row = 1,
            col = 1
        )

        # plot distribution std
        fig.add_trace(
            go.Scatter(
                x = [multiple_runs_dict[name]['mean_stimulus'] - multiple_runs_dict[name]['std_stimulus'],
                     multiple_runs_dict[name]['mean_stimulus'] + multiple_runs_dict[name]['std_stimulus']],
                y = [count * 0.5, count * 0.5],
                mode = 'lines+markers',
                name = 'Std ' + name.split('_')[-2] + '_' + name.split('_')[-1],
                line = dict(
                    color = COLOR_LIST_DISTRIBUTION_PLOTS[count]
                )
            ),
            row = 1,
            col = 1
        )
        count += 1

    # create distribution plots for response
    distribution_plot = ff.create_distplot(response_distributions, list(multiple_runs_dict.keys()), show_hist = False)

    # for each configuration (for STIMULUS)
    count = 0
    for name in list(multiple_runs_dict.keys()):
        # plot distribution
        fig.add_trace(
            go.Scatter(
                distribution_plot['data'][count],
                name = name.split('_')[-2] + '_' + name.split('_')[-1],
                line = dict(
                    color = COLOR_LIST_DISTRIBUTION_PLOTS[count]
                )
            ),
            row = 2,
            col = 1
        )

        # plot distribution mean
        fig.add_trace(
            go.Scatter(
                x = [multiple_runs_dict[name]['mean_response'], multiple_runs_dict[name]['mean_response']],
                y = [0, max(distribution_plot['data'][count].y)],
                mode = 'lines+markers',
                name = 'Mean ' + name.split('_')[-2] + '_' + name.split('_')[-1],
                line = dict(
                    color = COLOR_LIST_DISTRIBUTION_PLOTS[count]
                )
            ),
            row = 2,
            col = 1
        )

        # plot distribution std
        fig.add_trace(
            go.Scatter(
                x = [multiple_runs_dict[name]['mean_response'] - multiple_runs_dict[name]['std_response'],
                     multiple_runs_dict[name]['mean_response'] + multiple_runs_dict[name]['std_response']],
                y = [count * 0.5, count * 0.5],
                mode = 'lines+markers',
                name = 'Std ' + name.split('_')[-2] + '_' + name.split('_')[-1],
                line = dict(
                    color = COLOR_LIST_DISTRIBUTION_PLOTS[count]
                )
            ),
            row = 2,
            col = 1
        )
        count += 1

    # Add figure title
    fig.update_layout(
        title_text = "Performance distribution plots"
    )

    # save figure
    plotly.offline.plot(fig, filename = os.path.join(output_directory, 'Distribution_plots.html'), auto_open = False)

    # log distribution parameters
    for name in list(multiple_runs_dict.keys()):
        log(f'{name} :', log_file, widget)
        log(f'- Mean response: {multiple_runs_dict[name]["mean_response"]}', log_file, widget)
        log(f'- Std response: {multiple_runs_dict[name]["std_response"]}', log_file, widget)
        log(f'- Mean stimulus: {multiple_runs_dict[name]["mean_stimulus"]}', log_file, widget)
        log(f'- Std stimulus: {multiple_runs_dict[name]["std_stimulus"]}', log_file, widget)
