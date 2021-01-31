import os

import matplotlib.pyplot as plt
import numpy as np
import plotly
import plotly.graph_objects as go
from dtaidistance import dtw
from dtaidistance import dtw_visualisation as dtwvis

from util.constants import NUMBER_OF_TRIALS, TRIALS_FOR_STIMULUS
from util.util import log, exponential_moving_average


def dynamic_time_warping(metrics, output_path, stimulus_pairs, trial_dictionary):
    output_path = os.path.join(output_path, "DynamicTimeWarping")
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for metric in metrics:

        metric_dir = os.path.join(output_path, metric)
        if not os.path.exists(metric_dir):
            os.makedirs(metric_dir)

        for first_stimulus, second_stimulus in stimulus_pairs:
            first_values = []
            second_values = []

            first_text = []
            second_text = []

            for trial in TRIALS_FOR_STIMULUS[first_stimulus][1:]:
                for window in trial_dictionary[trial]:
                    first_values.append(float(trial_dictionary[trial][window][metric]))
                    first_text.append(f'{trial}_{window}')

            for trial in TRIALS_FOR_STIMULUS[second_stimulus][1:]:
                for window in trial_dictionary[trial]:
                    second_values.append(float(trial_dictionary[trial][window][metric]))
                    second_text.append(f'{trial}_{window}')

            if len(first_values) <= len(second_values):
                query = first_values
                template = second_values

                query_text = first_text
                template_text = second_text

                title = first_stimulus.split(' ')[0] + "+" + second_stimulus.split(' ')[0]
            else:
                query = second_values
                template = first_values

                query_text = second_text
                template_text = first_text

                title = second_stimulus.split(' ')[0] + "+" + first_stimulus.split(' ')[0]

            query = np.array(query)
            template = np.array(template)

            query_normalized = (query - query.min()) / (query.max() - query.min())
            template_normalized = (template - template.min()) / (
                    template.max() - template.min())

            _, paths = dtw.warping_paths(query_normalized, template_normalized, window = 10,
                                         psi = 0)
            best_path = dtw.best_path(paths)

            metric_file = os.path.join(metric_dir, f'{title}.txt')

            log(f'Similarity: {1 - paths[best_path[-1][0] + 1][best_path[-1][1] + 1] / len(best_path)}',
                file = metric_file)

            for pair in best_path:
                log(f'\tPair: {pair}. Match: {query_text[pair[0]]} {template_text[pair[1]]}', file = metric_file)

            fig, axes = dtwvis.plot_warpingpaths(query, template, paths, best_path)
            axes[0].texts[0].set_visible(False)
            axes[0].text(0, 0,
                         "Similarity = {:.4f}".format(
                             1 - paths[best_path[-1][0] + 1][best_path[-1][1] + 1] / len(best_path)))

            plt.savefig(os.path.join(metric_dir, f'{title}.png'))
            plt.close()


def trend_analysis(output_path, metrics, trial_dictionary, period = 80):
    output_path = os.path.join(output_path, 'TrendAnalysis')

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for metric in metrics:

        metric_values = []
        for trial in range(31, NUMBER_OF_TRIALS):
            for window in trial_dictionary[trial]:
                metric_values.append(trial_dictionary[trial][window][metric])

        figure = go.Figure()
        values = []

        for trial in range(31, NUMBER_OF_TRIALS):

            val = []
            for window in trial_dictionary[trial]:
                val.append(trial_dictionary[trial][window][metric])
            values.append(sum(val) / len(val))

        values = exponential_moving_average(values, period = period)

        figure.add_trace(
            go.Scatter(
                x = [i for i in range(len(values))],
                y = values,
                name = metric + '_mean',
                mode = 'lines'
            )
        )

        figure.update_layout(

            xaxis_title = 'trial',
            yaxis_title = metric,
            showlegend = True)

        figure.update_layout(barmode = 'overlay')

        plotly.offline.plot(figure, filename = os.path.join(output_path, f'{metric}.html'), auto_open = False)
