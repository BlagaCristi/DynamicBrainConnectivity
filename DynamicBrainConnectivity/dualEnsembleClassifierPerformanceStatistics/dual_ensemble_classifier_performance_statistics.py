import datetime
import os

from dualEnsembleClassifierPerformanceStatistics.distribution_plots import plot_distribution_for_multiple_runs
from dualEnsembleClassifierPerformanceStatistics.initial_performance import plot_initial_performance
from util.util import log


def dual_ensemble_classifier_performance_statistics(data_directory, multiple_runs_directory, output_directory,
                                                    create_simple_plots, create_distribution_plots, widget):
    log("Started Dual Ensemble Classifier Performance Statistics: " + str(datetime.datetime.now()), None, widget)

    # create output directory
    output_directory = os.path.join(output_directory, 'DualEnsembleClassifierPerformanceStatistics')
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # create simple plots
    if create_simple_plots:
        plot_initial_performance(data_directory, output_directory, widget)

    # create distribution plots
    if create_distribution_plots:
        plot_distribution_for_multiple_runs(multiple_runs_directory, output_directory, widget)

    log("Finished Dual Ensemble Classifier Performance Statistics: " + str(datetime.datetime.now()), None, widget)
