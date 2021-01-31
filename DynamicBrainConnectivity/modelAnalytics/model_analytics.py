from modelAnalytics.model_analytics_util import plot_heatmaps
from util.util import log


def model_analytics(multiple_runs_path, output_path, plot_weight_heatmaps, plot_collapsed_weight_heatmaps,
                    plot_collapsed_weight_heatmaps_aligned, widget):
    log("Started plotting weight heatmaps", None, widget)

    plot_heatmaps(multiple_runs_path, plot_weight_heatmaps, plot_collapsed_weight_heatmaps,
                  plot_collapsed_weight_heatmaps_aligned, output_path, widget)

    log("Finished plotting weight heatmaps", None, widget)
