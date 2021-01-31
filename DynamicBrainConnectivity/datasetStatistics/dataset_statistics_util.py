import os

import numpy as np
import plotly
import plotly.graph_objects as go


# plot the histogram and save it locally under a user-specified name
def show_histogram_dataset_statistics(values, title, output_folder, bin_size = 500, median_value = None):
    fig = go.Figure()
    max_val = max(values)
    fig.add_trace(go.Histogram(x = np.array(values), name = title, xbins = {
        'start': 0,
        'end': max_val,
        'size': bin_size
    }))

    if median_value is not None:
        fig.add_trace(
            go.Scatter(
                x = [median_value, median_value],
                y = [0, 50],
                mode = 'lines+markers',
                name = 'Median threshold'
            )
        )

    # Overlay both histograms
    fig.update_layout(barmode = 'overlay',
                      xaxis_title = "Trial length",
                      yaxis_title = "Frequency",
                      showlegend = False)

    # Reduce opacity to see both histograms
    fig.update_traces(opacity = 0.75)
    file_name = os.path.join(output_folder, title + ".html")
    plotly.offline.plot(fig, filename = file_name, auto_open = False)
