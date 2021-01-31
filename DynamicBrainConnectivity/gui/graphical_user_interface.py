import tkinter
from tkinter.ttk import Notebook

from gui.tabs.dataset_statistics_gui import DatasetStatisticsGui
from gui.tabs.dec_gui import DECGui
from gui.tabs.dec_performance_statistics_gui import DECPerformanceStatisticsGui
from gui.tabs.graph_analysis_gui import GraphAnalysisGui
from gui.tabs.graph_difference_visualization_gui import GraphDifferenceVisualizationGui
from gui.tabs.graph_metrics_gui import GraphMetricsGui
from gui.tabs.graph_window_visualization_gui import GraphWindowVisualizationGui
from gui.tabs.model_analytics_gui import ModelAnalyticsGui
from gui.tabs.model_classification_statistics_gui import ModelClassificationStatisticsGui
from gui.tabs.raw_data_filter_gui import RawDataFilterGui
from gui.tabs.rgw_gui import RGWGui
from gui.tabs.window_configuration_gui import WindowConfigurationGui


class GUI:

    def __init__(self, name, width, height):
        # define main window
        self.top_module = tkinter.Tk(className = name)

        # set window size
        self.top_module.geometry(f'{width}x{height}')

        # don't allow resize
        self.top_module.resizable(False, False)

        # inset a notebook form where we can have tabs
        self.tab_parent = Notebook(self.top_module)

        # insert tab for DEC
        self.tab_dec = tkinter.Frame(self.tab_parent)
        self.tab_parent.add(self.tab_dec, text = "   DEC\n"
                                                 "Training\n")

        # insert tab for DatasetStatistics
        self.tab_dataset_statistics = tkinter.Frame(self.tab_parent)
        self.tab_parent.add(self.tab_dataset_statistics, text = " Dataset\n"
                                                                "Statistics\n")

        # insert tab for RawDataFilter
        self.tab_raw_data_filter = tkinter.Frame(self.tab_parent)
        self.tab_parent.add(self.tab_raw_data_filter, text = " Raw\n"
                                                             "Data\n"
                                                             "Filter")

        # insert tab for DECPerformanceStatistics
        self.tab_dec_performance_statistics = tkinter.Frame(self.tab_parent)
        self.tab_parent.add(self.tab_dec_performance_statistics, text = "       DEC\n"
                                                                        "Performance\n"
                                                                        "   Statistics")

        # insert tab for ModelAnalytics
        self.tab_model_analytics = tkinter.Frame(self.tab_parent)
        self.tab_parent.add(self.tab_model_analytics, text = "  Model\n"
                                                             "Analytics\n")

        # insert tab for ModelClassificationStatistics
        self.tab_model_classification_statistics = tkinter.Frame(self.tab_parent)
        self.tab_parent.add(self.tab_model_classification_statistics, text = "    Model\n"
                                                                             "Classification\n"
                                                                             "  Statistics")

        self.tab_window_configuration = tkinter.Frame(self.tab_parent)
        self.tab_parent.add(self.tab_window_configuration, text = "   Window\n"
                                                                  "Configuration\n")

        self.tab_rgw_gui = tkinter.Frame(self.tab_parent)
        self.tab_parent.add(self.tab_rgw_gui, text = "\n RGW \n")

        self.tab_graph_window_visualization = tkinter.Frame(self.tab_parent)
        self.tab_parent.add(self.tab_graph_window_visualization, text = "    Graph\n"
                                                                        "   Window\n"
                                                                        "Visualization")

        self.tab_graph_difference_visualization = tkinter.Frame(self.tab_parent)
        self.tab_parent.add(self.tab_graph_difference_visualization, text = "    Graph\n"
                                                                            " Difference\n"
                                                                            "Visualization")

        self.tab_graph_metrics = tkinter.Frame(self.tab_parent)
        self.tab_parent.add(self.tab_graph_metrics, text = " Graph\n"
                                                           "Metrics\n")

        self.tab_graph_analysis = tkinter.Frame(self.tab_parent)
        self.tab_parent.add(self.tab_graph_analysis, text = " Graph\n"
                                                            "Analysis\n")

        # make tabs visible
        self.tab_parent.pack(expand = 1, fill = 'both')

        # init trial division classifier tab
        DECGui(self.tab_dec, self.top_module)

        # init dataset statistics tab
        DatasetStatisticsGui(self.tab_dataset_statistics, self.top_module)

        # init raw data filter tab
        RawDataFilterGui(self.tab_raw_data_filter, self.top_module)

        # initial trial division performance statistics tab
        DECPerformanceStatisticsGui(self.tab_dec_performance_statistics, self.top_module)

        # init model analytics tab
        ModelAnalyticsGui(self.tab_model_analytics, self.top_module)

        # init model classification statistics tab
        ModelClassificationStatisticsGui(self.tab_model_classification_statistics, self.top_module)

        # init window configuration tab
        WindowConfigurationGui(self.tab_window_configuration, self.top_module)

        # init rgw tab
        RGWGui(self.tab_rgw_gui, self.top_module)

        # init graph window visualization tab
        GraphWindowVisualizationGui(self.tab_graph_window_visualization, self.top_module)

        # init graph difference visualization tab
        GraphDifferenceVisualizationGui(self.tab_graph_difference_visualization, self.top_module)

        # init graph metrics tab
        GraphMetricsGui(self.tab_graph_metrics, self.top_module)

        # init graph analysis tab
        GraphAnalysisGui(self.tab_graph_analysis, self.top_module)

    def display(self):
        self.top_module.mainloop()
