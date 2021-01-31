import threading
import tkinter
from tkinter import filedialog

from graphAnalysis.graph_analysis import graph_analysis
from gui.logging_text_handler import TextHandler
from gui.popup import Popup
from gui.validator import check_if_path_valid, check_if_int
from util.constants import TRIALS_FOR_STIMULUS


class GraphAnalysisGui:

    def __init__(self, tab_graph_analysis, top_module):

        # set tab
        self.tab_graph_analysis = tab_graph_analysis

        # set parent
        self.top_module = top_module

        self.tab_graph_analysis.bind("<Visibility>", self.on_visibility)

        # set up tab
        self.graph_window_visualization_init()

    def graph_window_visualization_init(self):
        # FILTERED DATA
        folder_path_data = tkinter.StringVar()

        label_data = tkinter.Label(master = self.tab_graph_analysis)
        label_data.place(x = 20, y = 20)
        label_data.config(text = "Metric Results")

        textfield_data = tkinter.Entry(master = self.tab_graph_analysis, width = 100,
                                       state = 'disabled')
        textfield_data.place(x = 140, y = 20)

        browse_button_data = tkinter.Button(master = self.tab_graph_analysis, text = "Browse",
                                            command = lambda: self.folder_browse_button(folder_path_data,
                                                                                        textfield_data))
        browse_button_data.place(x = 750, y = 15)
        browse_button_data.config(width = 10, height = 1)

        # OUTPUT PATH
        folder_path_output = tkinter.StringVar()

        label_output = tkinter.Label(master = self.tab_graph_analysis, text = "Output path")
        label_output.place(x = 20, y = 80)
        label_output.config(text = "Output path")

        textfield_output = tkinter.Entry(master = self.tab_graph_analysis, width = 100,
                                         state = 'disabled')
        textfield_output.place(x = 140, y = 80)

        browse_button_output = tkinter.Button(master = self.tab_graph_analysis, text = "Browse",
                                              command = lambda: self.folder_browse_button(folder_path_output,
                                                                                          textfield_output))
        browse_button_output.place(x = 750, y = 75)
        browse_button_output.config(width = 10, height = 1)

        # TREND PERIOD
        label_trend_period = tkinter.Label(master = self.tab_graph_analysis, text = "Trend period")
        label_trend_period.place(x = 20, y = 140)
        label_trend_period.config(text = "Trend period")

        textfield_trend_period = tkinter.Entry(master = self.tab_graph_analysis, width = 100)
        textfield_trend_period.place(x = 140, y = 140)

        # LOGGING AREA
        text_area = tkinter.Text(self.tab_graph_analysis, height = 10, width = 100)
        text_area.place(x = 20, y = 220)
        widget = TextHandler(text_area)

        # START BUTTON
        start_button = tkinter.Button(master = self.tab_graph_analysis, text = "Start",
                                      command = lambda: self.start_graph_analysis(
                                          folder_path_data.get(),
                                          folder_path_output.get(),
                                          textfield_trend_period.get(),
                                          widget))
        start_button.place(x = 375, y = 410)
        start_button.config(width = 10, height = 1)

    def start_graph_analysis(self, data_path, output_path, trend_period, widget):

        if not check_if_path_valid(data_path):
            widget.emit("Metric results path is not valid!")
            Popup("Error", 400, 200).show("Metric results path is not valid!")
            return
        if not check_if_path_valid(output_path):
            widget.emit("Output path is not valid!")
            Popup("Error", 400, 200).show("Output path is not valid!")
            return

        if not check_if_int(trend_period):
            widget.emit("Trend period must be an integer!")
            Popup("Error", 400, 200).show("Trend period must be an integer!")
            return

        # start graph analysis
        thread = threading.Thread(target = graph_analysis, args = (
            data_path,
            output_path,
            list(TRIALS_FOR_STIMULUS.keys()),
            int(trend_period),
            widget
        ))

        thread.start()

    def folder_browse_button(self, folder_path, textfield):
        filename = filedialog.askdirectory()
        folder_path.set(filename)
        textfield.configure(state = 'normal')
        textfield.delete(0, "end")
        textfield.insert(0, filename)
        textfield.configure(state = 'disabled')

    def on_visibility(self, event):

        # resize window
        self.top_module.geometry(f'{850}x{505}')
