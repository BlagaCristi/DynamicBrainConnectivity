import threading
import tkinter
from tkinter import filedialog

from gui.logging_text_handler import TextHandler
from gui.popup import Popup
from gui.validator import check_if_path_valid, check_if_T_F
from modelAnalytics.model_analytics import model_analytics


class ModelAnalyticsGui:

    def __init__(self, tab_model_analytics, top_module):

        # set tab
        self.tab_model_analytics = tab_model_analytics

        # set parent
        self.top_module = top_module

        self.tab_model_analytics.bind("<Visibility>", self.on_visibility)

        # set up tab
        self.model_analytics_init()

    def model_analytics_init(self):
        # MULTIPLE RUNS PATH
        folder_path_multiple_runs = tkinter.StringVar()

        label_multiple_runs = tkinter.Label(master = self.tab_model_analytics,
                                            text = "Multiple Runs Folder")
        label_multiple_runs.place(x = 20, y = 20)
        label_multiple_runs.config(text = "Multiple Runs Folder")

        textfield_multiple_runs = tkinter.Entry(master = self.tab_model_analytics, width = 100,
                                                state = 'disabled')
        textfield_multiple_runs.place(x = 140, y = 20)

        browse_button_csv = tkinter.Button(master = self.tab_model_analytics, text = "Browse",
                                           command = lambda: self.folder_browse_button(folder_path_multiple_runs,
                                                                                       textfield_multiple_runs))
        browse_button_csv.place(x = 750, y = 15)
        browse_button_csv.config(width = 10, height = 1)

        # OUTPUT PATH
        folder_path_output = tkinter.StringVar()

        label_output = tkinter.Label(master = self.tab_model_analytics, text = "Output path")
        label_output.place(x = 20, y = 80)
        label_output.config(text = "Output path")

        textfield_output = tkinter.Entry(master = self.tab_model_analytics, width = 100, state = 'disabled')
        textfield_output.place(x = 140, y = 80)

        browse_button_output = tkinter.Button(master = self.tab_model_analytics, text = "Browse",
                                              command = lambda: self.folder_browse_button(folder_path_output,
                                                                                          textfield_output))
        browse_button_output.place(x = 750, y = 75)
        browse_button_output.config(width = 10, height = 1)

        # PLOT WEIGHT HEATMAPS
        label_weight_heatmaps = tkinter.Label(master = self.tab_model_analytics,
                                              text = "Weight heatmaps T/F")
        label_weight_heatmaps.place(x = 20, y = 140)
        label_weight_heatmaps.config(text = "Weight heatmaps T/F")

        textfield_weight_heatmaps = tkinter.Entry(master = self.tab_model_analytics, width = 100)
        textfield_weight_heatmaps.place(x = 220, y = 140)

        # PLOT COLLAPSED WEIGHT HEATMAPS
        label_collapsed_weight_heatmaps = tkinter.Label(master = self.tab_model_analytics,
                                                        text = "Collapsed heatmaps T/F")
        label_collapsed_weight_heatmaps.place(x = 20, y = 200)
        label_collapsed_weight_heatmaps.config(text = "Collapsed heatmaps T/F")

        textfield_collapsed_weight_heatmaps = tkinter.Entry(master = self.tab_model_analytics, width = 100)
        textfield_collapsed_weight_heatmaps.place(x = 220, y = 200)

        # PLOT COLLAPSED WEIGHT HEATMAPS ALIGNED
        label_collapsed_weight_heatmaps_aligned = tkinter.Label(master = self.tab_model_analytics,
                                                                text = "Collapsed heatmaps aligned T/F")
        label_collapsed_weight_heatmaps_aligned.place(x = 20, y = 260)
        label_collapsed_weight_heatmaps_aligned.config(text = "Collapsed heatmaps aligned T/F")

        textfield_collapsed_weight_heatmaps_aligned = tkinter.Entry(master = self.tab_model_analytics, width = 100)
        textfield_collapsed_weight_heatmaps_aligned.place(x = 220, y = 260)

        # LOGGING AREA

        text_area = tkinter.Text(self.tab_model_analytics, height = 10, width = 100)
        text_area.place(x = 20, y = 320)
        widget = TextHandler(text_area)

        # START BUTTON
        start_button = tkinter.Button(master = self.tab_model_analytics, text = "Start",
                                      command = lambda: self.start_model_analytics(
                                          folder_path_multiple_runs.get(),
                                          folder_path_output.get(),
                                          textfield_weight_heatmaps.get(),
                                          textfield_collapsed_weight_heatmaps.get(),
                                          textfield_collapsed_weight_heatmaps_aligned.get(),
                                          widget))
        start_button.place(x = 375, y = 510)
        start_button.config(width = 10, height = 1)

    def start_model_analytics(self, multiple_runs_path, output_path, plot_weight_heatmaps,
                              plot_collapsed_weight_heatmaps, plot_collapsed_weight_heatmaps_aligned, widget):
        if not check_if_path_valid(multiple_runs_path):
            widget.emit("Multiple runs path is not valid!")
            Popup("Error", 400, 200).show("Multiple runs path is not valid!")
            return

        if not check_if_path_valid(output_path):
            widget.emit("Output path is not valid!")
            Popup("Error", 400, 200).show("Output path is not valid!")
            return

        if not check_if_T_F(plot_weight_heatmaps):
            widget.emit("Plot weight heatmaps must be either 'T'(true) of 'F'(false)!")
            Popup("Error", 400, 200).show("Plot weight heatmaps must be either 'T'(true) of 'F'(false)!")
            return

        if not check_if_T_F(plot_collapsed_weight_heatmaps):
            widget.emit("Plot collapsed weight heatmaps must be either 'T'(true) of 'F'(false)!")
            Popup("Error", 400, 200).show("Plot collapsed weight heatmaps must be either 'T'(true) of 'F'(false)!")
            return

        if not check_if_T_F(plot_collapsed_weight_heatmaps_aligned):
            widget.emit("Plot collapsed weight heatmaps aligned must be either 'T'(true) of 'F'(false)!")
            Popup("Error", 400, 200).show(
                "Plot collapsed weight heatmaps aligned must be either 'T'(true) of 'F'(false)!")
            return

        # start model analytics
        thread = threading.Thread(target = model_analytics, args = (
            multiple_runs_path,
            output_path,
            plot_weight_heatmaps == 'T',
            plot_collapsed_weight_heatmaps == 'T',
            plot_collapsed_weight_heatmaps_aligned == 'T',
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
        self.top_module.geometry(f'{850}x{605}')
