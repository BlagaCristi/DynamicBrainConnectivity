import threading
import tkinter
from tkinter import filedialog

from gui.logging_text_handler import TextHandler
from gui.popup import Popup
from gui.validator import check_if_path_valid, check_if_T_F
from dualEnsembleClassifierPerformanceStatistics.dual_ensemble_classifier_performance_statistics import \
    dual_ensemble_classifier_performance_statistics


class DECPerformanceStatisticsGui:

    def __init__(self, tab_dec_performance_statistics, top_module):

        # set tab
        self.tab_dec_performance_statistics = tab_dec_performance_statistics

        # set parent
        self.top_module = top_module

        self.tab_dec_performance_statistics.bind("<Visibility>", self.on_visibility)

        # set up tab
        self.dec_performance_statistics_init()

    def dec_performance_statistics_init(self):
        # FILTERED DATA
        folder_path_data = tkinter.StringVar()

        label_data = tkinter.Label(master=self.tab_dec_performance_statistics)
        label_data.place(x=20, y=20)
        label_data.config(text="Initial Runs Folder")

        textfield_data = tkinter.Entry(master=self.tab_dec_performance_statistics, width=100,
                                       state='disabled')
        textfield_data.place(x=140, y=20)

        browse_button_data = tkinter.Button(master=self.tab_dec_performance_statistics, text="Browse",
                                            command=lambda: self.folder_browse_button(folder_path_data,
                                                                                      textfield_data))
        browse_button_data.place(x=750, y=15)
        browse_button_data.config(width=10, height=1)

        # MULTIPLE RUNS PATH
        folder_path_multiple_runs = tkinter.StringVar()

        label_multiple_runs = tkinter.Label(master=self.tab_dec_performance_statistics,
                                            text="Multiple Runs Folder")
        label_multiple_runs.place(x=20, y=80)
        label_multiple_runs.config(text="Multiple Runs Folder")

        textfield_multiple_runs = tkinter.Entry(master=self.tab_dec_performance_statistics, width=100,
                                                state='disabled')
        textfield_multiple_runs.place(x=140, y=80)

        browse_button_csv = tkinter.Button(master=self.tab_dec_performance_statistics, text="Browse",
                                           command=lambda: self.folder_browse_button(folder_path_multiple_runs,
                                                                                     textfield_multiple_runs))
        browse_button_csv.place(x=750, y=75)
        browse_button_csv.config(width=10, height=1)

        # OUTPUT PATH
        folder_path_output = tkinter.StringVar()

        label_output = tkinter.Label(master=self.tab_dec_performance_statistics, text="Output path")
        label_output.place(x=20, y=140)
        label_output.config(text="Output path")

        textfield_output = tkinter.Entry(master=self.tab_dec_performance_statistics, width=100,
                                         state='disabled')
        textfield_output.place(x=140, y=140)

        browse_button_output = tkinter.Button(master=self.tab_dec_performance_statistics, text="Browse",
                                              command=lambda: self.folder_browse_button(folder_path_output,
                                                                                        textfield_output))
        browse_button_output.place(x=750, y=135)
        browse_button_output.config(width=10, height=1)

        # GENERATE SIMPLE PLOTS
        label_simple_plots = tkinter.Label(master=self.tab_dec_performance_statistics,
                                           text="Simple plots T/F")
        label_simple_plots.place(x=20, y=200)
        label_simple_plots.config(text="Simple plots T/F")

        textfield_simple_plots = tkinter.Entry(master=self.tab_dec_performance_statistics, width=100)
        textfield_simple_plots.place(x=140, y=200)

        # GENERATE DISTRIBUTION PLOTS
        label_distribution_plots = tkinter.Label(master=self.tab_dec_performance_statistics,
                                                 text="Distribution plots T/F")
        label_distribution_plots.place(x=20, y=260)
        label_distribution_plots.config(text="Distribution plots T/F")

        textfield_distribution_plots = tkinter.Entry(master=self.tab_dec_performance_statistics, width=100)
        textfield_distribution_plots.place(x=140, y=260)

        # LOGGING AREA

        text_area = tkinter.Text(self.tab_dec_performance_statistics, height=10, width=100)
        text_area.place(x=20, y=320)
        widget = TextHandler(text_area)

        # START BUTTON
        start_button = tkinter.Button(master=self.tab_dec_performance_statistics, text="Start",
                                      command=lambda: self.start_dec_performance_statistics(
                                          folder_path_data.get(),
                                          folder_path_multiple_runs.get(),
                                          folder_path_output.get(),
                                          textfield_simple_plots.get(),
                                          textfield_distribution_plots.get(),
                                          widget))
        start_button.place(x=375, y=510)
        start_button.config(width=10, height=1)

    def start_dec_performance_statistics(self, data_path, multiple_runs_path, output_path, simple_plots,
                                         distribution_plots, widget):
        if not check_if_path_valid(data_path):
            widget.emit("Data path is not valid!")
            Popup("Error", 400, 200).show("Data path is not valid!")
            return

        if not check_if_path_valid(multiple_runs_path):
            widget.emit("Multiple runs path is not valid!")
            Popup("Error", 400, 200).show("Multiple runs path is not valid!")
            return

        if not check_if_path_valid(output_path):
            widget.emit("Output path is not valid!")
            Popup("Error", 400, 200).show("Output path is not valid!")
            return

        if not check_if_T_F(simple_plots):
            widget.emit("Simple plots must be either 'T'(true) of 'F'(false)!")
            Popup("Error", 400, 200).show("Simple plots must be either 'T'(true) of 'F'(false)!")
            return

        if not check_if_T_F(distribution_plots):
            widget.emit("Distribution plots must be either 'T'(true) of 'F'(false)!")
            Popup("Error", 400, 200).show("Distribution plots visdom must be either 'T'(true) of 'F'(false)!")
            return

        # start trial_division_performance_statistics
        thread = threading.Thread(target=dual_ensemble_classifier_performance_statistics, args=(
            data_path,
            multiple_runs_path,
            output_path,
            simple_plots == 'T',
            distribution_plots == 'T',
            widget
        ))

        thread.start()

    def folder_browse_button(self, folder_path, textfield):
        filename = filedialog.askdirectory()
        folder_path.set(filename)
        textfield.configure(state='normal')
        textfield.delete(0, "end")
        textfield.insert(0, filename)
        textfield.configure(state='disabled')

    def on_visibility(self, event):

        # resize window
        self.top_module.geometry(f'{850}x{605}')
