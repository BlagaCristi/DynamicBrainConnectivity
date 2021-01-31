import threading
import tkinter
from tkinter import filedialog

from gui.logging_text_handler import TextHandler
from gui.popup import Popup
from gui.validator import check_if_path_valid, check_if_T_F, check_if_int
from classificationStatistics.model_classification_statistics import model_classification_statistics


class ModelClassificationStatisticsGui:

    def __init__(self, tab_model_classification_statistics, top_module):

        # set tab
        self.tab_model_classification_statistics = tab_model_classification_statistics

        # set parent
        self.top_module = top_module

        self.tab_model_classification_statistics.bind("<Visibility>", self.on_visibility)

        # set up tab
        self.model_classification_statistics_init()

    def model_classification_statistics_init(self):
        # MODEL PATH
        folder_path_model = tkinter.StringVar()

        label_model = tkinter.Label(master=self.tab_model_classification_statistics, text="Model (with loaders)")
        label_model.place(x=20, y=20)
        label_model.config(text="Model (with loaders)")

        textfield_model = tkinter.Entry(master=self.tab_model_classification_statistics, width=100, state='disabled')
        textfield_model.place(x=140, y=20)

        browse_button_model = tkinter.Button(master=self.tab_model_classification_statistics, text="Browse",
                                             command=lambda: self.folder_browse_button(folder_path_model,
                                                                                       textfield_model))
        browse_button_model.place(x=750, y=15)
        browse_button_model.config(width=10, height=1)

        # TRIAL LENGTHS PATH
        folder_path_trial_lengths = tkinter.StringVar()

        label_trial_lengths = tkinter.Label(master=self.tab_model_classification_statistics, text="Trial lengths folder")
        label_trial_lengths.place(x=20, y=80)
        label_trial_lengths.config(text="Trial lengths file")

        textfield_trial_lengths = tkinter.Entry(master=self.tab_model_classification_statistics, width=100,
                                                state='disabled')
        textfield_trial_lengths.place(x=140, y=80)

        browse_button_model = tkinter.Button(master=self.tab_model_classification_statistics, text="Browse",
                                             command=lambda: self.folder_browse_button(folder_path_trial_lengths,
                                                                                       textfield_trial_lengths))
        browse_button_model.place(x=750, y=75)
        browse_button_model.config(width=10, height=1)

        # OUTPUT PATH
        folder_path_output = tkinter.StringVar()

        label_output = tkinter.Label(master=self.tab_model_classification_statistics, text="Output path")
        label_output.place(x=20, y=140)
        label_output.config(text="Output path")

        textfield_output = tkinter.Entry(master=self.tab_model_classification_statistics, width=100, state='disabled')
        textfield_output.place(x=140, y=140)

        browse_button_output = tkinter.Button(master=self.tab_model_classification_statistics, text="Browse",
                                              command=lambda: self.folder_browse_button(folder_path_output,
                                                                                        textfield_output))
        browse_button_output.place(x=750, y=135)
        browse_button_output.config(width=10, height=1)

        # MEDIAN VALUE
        label_median_value = tkinter.Label(master=self.tab_model_classification_statistics, text="Median value")
        label_median_value.place(x=20, y=200)
        label_median_value.config(text="Median value")

        textfield_median_value = tkinter.Entry(master=self.tab_model_classification_statistics, width=100)
        textfield_median_value.place(x=140, y=200)

        # GENERATE FROM TRAIN
        label_generate_from_train = tkinter.Label(master=self.tab_model_classification_statistics,
                                                  text="Generate from train T/F")
        label_generate_from_train.place(x=20, y=260)
        label_generate_from_train.config(text="Generate from train T/F")

        textfield_generate_from_train = tkinter.Entry(master=self.tab_model_classification_statistics, width=100)
        textfield_generate_from_train.place(x=220, y=260)

        # PLOT GENERATE FROM CROSS
        label_generate_from_cross = tkinter.Label(master=self.tab_model_classification_statistics,
                                                  text="Generate from cross T/F")
        label_generate_from_cross.place(x=20, y=320)
        label_generate_from_cross.config(text="Generate from cross T/F")

        textfield_generate_from_cross = tkinter.Entry(master=self.tab_model_classification_statistics, width=100)
        textfield_generate_from_cross.place(x=220, y=320)

        # GENERATE FROM TEST
        label_generate_from_test = tkinter.Label(master=self.tab_model_classification_statistics,
                                                 text="Generate from test T/F")
        label_generate_from_test.place(x=20, y=380)
        label_generate_from_test.config(text="Generate from test T/F")

        textfield_generate_from_test = tkinter.Entry(master=self.tab_model_classification_statistics, width=100)
        textfield_generate_from_test.place(x=220, y=380)

        # LOGGING AREA

        text_area = tkinter.Text(self.tab_model_classification_statistics, height=10, width=100)
        text_area.place(x=20, y=420)
        widget = TextHandler(text_area)

        # START BUTTON
        start_button = tkinter.Button(master=self.tab_model_classification_statistics, text="Start",
                                      command=lambda: self.start_model_classification_statistics(
                                          folder_path_model.get(),
                                          folder_path_trial_lengths.get(),
                                          folder_path_output.get(),
                                          textfield_median_value.get(),
                                          textfield_generate_from_train.get(),
                                          textfield_generate_from_cross.get(),
                                          textfield_generate_from_test.get(),
                                          widget))
        start_button.place(x=375, y=605)
        start_button.config(width=10, height=1)

    def start_model_classification_statistics(self, model_path, trial_lengths, output_path,
                                              median_value,
                                              generate_from_train, generate_from_cross,
                                              generate_from_test, widget):
        if not check_if_path_valid(model_path):
            widget.emit("Model path is not valid!")
            Popup("Error", 400, 200).show("Model path is not valid!")
            return

        if not check_if_path_valid(output_path):
            widget.emit("Output path is not valid!")
            Popup("Error", 400, 200).show("Output path is not valid!")
            return

        if not check_if_path_valid(trial_lengths):
            widget.emit("Trial lengths path is not valid!")
            Popup("Error", 400, 200).show("Trial lengths path is not valid!")
            return

        if not check_if_int(median_value):
            widget.emit("Median value must be an integer!")
            Popup("Error", 400, 200).show("Median value must be an integer!")
            return

        if not check_if_T_F(generate_from_train):
            widget.emit("Generate from train must be either 'T'(true) of 'F'(false)!")
            Popup("Error", 400, 200).show("Generate from train must be either 'T'(true) of 'F'(false)!")
            return

        if not check_if_T_F(generate_from_cross):
            widget.emit("Generate from cross must be either 'T'(true) of 'F'(false)!")
            Popup("Error", 400, 200).show("Generate from cross must be either 'T'(true) of 'F'(false)!")
            return

        if not check_if_T_F(generate_from_test):
            widget.emit("Generate from test must be either 'T'(true) of 'F'(false)!")
            Popup("Error", 400, 200).show(
                "Generate from test must be either 'T'(true) of 'F'(false)!")
            return

        # start model classification statistics
        thread = threading.Thread(target=model_classification_statistics, args=(
            model_path,
            trial_lengths,
            output_path,
            median_value,
            generate_from_train == 'T',
            generate_from_cross == 'T',
            generate_from_test == 'T',
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
        self.top_module.geometry(f'{850}x{695}')
