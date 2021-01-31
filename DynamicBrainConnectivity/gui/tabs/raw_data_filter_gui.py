import threading
import tkinter
from tkinter import filedialog

from gui.logging_text_handler import TextHandler
from gui.popup import Popup
from gui.validator import check_if_path_valid, check_if_int
from rawDataFilter.raw_data_filter import raw_data_filter


class RawDataFilterGui:

    def __init__(self, tab_raw_data_filter, top_module):

        # set tab
        self.tab_raw_data_filter = tab_raw_data_filter

        # set parent
        self.top_module = top_module

        self.tab_raw_data_filter.bind("<Visibility>", self.on_visibility)

        # set up tab
        self.raw_data_filter_init()

    def raw_data_filter_init(self):
        # FILTERED DATA
        folder_path_data = tkinter.StringVar()

        label_data = tkinter.Label(master = self.tab_raw_data_filter)
        label_data.place(x = 20, y = 20)
        label_data.config(text = "EEG Dots folder")

        textfield_data = tkinter.Entry(master = self.tab_raw_data_filter, width = 100, state = 'disabled')
        textfield_data.place(x = 140, y = 20)

        browse_button_data = tkinter.Button(master = self.tab_raw_data_filter, text = "Browse",
                                            command = lambda: self.folder_browse_button(folder_path_data,
                                                                                        textfield_data))
        browse_button_data.place(x = 750, y = 15)
        browse_button_data.config(width = 10, height = 1)

        # OUTPUT PATH
        folder_path_output = tkinter.StringVar()

        label_output = tkinter.Label(master = self.tab_raw_data_filter, text = "Output path")
        label_output.place(x = 20, y = 80)
        label_output.config(text = "Output path")

        textfield_output = tkinter.Entry(master = self.tab_raw_data_filter, width = 100, state = 'disabled')
        textfield_output.place(x = 140, y = 80)

        browse_button_output = tkinter.Button(master = self.tab_raw_data_filter, text = "Browse",
                                              command = lambda: self.folder_browse_button(folder_path_output,
                                                                                          textfield_output))
        browse_button_output.place(x = 750, y = 75)
        browse_button_output.config(width = 10, height = 1)

        # DEGREE OF PARALLELISM
        label_degree_of_parallelism = tkinter.Label(master = self.tab_raw_data_filter, text = "Degree of parallelism")
        label_degree_of_parallelism.place(x = 20, y = 140)
        label_degree_of_parallelism.config(text = "Degree of parallelism")

        textfield_degree_of_parallelism = tkinter.Entry(master = self.tab_raw_data_filter, width = 100)
        textfield_degree_of_parallelism.place(x = 140, y = 140)

        # TRIAL FILTER LENGTH
        label_trial_filter_length = tkinter.Label(master = self.tab_raw_data_filter, text = "Trial filter length")
        label_trial_filter_length.place(x = 20, y = 200)
        label_trial_filter_length.config(text = "Trial filter length")

        textfield_trial_filter_length = tkinter.Entry(master = self.tab_raw_data_filter, width = 100)
        textfield_trial_filter_length.place(x = 140, y = 200)
        textfield_trial_filter_length.insert(0, '627')

        # LOGGING AREA
        text_area = tkinter.Text(self.tab_raw_data_filter, height = 10, width = 100)
        text_area.place(x = 20, y = 260)
        widget = TextHandler(text_area)

        # START BUTTON
        start_button = tkinter.Button(master = self.tab_raw_data_filter, text = "Start",
                                      command = lambda: self.start_raw_data_filter(folder_path_data.get(),
                                                                                   folder_path_output.get(),
                                                                                   textfield_degree_of_parallelism.get(),
                                                                                   textfield_trial_filter_length.get(),
                                                                                   widget))
        start_button.place(x = 375, y = 450)
        start_button.config(width = 10, height = 1)

    def start_raw_data_filter(self, data_path, output_path, degree_of_parallelism, trial_filter_length,
                              widget):

        if not check_if_path_valid(data_path):
            widget.emit("Dots path is not valid!")
            Popup("Error", 400, 200).show("Dots path is not valid!")
            return
        if not check_if_path_valid(output_path):
            widget.emit("Output path is not valid!")
            Popup("Error", 400, 200).show("Output path is not valid!")
            return

        if not check_if_int(degree_of_parallelism):
            widget.emit("Degree of parallelism must be an integer!")
            Popup("Error", 400, 200).show("Degree of parallelism must be an integer!")
            return

        if not check_if_int(trial_filter_length):
            widget.emit("Trial filter length must be an integer!")
            Popup("Error", 400, 200).show("Trial filter length must be an integer!")
            return

        # start raw data filter
        thread = threading.Thread(target = raw_data_filter, args = (
            data_path,
            output_path,
            int(degree_of_parallelism),
            int(trial_filter_length),
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
        self.top_module.geometry(f'{850}x{550}')
