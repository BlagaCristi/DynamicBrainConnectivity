import threading
import tkinter
from tkinter import filedialog

from datasetStatistics.dataset_statistics import dataset_statistics
from gui.logging_text_handler import TextHandler
from gui.popup import Popup
from gui.validator import check_if_path_valid


class DatasetStatisticsGui:

    def __init__(self, tab_dataset_statistics, top_module):

        # set tab
        self.tab_dataset_statistics = tab_dataset_statistics

        # set parent
        self.top_module = top_module

        self.tab_dataset_statistics.bind("<Visibility>", self.on_visibility)

        # set up tab
        self.dataset_statistics_init()

    def dataset_statistics_init(self):
        # FILTERED DATA
        folder_path_data = tkinter.StringVar()

        label_data = tkinter.Label(master = self.tab_dataset_statistics)
        label_data.place(x = 20, y = 20)
        label_data.config(text = "EEG Dots Folder")

        textfield_data = tkinter.Entry(master = self.tab_dataset_statistics, width = 100, state = 'disabled')
        textfield_data.place(x = 140, y = 20)

        browse_button_data = tkinter.Button(master = self.tab_dataset_statistics, text = "Browse",
                                            command = lambda: self.folder_browse_button(folder_path_data,
                                                                                        textfield_data))
        browse_button_data.place(x = 750, y = 15)
        browse_button_data.config(width = 10, height = 1)

        # OUTPUT PATH
        folder_path_output = tkinter.StringVar()

        label_output = tkinter.Label(master = self.tab_dataset_statistics, text = "Output path")
        label_output.place(x = 20, y = 80)
        label_output.config(text = "Output path")

        textfield_output = tkinter.Entry(master = self.tab_dataset_statistics, width = 100, state = 'disabled')
        textfield_output.place(x = 140, y = 80)

        browse_button_output = tkinter.Button(master = self.tab_dataset_statistics, text = "Browse",
                                              command = lambda: self.folder_browse_button(folder_path_output,
                                                                                          textfield_output))
        browse_button_output.place(x = 750, y = 75)
        browse_button_output.config(width = 10, height = 1)

        # LOGGING AREA
        text_area = tkinter.Text(self.tab_dataset_statistics, height = 10, width = 100)
        text_area.place(x = 20, y = 140)
        widget = TextHandler(text_area)

        # START BUTTON
        start_button = tkinter.Button(master = self.tab_dataset_statistics, text = "Start",
                                      command = lambda: self.start_dataset_statistics(folder_path_data.get(),
                                                                                             folder_path_output.get(),
                                                                                             widget))
        start_button.place(x = 375, y = 325)
        start_button.config(width = 10, height = 1)

    def start_dataset_statistics(self, data_path, output_path, widget):
        if not check_if_path_valid(data_path):
            widget.emit("Dots path is not valid!")
            Popup("Error", 400, 200).show("Dots path is not valid!")
            return

        if not check_if_path_valid(output_path):
            widget.emit("Output path is not valid!")
            Popup("Error", 400, 200).show("Output path is not valid!")
            return

        # start dataset statistics
        thread = threading.Thread(target = dataset_statistics, args = (
            data_path,
            output_path,
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
        self.top_module.geometry(f'{850}x{420}')
