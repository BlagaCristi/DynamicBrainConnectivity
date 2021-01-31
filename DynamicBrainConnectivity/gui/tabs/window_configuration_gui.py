import threading
import tkinter
from tkinter import filedialog

from gui.logging_text_handler import TextHandler
from gui.popup import Popup
from gui.validator import check_if_path_valid, check_if_int
from trialWindowConfiguration.trial_window_configuration import trial_window_configuration


class WindowConfigurationGui:

    def __init__(self, tab_window_configuration, top_module):

        # set tab
        self.tab_window_configuration = tab_window_configuration

        # set parent
        self.top_module = top_module

        self.tab_window_configuration.bind("<Visibility>", self.on_visibility)

        # set up tab
        self.window_configuration_filter_init()

    def window_configuration_filter_init(self):
        # FILTERED DATA
        folder_path_data = tkinter.StringVar()

        label_data = tkinter.Label(master = self.tab_window_configuration)
        label_data.place(x = 20, y = 20)
        label_data.config(text = "EEG Dots folder")

        textfield_data = tkinter.Entry(master = self.tab_window_configuration, width = 100, state = 'disabled')
        textfield_data.place(x = 140, y = 20)

        browse_button_data = tkinter.Button(master = self.tab_window_configuration, text = "Browse",
                                            command = lambda: self.folder_browse_button(folder_path_data,
                                                                                        textfield_data))
        browse_button_data.place(x = 750, y = 15)
        browse_button_data.config(width = 10, height = 1)

        # OUTPUT PATH
        folder_path_output = tkinter.StringVar()

        label_output = tkinter.Label(master = self.tab_window_configuration, text = "Output path")
        label_output.place(x = 20, y = 80)
        label_output.config(text = "Output path")

        textfield_output = tkinter.Entry(master = self.tab_window_configuration, width = 100, state = 'disabled')
        textfield_output.place(x = 140, y = 80)

        browse_button_output = tkinter.Button(master = self.tab_window_configuration, text = "Browse",
                                              command = lambda: self.folder_browse_button(folder_path_output,
                                                                                          textfield_output))
        browse_button_output.place(x = 750, y = 75)
        browse_button_output.config(width = 10, height = 1)

        # WINDOW SIZE
        label_window_size = tkinter.Label(master = self.tab_window_configuration, text = "Window size")
        label_window_size.place(x = 20, y = 140)
        label_window_size.config(text = "Window size")

        textfield_window_size = tkinter.Entry(master = self.tab_window_configuration, width = 100)
        textfield_window_size.place(x = 140, y = 140)

        # WINDOW OFFSET
        label_window_offset = tkinter.Label(master = self.tab_window_configuration, text = "Window offset")
        label_window_offset.place(x = 20, y = 200)
        label_window_offset.config(text = "Window offset")

        textfield_window_offset = tkinter.Entry(master = self.tab_window_configuration, width = 100)
        textfield_window_offset.place(x = 140, y = 200)

        # THRESHOLD VALUE
        label_threshold_value = tkinter.Label(master = self.tab_window_configuration, text = "Threshold value")
        label_threshold_value.place(x = 20, y = 260)
        label_threshold_value.config(text = "Threshold value")

        textfield_threshold_value = tkinter.Entry(master = self.tab_window_configuration, width = 100)
        textfield_threshold_value.place(x = 140, y = 260)

        # LOGGING AREA
        text_area = tkinter.Text(self.tab_window_configuration, height = 10, width = 100)
        text_area.place(x = 20, y = 320)
        widget = TextHandler(text_area)

        # START BUTTON
        start_button = tkinter.Button(master = self.tab_window_configuration, text = "Start",
                                      command = lambda: self.start_window_configuration(folder_path_data.get(),
                                                                                        folder_path_output.get(),
                                                                                        textfield_window_size.get(),
                                                                                        textfield_window_offset.get(),
                                                                                        textfield_threshold_value.get(),
                                                                                        widget))
        start_button.place(x = 375, y = 510)
        start_button.config(width = 10, height = 1)

    def start_window_configuration(self, data_path, output_path, window_size, window_offset, threshold_value,
                                   widget):

        if not check_if_path_valid(data_path):
            widget.emit("Dots path is not valid!")
            Popup("Error", 400, 200).show("Dots path is not valid!")
            return
        if not check_if_path_valid(output_path):
            widget.emit("Output path is not valid!")
            Popup("Error", 400, 200).show("Output path is not valid!")
            return

        if not check_if_int(window_size):
            widget.emit("Window size must be an integer!")
            Popup("Error", 400, 200).show("Window size must be an integer!")
            return

        if not check_if_int(window_offset):
            widget.emit("Window offset must be an integer!")
            Popup("Error", 400, 200).show("Window offset must be an integer!")
            return

        if not check_if_int(threshold_value):
            widget.emit("Threshold must be an integer!")
            Popup("Error", 400, 200).show("Threshold must be an integer!")
            return

        # start trial window configuration
        thread = threading.Thread(target = trial_window_configuration, args = (
            data_path,
            output_path,
            int(window_size),
            int(window_offset),
            int(threshold_value),
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
