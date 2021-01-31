import threading
import tkinter
from tkinter import filedialog

from gui.logging_text_handler import TextHandler
from gui.popup import Popup
from gui.validator import check_if_path_valid, check_if_int, check_if_T_F
from dualEnsembleClassifier.dual_ensemble_classifier import dual_ensemble_classifier


class DECGui:

    def __init__(self, tab_dec, top_module):

        # set tab
        self.tab_dec = tab_dec

        # set parent
        self.top_module = top_module

        self.tab_dec.bind("<Visibility>", self.on_visibility)

        # set up tab
        self.dec_init()

    def dec_init(self):
        # FILTERED DATA
        folder_path_data = tkinter.StringVar()

        label_data = tkinter.Label(master = self.tab_dec)
        label_data.place(x = 20, y = 20)
        label_data.config(text = "EEG Raw Data Filtered")

        textfield_data = tkinter.Entry(master = self.tab_dec, width = 100, state ='disabled')
        textfield_data.place(x = 140, y = 20)

        browse_button_data = tkinter.Button(master = self.tab_dec, text ="Browse",
                                            command = lambda: self.folder_browse_button(folder_path_data,
                                                                                        textfield_data))
        browse_button_data.place(x = 750, y = 15)
        browse_button_data.config(width = 10, height = 1)

        # CSV PATH
        folder_path_csv = tkinter.StringVar()

        label_csv = tkinter.Label(master = self.tab_dec, text ="EEG Dots folder")
        label_csv.place(x = 20, y = 80)
        label_csv.config(text = "EEG Dots folder")

        textfield_csv = tkinter.Entry(master = self.tab_dec, width = 100, state ='disabled')
        textfield_csv.place(x = 140, y = 80)

        browse_button_csv = tkinter.Button(master = self.tab_dec, text ="Browse",
                                           command = lambda: self.folder_browse_button(folder_path_csv,
                                                                                       textfield_csv))
        browse_button_csv.place(x = 750, y = 75)
        browse_button_csv.config(width = 10, height = 1)

        # OUTPUT PATH
        folder_path_output = tkinter.StringVar()

        label_output = tkinter.Label(master = self.tab_dec, text ="Output path")
        label_output.place(x = 20, y = 140)
        label_output.config(text = "Output path")

        textfield_output = tkinter.Entry(master = self.tab_dec, width = 100, state ='disabled')
        textfield_output.place(x = 140, y = 140)

        browse_button_output = tkinter.Button(master = self.tab_dec, text ="Browse",
                                              command = lambda: self.folder_browse_button(folder_path_output,
                                                                                          textfield_output))
        browse_button_output.place(x = 750, y = 135)
        browse_button_output.config(width = 10, height = 1)

        # WINDOW SIZE
        label_window_size = tkinter.Label(master = self.tab_dec, text ="Window size")
        label_window_size.place(x = 20, y = 200)
        label_window_size.config(text = "Window size")

        textfield_window_size = tkinter.Entry(master = self.tab_dec, width = 100)
        textfield_window_size.place(x = 140, y = 200)

        # WINDOW OFFSET
        label_window_offset = tkinter.Label(master = self.tab_dec, text ="Window offset")
        label_window_offset.place(x = 20, y = 260)
        label_window_offset.config(text = "Window offset")

        textfield_window_offset = tkinter.Entry(master = self.tab_dec, width = 100)
        textfield_window_offset.place(x = 140, y = 260)

        # DIVISION SIZE
        label_division_size = tkinter.Label(master = self.tab_dec, text ="Division size")
        label_division_size.place(x = 20, y = 320)
        label_division_size.config(text = "Division size")

        textfield_division_size = tkinter.Entry(master = self.tab_dec, width = 100)
        textfield_division_size.place(x = 140, y = 320)

        # ONLY TWO SUBJECTS
        label_only_two = tkinter.Label(master = self.tab_dec,
                                       text = "Only two subjects T/F")
        label_only_two.place(x = 20, y = 380)
        label_only_two.config(text = "Only two subjects T/F")

        textfield_only_two = tkinter.Entry(master = self.tab_dec, width = 100)
        textfield_only_two.place(x = 140, y = 380)

        # WITH VISDOM
        label_with_visdom = tkinter.Label(master = self.tab_dec, text ="With visdom T/F")
        label_with_visdom.place(x = 20, y = 440)
        label_with_visdom.config(text = "With visdom T/F")

        textfield_with_visdom = tkinter.Entry(master = self.tab_dec, width = 100)
        textfield_with_visdom.place(x = 140, y = 440)

        # WITH VISDOM
        label_save_loaders = tkinter.Label(master = self.tab_dec,
                                           text = "Save loaders T/F")
        label_save_loaders.place(x = 20, y = 500)
        label_save_loaders.config(text = "Save loaders T/F")

        textfield_save_loaders = tkinter.Entry(master = self.tab_dec, width = 100)
        textfield_save_loaders.place(x = 140, y = 500)

        # LOGGING AREA

        text_area = tkinter.Text(self.tab_dec, height = 10, width = 100)
        text_area.place(x = 20, y = 540)
        widget = TextHandler(text_area)

        # START BUTTON
        start_button = tkinter.Button(master = self.tab_dec, text ="Start",
                                      command = lambda: self.start_dec(folder_path_data.get(),
                                                                       folder_path_csv.get(),
                                                                       folder_path_output.get(),
                                                                       textfield_window_size.get(),
                                                                       textfield_window_offset.get(),
                                                                       textfield_division_size.get(),
                                                                       textfield_only_two.get(),
                                                                       textfield_with_visdom.get(),
                                                                       textfield_save_loaders.get(),
                                                                       widget))
        start_button.place(x = 375, y = 725)
        start_button.config(width = 10, height = 1)

    def start_dec(self, data_path, csv_path, output_path, window_size, window_offset,
                  division_size, only_two_subjects, with_visdom, save_loader, widget):
        if not check_if_path_valid(data_path):
            widget.emit("Data path is not valid!")
            Popup("Error", 400, 200).show("Data path is not valid!")
            return

        if not check_if_path_valid(csv_path):
            widget.emit("Csv path is not valid!")
            Popup("Error", 400, 200).show("Csv path is not valid!")
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

        if not check_if_int(division_size):
            widget.emit("Division size must be an integer!")
            Popup("Error", 400, 200).show("Division size must be an integer!")
            return

        if not check_if_T_F(only_two_subjects):
            widget.emit("Only two subjects must be either 'T'(true) of 'F'(false)!")
            Popup("Error", 400, 200).show("Only two subjects must be either 'T'(true) of 'F'(false)!")
            return

        if not check_if_T_F(with_visdom):
            widget.emit("With visdom must be either 'T'(true) of 'F'(false)!")
            Popup("Error", 400, 200).show("With visdom must be either 'T'(true) of 'F'(false)!")
            return

        if not check_if_T_F(save_loader):
            widget.emit("Save loaders must be either 'T'(true) of 'F'(false)!")
            Popup("Error", 400, 200).show("Save loaders must be either 'T'(true) of 'F'(false)!")
            return

        # start dec
        thread = threading.Thread(target = dual_ensemble_classifier, args = (
            data_path,
            csv_path,
            output_path,
            int(window_size),
            int(window_offset),
            int(division_size),
            only_two_subjects == 'T',
            with_visdom == 'T',
            save_loader == 'T',
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
        self.top_module.geometry(f'{850}x{820}')
