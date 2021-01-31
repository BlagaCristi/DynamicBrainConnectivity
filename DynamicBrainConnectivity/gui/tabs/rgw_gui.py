import threading
import tkinter
from tkinter import filedialog

from gui.logging_text_handler import TextHandler
from gui.popup import Popup
from gui.validator import check_if_path_valid, check_if_int, check_if_trial_or_window, check_if_T_F
from recurrentGraphWavenet.recurrent_graph_wavenet import recurrent_graph_wavenet


class RGWGui:
    def __init__(self, tab_rgw, top_module):
        # set tab
        self.tab_rgw = tab_rgw

        # set parent
        self.top_module = top_module

        self.tab_rgw.bind("<Visibility>", self.on_visibility)

        # set up tab
        self.rgw_init()

    def rgw_init(self):
        # FOLDER DATA PATH
        folder_path_data = tkinter.StringVar()

        label_data = tkinter.Label(master = self.tab_rgw)
        label_data.place(x = 20, y = 20)
        label_data.config(text = "EEG Dots folder")

        textfield_data = tkinter.Entry(master = self.tab_rgw, width = 100, state = 'disabled')
        textfield_data.place(x = 140, y = 20)

        browse_button_data = tkinter.Button(master = self.tab_rgw, text = "Browse",
                                            command = lambda: self.folder_browse_button(folder_path_data,
                                                                                        textfield_data))
        browse_button_data.place(x = 750, y = 15)
        browse_button_data.config(width = 10, height = 1)

        # TRIAL WINDOW CONFIGURATION PATH
        trial_window_configuration_file_path = tkinter.StringVar()

        trial_window_configuration_label = tkinter.Label(master = self.tab_rgw,
                                                         text = "Trial config path")
        trial_window_configuration_label.place(x = 20, y = 80)
        trial_window_configuration_label.config(text = "Trial config path")

        textfield_division_file = tkinter.Entry(master = self.tab_rgw, width = 100, state = 'disabled')
        textfield_division_file.place(x = 140, y = 80)

        browse_button_csv = tkinter.Button(master = self.tab_rgw, text = "Browse",
                                           command = lambda: self.folder_browse_button(
                                               trial_window_configuration_file_path,
                                               textfield_division_file))
        browse_button_csv.place(x = 750, y = 75)
        browse_button_csv.config(width = 10, height = 1)

        # FUNCTIONAL NETWORK PATH
        functional_network_path = tkinter.StringVar()

        label_functional_network = tkinter.Label(master = self.tab_rgw, text = "FN path")
        label_functional_network.place(x = 20, y = 140)
        label_functional_network.config(text = "FN path")

        textfield_functional_network = tkinter.Entry(master = self.tab_rgw, width = 100, state = 'disabled')
        textfield_functional_network.place(x = 140, y = 140)

        browse_button_functional_network = tkinter.Button(master = self.tab_rgw, text = "Browse",
                                                          command = lambda: self.folder_browse_button(
                                                              functional_network_path,
                                                              textfield_functional_network))
        browse_button_functional_network.place(x = 750, y = 135)
        browse_button_functional_network.config(width = 10, height = 1)

        # PREVIOUS WEIGHT MATRIX PATH
        previous_matrix_path = tkinter.StringVar()

        label_weight_matrix = tkinter.Label(master = self.tab_rgw, text = "Previous matrix path")
        label_weight_matrix.place(x = 20, y = 200)
        label_weight_matrix.config(text = "Previous matrix path")

        textfield_previous_matrix = tkinter.Entry(master = self.tab_rgw, width = 100, state = 'disabled')
        textfield_previous_matrix.place(x = 140, y = 200)

        browse_button_output = tkinter.Button(master = self.tab_rgw, text = "Browse",
                                              command = lambda: self.folder_browse_button(previous_matrix_path,
                                                                                          textfield_previous_matrix))
        browse_button_output.place(x = 750, y = 195)
        browse_button_output.config(width = 10, height = 1)

        # OUTPUT PATH
        output_path = tkinter.StringVar()

        label_output = tkinter.Label(master = self.tab_rgw, text = "Output path")
        label_output.place(x = 20, y = 260)
        label_output.config(text = "Output path")

        textfield_output = tkinter.Entry(master = self.tab_rgw, width = 100, state = 'disabled')
        textfield_output.place(x = 140, y = 260)

        browse_button_output = tkinter.Button(master = self.tab_rgw, text = "Browse",
                                              command = lambda: self.folder_browse_button(output_path,
                                                                                          textfield_output))
        browse_button_output.place(x = 750, y = 255)
        browse_button_output.config(width = 10, height = 1)

        # SUBJECT NUMBER
        label_subject_number = tkinter.Label(master = self.tab_rgw,
                                             text = "Subject number")
        label_subject_number.place(x = 20, y = 320)
        label_subject_number.config(text = "Subject number")

        textfield_subject_number = tkinter.Entry(master = self.tab_rgw, width = 10)
        textfield_subject_number.place(x = 140, y = 320)

        # TRIAL INDEX
        label_trial_index = tkinter.Label(master = self.tab_rgw,
                                          text = "Trial index")
        label_trial_index.place(x = 250, y = 320)
        label_trial_index.config(text = "Trial index")

        textfield_trial_index = tkinter.Entry(master = self.tab_rgw, width = 10)
        textfield_trial_index.place(x = 330, y = 320)

        # WINDOW INDEX
        label_window_index = tkinter.Label(master = self.tab_rgw,
                                           text = "Window index")
        label_window_index.place(x = 430, y = 320)
        label_window_index.config(text = "Window index")

        textfield_window_index = tkinter.Entry(master = self.tab_rgw, width = 10)
        textfield_window_index.place(x = 520, y = 320)

        # BATCH SIZE
        label_batch_size = tkinter.Label(master = self.tab_rgw,
                                         text = "Batch size")
        label_batch_size.place(x = 610, y = 320)
        label_batch_size.config(text = "Batch size")

        textfield_batch_size = tkinter.Entry(master = self.tab_rgw, width = 10)
        textfield_batch_size.place(x = 700, y = 320)

        # INPUT LENGTH
        label_input_length = tkinter.Label(master = self.tab_rgw,
                                           text = "Input length")
        label_input_length.place(x = 20, y = 380)
        label_input_length.config(text = "Input length")

        textfield_input_length = tkinter.Entry(master = self.tab_rgw, width = 10)
        textfield_input_length.place(x = 140, y = 380)

        # OUTPUT LENGTH
        label_output_length = tkinter.Label(master = self.tab_rgw,
                                            text = "Output length")
        label_output_length.place(x = 240, y = 380)
        label_output_length.config(text = "Output length")

        textfield_output_length = tkinter.Entry(master = self.tab_rgw, width = 10)
        textfield_output_length.place(x = 330, y = 380)

        # NR OF BLOCKS
        label_nr_blocks = tkinter.Label(master = self.tab_rgw,
                                        text = "Blocks")
        label_nr_blocks.place(x = 450, y = 380)
        label_nr_blocks.config(text = "Blocks")

        textfield_nr_blocks = tkinter.Entry(master = self.tab_rgw, width = 10)
        textfield_nr_blocks.place(x = 520, y = 380)

        # NR OF LAYERS
        label_nr_layers = tkinter.Label(master = self.tab_rgw,
                                        text = "Layers")
        label_nr_layers.place(x = 620, y = 380)
        label_nr_layers.config(text = "Layers")

        textfield_nr_layers = tkinter.Entry(master = self.tab_rgw, width = 10)
        textfield_nr_layers.place(x = 700, y = 380)

        # LOADER OPTION
        label_loader_option = tkinter.Label(master = self.tab_rgw,
                                            text = "Loader option")
        label_loader_option.place(x = 20, y = 440)
        label_loader_option.config(text = "Loader option")

        textfield_loader_option = tkinter.Entry(master = self.tab_rgw, width = 10)
        textfield_loader_option.place(x = 140, y = 440)

        # NR OF EPOCHS
        label_epochs = tkinter.Label(master = self.tab_rgw,
                                     text = "Epochs")
        label_epochs.place(x = 260, y = 440)
        label_epochs.config(text = "Epochs")

        textfield_epochs = tkinter.Entry(master = self.tab_rgw, width = 10)
        textfield_epochs.place(x = 330, y = 440)

        # INITIAL TRAIN PERCENTAGE
        label_train_percentage = tkinter.Label(master = self.tab_rgw,
                                               text = "roFV train")
        label_train_percentage.place(x = 448, y = 440)
        label_train_percentage.config(text = "roFV train")

        textfield_train_percentage = tkinter.Entry(master = self.tab_rgw, width = 10)
        textfield_train_percentage.place(x = 520, y = 440)

        # INITIAL CROSS PERCENTAGE
        label_cross_percentage = tkinter.Label(master = self.tab_rgw,
                                               text = "roFV cross")
        label_cross_percentage.place(x = 610, y = 440)
        label_cross_percentage.config(text = "roFV cross")

        textfield_cross_percentage = tkinter.Entry(master = self.tab_rgw, width = 10)
        textfield_cross_percentage.place(x = 700, y = 440)

        # USE FN
        label_use_fn = tkinter.Label(master = self.tab_rgw,
                                     text = "Use FN")
        label_use_fn.place(x = 30, y = 500)
        label_use_fn.config(text = "Use FN")

        textfield_use_fn = tkinter.Entry(master = self.tab_rgw, width = 10)
        textfield_use_fn.place(x = 140, y = 500)

        # USE PREVIOUS MATRIX
        label_previous_matrix = tkinter.Label(master = self.tab_rgw,
                                              text = "Use weight matrix")
        label_previous_matrix.place(x = 230, y = 500)
        label_previous_matrix.config(text = "Use weight matrix")

        textfield_use_previous_matrix = tkinter.Entry(master = self.tab_rgw, width = 10)
        textfield_use_previous_matrix.place(x = 330, y = 500)

        # INCLUDE CROSS
        label_include_cross = tkinter.Label(master = self.tab_rgw,
                                            text = "Include cross")
        label_include_cross.place(x = 440, y = 500)
        label_include_cross.config(text = "Include cross")

        textfield_include_cross = tkinter.Entry(master = self.tab_rgw, width = 10)
        textfield_include_cross.place(x = 520, y = 500)

        # USE GPU
        label_use_gpu = tkinter.Label(master = self.tab_rgw,
                                      text = "Use GPU")
        label_use_gpu.place(x = 620, y = 500)
        label_use_gpu.config(text = "Use GPU")

        textfield_use_gpu = tkinter.Entry(master = self.tab_rgw, width = 10)
        textfield_use_gpu.place(x = 700, y = 500)

        # LOGGING AREA
        text_area = tkinter.Text(self.tab_rgw, height = 10, width = 100)
        text_area.place(x = 20, y = 560)
        widget = TextHandler(text_area)

        # START BUTTON
        start_button = tkinter.Button(master = self.tab_rgw, text = "Start",
                                      command = lambda: self.start_rgw(data_path = folder_path_data.get(),
                                                                       trial_window_configuration_path = trial_window_configuration_file_path.get(),
                                                                       functional_network_path = functional_network_path.get(),
                                                                       previous_weight_matrix_path = previous_matrix_path.get(),
                                                                       output_path = output_path.get(),
                                                                       subject_number = textfield_subject_number.get(),
                                                                       trial_index = textfield_trial_index.get(),
                                                                       window_index = textfield_window_index.get(),
                                                                       batch_size = textfield_batch_size.get(),
                                                                       input_length = textfield_input_length.get(),
                                                                       output_length = textfield_output_length.get(),
                                                                       nr_of_blocks = textfield_nr_blocks.get(),
                                                                       nr_of_layers = textfield_nr_layers.get(),
                                                                       loader_option = textfield_loader_option.get(),
                                                                       nr_of_epochs = textfield_epochs.get(),
                                                                       initial_train_percentage = textfield_train_percentage.get(),
                                                                       initial_cross_percentage = textfield_cross_percentage.get(),
                                                                       use_functional_network = textfield_use_fn.get(),
                                                                       use_previous_matrix = textfield_use_previous_matrix.get(),
                                                                       include_cross = textfield_include_cross.get(),
                                                                       use_gpu = textfield_use_gpu.get(),
                                                                       widget = widget))

        start_button.place(x = 375, y = 750)
        start_button.config(width = 10, height = 1)

    def start_rgw(self, data_path, trial_window_configuration_path, functional_network_path,
                  previous_weight_matrix_path,
                  output_path, subject_number,
                  trial_index, window_index, batch_size, input_length, output_length, nr_of_blocks, nr_of_layers,
                  loader_option, nr_of_epochs,
                  initial_train_percentage, initial_cross_percentage, use_functional_network, use_previous_matrix,
                  include_cross, use_gpu, widget = None):

        if not check_if_path_valid(data_path):
            widget.emit("EEG path is not valid!")
            Popup("Error", 400, 200).show("EEG path is not valid!")
            return

        if not check_if_path_valid(trial_window_configuration_path):
            widget.emit("Trial window configuration path is not valid!")
            Popup("Error", 400, 200).show("Trial window configuration path is not valid!")
            return

        if not check_if_path_valid(functional_network_path):
            widget.emit("Functional network path is not valid!")
            Popup("Error", 400, 200).show("Functional network path is not valid!")
            return

        if not check_if_path_valid(previous_weight_matrix_path):
            widget.emit("Previous weight matrix path is not valid!")
            Popup("Error", 400, 200).show("Previous weight matrix path is not valid!")
            return

        if not check_if_path_valid(output_path):
            widget.emit("Output path is not valid!")
            Popup("Error", 400, 200).show("Output path is not valid!")
            return

        if not check_if_int(subject_number):
            widget.emit("Subject number must be an integer!")
            Popup("Error", 400, 200).show("Subject number must be an integer!")
            return

        if not check_if_int(trial_index):
            widget.emit("Trial index must be an integer!")
            Popup("Error", 400, 200).show("Trial index must be an integer!")
            return

        if not check_if_int(window_index):
            widget.emit("Window index must be an integer!")
            Popup("Error", 400, 200).show("Window index must be an integer!")
            return

        if not check_if_int(batch_size):
            widget.emit("Batch size must be an integer!")
            Popup("Error", 400, 200).show("Batch size must be an integer!")
            return

        if not check_if_int(input_length):
            widget.emit("Input length must be an integer!")
            Popup("Error", 400, 200).show("Input length must be an integer!")
            return

        if not check_if_int(output_length):
            widget.emit("Output length must be an integer!")
            Popup("Error", 400, 200).show("Output length must be an integer!")
            return

        if not check_if_int(nr_of_blocks):
            widget.emit("Number of blocks must be an integer!")
            Popup("Error", 400, 200).show("Number of blocks must be an integer!")
            return

        if not check_if_int(nr_of_layers):
            widget.emit("Number of layers must be an integer!")
            Popup("Error", 400, 200).show("Number of layers must be an integer!")
            return

        if not check_if_trial_or_window(loader_option):
            widget.emit("Loader option must be either 'Trial' or 'Window'!")
            Popup("Error", 400, 200).show("Loader option must be either 'Trial' or 'Window'!")
            return

        if not check_if_int(nr_of_epochs):
            widget.emit("Number of epochs must be an integer!")
            Popup("Error", 400, 200).show("Number of epochs must be an integer!")
            return

        if not check_if_int(initial_train_percentage):
            widget.emit("Initial train percentage must be an integer!")
            Popup("Error", 400, 200).show("Initial train percentage must be an integer!")
            return

        if not check_if_int(initial_cross_percentage):
            widget.emit("Initial train cross must be an integer!")
            Popup("Error", 400, 200).show("Initial train cross must be an integer!")
            return

        if not check_if_int(initial_cross_percentage):
            widget.emit("Initial train cross must be an integer!")
            Popup("Error", 400, 200).show("Initial train cross must be an integer!")
            return

        if not check_if_T_F(use_functional_network):
            widget.emit("Use functional network must be either 'T'(true) of 'F'(false)!")
            Popup("Error", 400, 200).show(
                "Use functional network must be either 'T'(true) of 'F'(false)!")
            return

        if not check_if_T_F(use_previous_matrix):
            widget.emit("Use previous matrix must be either 'T'(true) of 'F'(false)!")
            Popup("Error", 400, 200).show(
                "Use previous matrix must be either 'T'(true) of 'F'(false)!")
            return

        if not check_if_T_F(include_cross):
            widget.emit("Include cross must be either 'T'(true) of 'F'(false)!")
            Popup("Error", 400, 200).show(
                "Include cross must be either 'T'(true) of 'F'(false)!")
            return

        if not check_if_T_F(use_gpu):
            widget.emit("Use GPU must be either 'T'(true) of 'F'(false)!")
            Popup("Error", 400, 200).show(
                "Use GPU must be either 'T'(true) of 'F'(false)!")
            return

        # start rgw
        thread = threading.Thread(target = recurrent_graph_wavenet, args = (
            data_path,
            trial_window_configuration_path,
            output_path,
            int(subject_number),
            int(trial_index),
            int(window_index),
            int(input_length),
            int(output_length),
            int(batch_size),
            loader_option,
            widget,
            int(nr_of_blocks),
            int(nr_of_layers),
            int(nr_of_epochs),
            int(initial_train_percentage),
            int(initial_cross_percentage),
            use_functional_network == 'T',
            functional_network_path,
            use_previous_matrix == 'T',
            previous_weight_matrix_path,
            include_cross == 'T',
            use_gpu == 'T'
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
        self.top_module.geometry(f'{850}x{840}')
