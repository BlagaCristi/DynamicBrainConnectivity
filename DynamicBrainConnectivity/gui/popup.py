import tkinter


class Popup:

    def __init__(self, name, width, height):
        # define main window
        self.top_module = tkinter.Tk(className = name)

        # set window size
        self.top_module.geometry(f'{width}x{height}')

        # don't allow resize
        self.top_module.resizable(False, False)

    def show(self, message):
        # text to be shown
        text = tkinter.Label(master = self.top_module, text = message)
        text.pack()
        text.config(text = message)

        # center text
        text.grid(column = 0, row = 0)
        self.top_module.columnconfigure(0, weight = 1)
        self.top_module.rowconfigure(0, weight = 1)

        # display
        self.top_module.mainloop()
