import tkinter
from tkinter import ttk

import sv_ttk

from src.gui.home.Home import Home

class App(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent, padding=15)

        self.columnconfigure(0, weight=1)

        Home(self).grid(row=0, column=0, sticky="nsew")


def main():
    root = tkinter.Tk()
    root.title("")

    sv_ttk.set_theme("dark")

    App(root).pack(expand=True, fill="both")

    root.mainloop()


if __name__ == "__main__":
    main()