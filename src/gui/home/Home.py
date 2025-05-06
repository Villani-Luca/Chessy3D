import tkinter
from tkinter import ttk

import chess

from src.gui.home.Chessboard import Chessboard
from src.gui.home.HomeControls import HomeControls


class Home(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)

        for index in range(2):
            self.columnconfigure(index, weight=1)


        Chessboard(self, chess.Board()).grid(row=0, column=1, rowspan=2, padx=10, pady=(10, 0), sticky="nsew")
        HomeControls(self).grid(row=0, column=3, rowspan=2, padx=10, pady=(10, 0), sticky="nsew")
