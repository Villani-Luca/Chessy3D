import tkinter
from tkinter import ttk

class HomeControls(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent, style="Card.TFrame", padding=15)

        self.columnconfigure(0, weight=1)

        self.add_widgets()

    def add_widgets(self):
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(expand=True, fill="both")

        for n in range(1, 3):
            setattr(self, f"tab_{n}", ttk.Frame(self.notebook))
            self.notebook.add(getattr(self, f"tab_{n}"), text=f"Tab {n}")

    def setup_first_tab(self):
        setattr(self, f"tab_{n}", ttk.Frame(self.notebook))
        self.notebook.add(getattr(self, f"tab_{n}"), text=f"Tab {n}")

        self.button = ttk.Button(self, text="Click me!")

    def setup_second_tab(self):
        setattr(self, f"tab_{n}", ttk.Frame(self.notebook))
        self.notebook.add(getattr(self, f"tab_2"), text=f"Tab 2")