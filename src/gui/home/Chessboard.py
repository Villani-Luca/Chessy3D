import tkinter as tk

import chess
from PIL import ImageTk

class Chessboard(tk.Frame):
    color1 = "white"
    color2 = "grey"

    rows = 8
    columns = 8

    @property
    def canvas_size(self):
        return self.columns * self.square_size, self.rows * self.square_size

    def __init__(self, parent, chessboard: chess.Board, square_size=64):

        self.icons: dict[str, tk.PhotoImage] = {}
        self.chessboard = chessboard
        self.square_size = square_size
        self.parent = parent

        canvas_width = self.columns * square_size
        canvas_height = self.rows * square_size

        tk.Frame.__init__(self, parent)

        self.canvas = tk.Canvas(self, width=canvas_width, height=canvas_height, background="grey")
        self.canvas.pack(side="top", fill="both", anchor="center", expand=True)

        self.canvas.bind("<Configure>", self.refresh)

        self.statusbar = tk.Frame(self, height=64)

        self.label_status = tk.Label(self.statusbar, text="   White's turn  ", fg="black")
        self.label_status.pack(side=tk.LEFT, expand=0, in_=self.statusbar)

        self.button_quit = tk.Button(self.statusbar, text="Quit", fg="black", command=self.parent.destroy)
        self.button_quit.pack(side=tk.RIGHT, in_=self.statusbar)
        self.statusbar.pack(expand=False, fill="x", side='bottom')


    #def hilight(self, pos):
    #    piece = self.chessboard[pos]
    #    if piece is not None and (piece.color == self.chessboard.player_turn):
    #        self.selected_piece = (self.chessboard[pos], pos)
    #        self.hilighted = map(self.chessboard.number_notation, (self.chessboard[pos].possible_moves(pos)))

    def refresh(self, event):
        '''Redraw the board'''
        self.canvas.delete("square")
        color = self.color2
        for row in range(self.rows):
            color = self.color1 if color == self.color2 else self.color2
            for col in range(self.columns):
                x1 = (col * self.square_size)
                y1 = ((7-row) * self.square_size)
                x2 = x1 + self.square_size
                y2 = y1 + self.square_size

                self.canvas.create_rectangle(x1, y1, x2, y2, outline="", fill=color, tags="square")
                color = self.color1 if color == self.color2 else self.color2

                piece = self.chessboard.piece_at(row*8+col)
                if piece is not None:
                    piece_id = piece.piece_type + (6 if piece.color else 0)
                    filename = rf"D:/Projects/Uni/Chessy3D/src/gui/assets/pieces/{piece_id}.png"
                    if filename not in self.icons:
                        self.icons[filename] = ImageTk.PhotoImage(file=filename, width=32, height=32)

                    img = self.icons[filename]
                    self.canvas.create_image(self.square_size * col + 32, self.square_size * row + 32, image=img, tags=(piece_id, "piece"), anchor="c")

        self.canvas.tag_raise("piece")
        self.canvas.tag_lower("square")

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Simple Python Chess")

    gui = Chessboard(root, chess.Board())
    gui.pack(side="top", fill="both", expand="true", padx=4, pady=4)

    #root.resizable(0,0)
    root.mainloop()