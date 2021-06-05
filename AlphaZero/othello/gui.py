import tkinter as tk
import time

import numpy as np

import othello


class Othello(tk.Frame):

    def __init__(self, npc_type="random", model_path=None):

        tk.Frame.__init__(self, master=None)

        self.master.title("app")

        self.w, self.h = (othello.N_COLS * 100, othello.N_ROWS * 100)
        self.cv = tk.Canvas(self, width=self.w, height=self.h)
        self.cv.bind("<Button-1>", self.player_action)
        self.cv.pack()

        self.label = tk.Label(
            self, text="", height=2, padx=10,
            justify="left", anchor="w", font=("", 12))
        self.label.pack(fill="both")

        self.npc_type = npc_type
        self.model_path = model_path
        self.reset()

    def reset(self):
        self.state = othello.get_initial_state()
        self.refresh()
        self.is_player_turn = True
        self.update_label()

    def player_action(self, event):

        if not self.is_player_turn:
            return
        else:
            self.is_player_turn = False

        print("Player action")

        col = event.x // 100
        row = event.y // 100
        action = row * othello.N_ROWS + col

        valid_actions = othello.get_valid_actions(self.state, 1)
        print(valid_actions, action)

        if action in valid_actions:
            self.state = othello.step(self.state, action, 1)
            self.refresh()
            self.update_label()
            time.sleep(0.5)

            self.npc_action()
        else:
            print("Invalid action")

        self.is_player_turn = True
        return

    def npc_action(self):

        print("NPC action")

        self.refresh()
        self.update_label()

    def refresh(self):

        print(np.array(self.state).reshape(othello.N_ROWS, othello.N_COLS))

        self.cv.delete('all')
        self.cv.create_rectangle(0, 0, self.w, self.h, fill="#2f4f4f")
        self.cv.create_rectangle(0, self.h, self.w, self.h+50, fill="darkgrey")

        for i in range(othello.N_COLS):
            self.cv.create_line(0, i*100, self.w, i*100)
        for i in range(othello.N_ROWS):
            self.cv.create_line(i*100, 0, i*100, self.h)

        for i in range(othello.N_ROWS * othello.N_COLS):
            v = self.state[i]
            row, col = i // othello.N_ROWS, i % othello.N_COLS
            cx, cy = (100 * row + 10, 100 * col + 10)
            if v == 1:
                self.cv.create_oval(cx, cy, cx+80, cy+80, fill="black")
            elif v == -1:
                self.cv.create_oval(cx, cy, cx+80, cy+80, fill="white")

    def update_label(self):
        first, second = othello.count_stone(self.state)
        self.label.configure(text=f"[You]  {first} - {second} [NPC]")


if __name__ == "__main__":
    app = Othello()
    app.pack()
    app.mainloop()
