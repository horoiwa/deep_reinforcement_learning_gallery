import tkinter as tk
import random
import time

import numpy as np

import othello
from network import AlphaZeroNetwork
from mcts import MCTS


class Othello(tk.Frame):

    def __init__(self, npc_type, weights_path):
        """
            どちらかがパス（合法手なし）時点でゲーム終了,
            その時点で石の多い方が勝ち
            人間が先手で固定
        """
        tk.Frame.__init__(self, master=None)

        assert npc_type in ["random", "eps-greedy", "alphazero"]

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
        self.weights_path = weights_path
        self.epsilon = 0.5
        if self.npc_type == "alphazero":
            self.setup()

        self.human = 1
        self.npc = -1

        self.is_gameend = False
        self.reset()

    def setup(self):

        state = othello.get_initial_state()

        self.network = AlphaZeroNetwork(action_space=othello.ACTION_SPACE)

        self.network.predict(othello.encode_state(state, 1))

        self.network.load_weights(self.weights_path)

        self.mcts = MCTS(network=self.network, alpha=0.15)

    def reset(self):
        self.is_gameend = False
        self.state = othello.get_initial_state()
        self.refresh()
        self.is_player_turn = True
        self.update_label()

    def player_action(self, event):

        if not self.is_player_turn or self.is_gameend:
            return
        else:
            self.is_player_turn = False

        print("Player action")

        row = event.y // 100
        col = event.x // 100

        action = othello.xy_to_idx(row, col)

        valid_actions = othello.get_valid_actions(self.state, self.human)
        print(valid_actions, action)

        if action in valid_actions:

            self.state, done = othello.step(self.state, action, self.human)
            self.refresh()
            self.update_label()
            if done:
                self.update_label(game_end=True)
                return

            time.sleep(0.3)

            self.npc_action()
            if self.is_gameend:
                return

        else:
            print("Invalid action")

        self.is_player_turn = True

        return

    def npc_action(self):
        print("NPC action")

        valid_actions = othello.get_valid_actions(self.state, self.npc)

        if self.npc_type == "random":
            action = random.choice(valid_actions)
            self.state, done = othello.step(self.state, action, self.npc)

            self.refresh()
            self.update_label()
            if done:
                self.update_label(game_end=True)
                return

        elif self.npc_type == "eps-greedy":

            if random.random() > self.epsilon:
                best_action = None
                best_score = 0
                for action in valid_actions:
                    next_state, done = othello.step(self.state, action, self.npc)
                    _, score = othello.count_stone(next_state)
                    if score > best_score:
                        best_score = score
                        best_action = action

                self.state, done = othello.step(self.state, best_action, self.npc)
            else:
                action = random.choice(valid_actions)
                self.state, done = othello.step(self.state, action, self.npc)

            self.refresh()
            self.update_label()
            if done:
                self.update_label(game_end=True)
                return

        elif self.npc_type == "alphazero":
            mcts_policy = self.mcts.search(root_state=self.state,
                                           current_player=self.npc,
                                           num_simulations=50)

            action = np.argmax(mcts_policy)
            self.state, done = othello.step(self.state, action, self.npc)

            self.refresh()
            self.update_label()
            if done:
                self.update_label(game_end=True)
                return
        else:
            raise NotImplementedError()

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
            cy, cx = (100 * row + 10, 100 * col + 10)
            if v == 1:
                self.cv.create_oval(cx, cy, cx+80, cy+80, fill="black")
            elif v == -1:
                self.cv.create_oval(cx, cy, cx+80, cy+80, fill="white")

    def update_label(self, game_end=False):
        first, second = othello.count_stone(self.state)
        if not game_end:
            self.label.configure(text=f"[You]  {first} - {second} [NPC]")
        else:
            message = "Human win" if first > second else "NPC win"
            self.label.configure(
                text=f"[You]  {first} - {second} [NPC] {message}")
            self.is_player_turn = False
            self.is_gameend = True


if __name__ == "__main__":
    app = Othello(npc_type="alphazero", weights_path="checkpoints/network")
    app.pack()
    app.mainloop()
