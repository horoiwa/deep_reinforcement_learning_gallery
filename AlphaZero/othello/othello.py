import functools
import copy


N_COLS = N_ROWS = 6


def xy_to_idx(row, col):
    return N_ROWS * row + col


def idx_to_xy(i):
    x, y = i // N_ROWS, i % N_ROWS
    return x, y


def get_initial_state():

    state = [0] * (N_ROWS * N_COLS)

    state[xy_to_idx(N_ROWS // 2 - 1, N_COLS // 2 - 1)] = 1
    state[xy_to_idx(N_ROWS // 2, N_COLS // 2)] = 1
    state[xy_to_idx(N_ROWS // 2 - 1, N_COLS // 2)] = -1
    state[xy_to_idx(N_ROWS // 2, N_COLS // 2 - 1)] = -1

    return state


@functools.lru_cache(maxsize=128)
def get_directions(i):

    _, col = idx_to_xy(i)

    up = list(range(i, -1, -N_COLS))[1:]
    down = list(range(i, N_ROWS*N_COLS, N_COLS))[1:]

    left = list(reversed(range(i-col, i, 1)))
    right = list(range(i+1, i + N_COLS - col))

    ul = list(range(i, -1, -N_COLS-1))[1:col+1]
    ur = list(range(i, -1, -N_COLS+1))[1:N_COLS - col]

    ll = list(range(i, N_ROWS * N_COLS, N_COLS-1))[1:col+1]
    lr = list(range(i, N_ROWS * N_COLS, N_COLS+1))[1:N_COLS - col]

    return [up, down, left, right, ul, ur, ll, lr]


def is_valid_action(state: list, action: int, player: int):

    #: すでに石がある
    if state[action] != 0:
        return False

    directions = get_directions(action)

    for direction in directions:
        stones = [state[i] for i in direction]
        if player in stones and -player in stones:
            stones = stones[:stones.index(player)]
            if stones and all(i == -player for i in stones):
                return True

    return False


def get_valid_actions(state: list, player: int):

    valid_actions = [action for action in range(N_ROWS * N_COLS)
                     if is_valid_action(state, action, player)]

    return valid_actions


def get_next_state(state: list, action: int, player: int):

    assert is_valid_action(state, action, player)
    next_state = copy.deepcopy(state)

    directions = get_directions(action)

    for direction in directions:
        stones = [state[i] for i in direction]
        if player in stones and -player in stones:
            idx = stones.index(player)
            stones = stones[:idx]
            if stones and all(i == -player for i in stones):
                for i in direction[:idx]:
                    next_state[i] = player

    next_state[action] = player

    return next_state


def count_stone(state: list):
    first = sum([1 for i in state if i == 1])
    second = sum([1 for i in state if i == -1])
    return (first, second)

