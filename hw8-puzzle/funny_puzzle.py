import heapq
from enum import Enum

WIDTH = 3

class Direction(Enum):
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4


def get_manhattan_distance(from_state, to_state=[1, 2, 3, 4, 5, 6, 7, 0, 0]):
    """
    INPUT: 
        Two states (if second state is omitted then it is assumed that it is the goal state)

    RETURNS:
        A scalar that is the sum of Manhattan distances for all tiles.
    """
    distance = x1 = y1 = x2 = y2 = 0
    for num in range(1,8):
        for r in range(WIDTH):
            for c in range(WIDTH):
                idx = WIDTH * r + c
                if from_state[idx] == num:
                    x1 = c
                    y1 = r
                if to_state[idx] == num:
                    x2 = c
                    y2 = r
        distance += abs(x1-x2) + abs(y1-y2)

    return distance


def print_succ(state):
    """
    INPUT: 
        A state (list of length 9)

    WHAT IT DOES:
        Prints the list of all the valid successors in the puzzle. 
    """
    succ_states = get_succ(state)

    for succ_state in succ_states:
        print(succ_state, "h={}".format(get_manhattan_distance(succ_state)))


def get_succ(state):
    """
    INPUT: 
        A state (list of length 9)

    RETURNS:
        A list of all the valid successors in the puzzle (don't forget to sort the result as done below). 
    """
    succ_states = []

    # go through each cell
    for r in range(WIDTH):
        for c in range(WIDTH):
            if state[WIDTH * r + c] != 0:
                # check if current cell can move and which directions
                directions = helper_movable_dirs(state, r, c)

                # move the cell in those directions and add to succ_states
                for dir in directions:
                    succ_state = helper_move(state, r, c, dir)
                    succ_states.append(succ_state)

    return sorted(succ_states)


def helper_movable_dirs(state, row, col):
    directions = []

    # check up
    if row - 1 >= 0:
        idx = WIDTH * (row - 1) + col
        if state[idx] == 0:
            directions.append(Direction.UP)

    # check down
    if row + 1 < WIDTH:
        idx = WIDTH * (row + 1) + col
        if state[idx] == 0:
            directions.append(Direction.DOWN)
    
    # check left
    if col - 1 >= 0:
        idx = WIDTH * row + (col - 1)
        if state[idx] == 0:
            directions.append(Direction.LEFT)

    # check right
    if col + 1 < WIDTH:
        idx = WIDTH * row + (col + 1)
        if state[idx] == 0:
            directions.append(Direction.RIGHT)
    
    return directions


def helper_move(state, row, col, direction):
    succ_state = state.copy()
    old_idx = WIDTH * row + col

    if direction == Direction.UP:
        new_idx = WIDTH * (row - 1) + col
    elif direction == Direction.DOWN:
        new_idx = WIDTH * (row + 1) + col
    elif direction == Direction.LEFT:
        new_idx = WIDTH * row + (col - 1)
    elif direction == Direction.RIGHT:
        new_idx = WIDTH * row + (col + 1)

    succ_state[new_idx] = succ_state[old_idx]
    succ_state[old_idx] = 0

    return succ_state


def solve(state, goal_state=[1, 2, 3, 4, 5, 6, 7, 0, 0]):
    """
    INPUT: 
        An initial state (list of length 9)

    WHAT IT SHOULD DO:
        Prints a path of configurations from initial state to goal state along  h values, number of moves, and max queue number in the format specified in the pdf.
    """

if __name__ == "__main__":
    """
    Feel free to write your own test code here to exaime the correctness of your functions. 
    Note that this part will not be graded.
    2 5 1
    4 0 6
    7 0 3
    """
    print_succ([2,5,1,4,0,6,7,0,3])
    print()

    print(get_manhattan_distance([2,5,1,4,0,6,7,0,3], [1, 2, 3, 4, 5, 6, 7, 0, 0]))
    print()

    # solve([2,5,1,4,0,6,7,0,3])
    # print()
