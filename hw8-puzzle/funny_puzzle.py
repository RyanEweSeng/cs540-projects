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
    open = list()
    close = set()
    trace = list() # trace exists to track the nodes we consider part of the best solution
    g = 0
    parent_index = -1
    id = 1
    prev_id = -1
    max_length = 0

    # entry is in the form (cost, state, (g, h, parent_index), id, prev_id)
    # id serves to differentiate between entries with the same parent_index
    # prev_id serves to point to the entry that preceeded it
    # used this method because parent index was not sufficient to determine which was the preceeding entry
    h = get_manhattan_distance(state)
    heapq.heappush(open, (g + h, state, (g, h, parent_index), 0, prev_id))
    close.add(tuple(state)) # lists can't be added to sets so typecast to a tuple

    while open:
        max_length = max(max_length, len(open))

        #print('heap:')
        #print(*open, sep='\n')
        #print()

        curr_entry = heapq.heappop(open)
        curr_state = curr_entry[1]
        curr_g = curr_entry[2][0]
        curr_parent_index = curr_entry[2][2]
        curr_id = curr_entry[3]

        close.add(tuple(curr_state)) # once we popped from the queue, we add it to the close/visited set

        #print(curr_state)
        #print('---------------------------')

        if curr_state == goal_state:
            break

        curr_successors = get_succ(curr_state)
        g = curr_g + 1
        parent_index = curr_parent_index + 1
        prev_id = curr_id # the prev_id for the successor(s) should be the curr_id of the curr_entry
        for succ in curr_successors:
            if tuple(succ) not in close:
                #print(succ, g, h, parent_index, id, prev_id)
                h = get_manhattan_distance(succ)
                heapq.heappush(open, (g + h, succ, (g, h, parent_index), id, prev_id))
                id += 1 # update the id value
        trace.append(curr_entry) # add the curr_entry as it is a candidate for the best solution
        
        #print()
        #print('trace:')
        #print(*trace, sep='\n')
        #print()

    final_path = [curr_state]
    trace_parent_idx = curr_parent_index
    trace_prev_id = curr_entry[4]
    while trace_parent_idx != -1:
        for entry in trace:
            if entry[2][2] == trace_parent_idx - 1 and entry[3] == trace_prev_id: # finds the parent index and the connecting id
                final_path.append(entry[1]) # this will build us the best final path
                trace_parent_idx = entry[2][2]
                trace_prev_id = entry[4]
                break

    move = 0
    while final_path:
        node = final_path.pop()
        print(node, "h={} moves: {}".format(get_manhattan_distance(node), move))
        move += 1
    print("Max queue length:", max_length)
        

if __name__ == "__main__":
    """
    Feel free to write your own test code here to exaime the correctness of your functions. 
    Note that this part will not be graded.
    2 5 1
    4 0 6
    7 0 3
    """
    # print_succ([2,5,1,4,0,6,7,0,3])
    # print()

    # print(get_manhattan_distance([2,5,1,4,0,6,7,0,3], [1, 2, 3, 4, 5, 6, 7, 0, 0]))
    # print()

    test_cases = [
        [4, 3, 0, 5, 1, 6, 7, 2, 0],
        [3, 4, 6, 0, 0, 1, 7, 2, 5],
        [6, 0, 0, 3, 5, 1, 7, 2, 4],
        [0, 4, 7, 1, 3, 0, 6, 2, 5],
        [5, 2, 3, 0, 6, 4, 7, 1, 0],
        [1, 7, 0, 6, 3, 2, 0, 4, 5]
    ]
    i = 0
    for state in test_cases:
        print('test case #',i,':')
        solve(state)
        print()
        i += 1
