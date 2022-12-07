import random
import copy
import time

class TeekoPlayer:
    """ An object representation for an AI game player for the game Teeko.
    """
    board = [[' ' for j in range(5)] for i in range(5)]
    pieces = ['b', 'r']

    def __init__(self):
        """ Initializes a TeekoPlayer object by randomly selecting red or black as its
        piece color.
        """
        self.my_piece = random.choice(self.pieces)
        self.opp = self.pieces[0] if self.my_piece == self.pieces[1] else self.pieces[1]
        self.depth_limit = 2


    def make_move(self, state):
        """ Selects a (row, col) space for the next move. You may assume that whenever
        this function is called, it is this player's turn to move.

        Args:
            state (list of lists): should be the current state of the game as saved in
                this TeekoPlayer object. Note that this is NOT assumed to be a copy of
                the game state and should NOT be modified within this method (use
                place_piece() instead). Any modifications (e.g. to generate successors)
                should be done on a deep copy of the state.

                In the "drop phase", the state will contain less than 8 elements which
                are not ' ' (a single space character).

        Return:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.

        Note that without drop phase behavior, the AI will just keep placing new markers
            and will eventually take over the board. This is not a valid strategy and
            will earn you no points.
        """
        score = float("-inf")
        move = []
        possible_move = None
        possible_old = None
        
        local_time = time.ctime(time.time())
        print("Start time:", local_time)

        successors = self.succ(state, self.my_piece)
        for succ in successors:
            drop_phase = self.get_drop_phase(state)
            state_copy = copy.deepcopy(state)

            if drop_phase:
                state_copy[succ[0]][succ[1]] = self.my_piece
            else:
                state_copy[succ[0]][succ[1]] = ' '
                state_copy[succ[2]][succ[3]] = self.my_piece

            possible_score = self.max_value(state_copy, 0, float("-inf"), float("inf"))

            if possible_score > score:
                score = possible_score
                possible_move = (succ[2], succ[3])
                if not drop_phase:
                    possible_old = (succ[0], succ[1])

        if possible_old is not None:
            move.append(possible_move)
            move.append(possible_old)
        else:
            move.append(possible_move)

        local_time = time.ctime(time.time())
        print("End time:", local_time)
        return move


    def succ(self, state, teamcolor):
        # successors are in the form:
        #     (old_r, old_c, new_r, new_c) where new is the piece to place/move to 
        successors = []
        drop_phase = self.get_drop_phase(state)
        
        if drop_phase:
           # any empty cell is a possible location for a piece which is a successor
           for r in range(5):
               for c in range(5):
                   if state[r][c] == ' ':
                       successors.append([r, c, r, c])
        else:
            # look for the cells with pieces and check the adjacent moves
            for r in range(5):
                for c in range(5):
                    if state[r][c] == teamcolor:
                        moves = self.get_adj_moves(state, r, c)
                        for move in moves:
                            successors.append([r, c, move[0], move[1]])

        return successors


    def get_adj_moves(self, state, row, col):
        dim = 5
        moves = []

        if row + 1 < dim and state[row+1][col] == ' ': # DOWN
            moves.append((row+1, col))
        if row - 1 >= 0 and state[row-1][col] == ' ': # UP
            moves.append((row-1, col))
        if col + 1 < dim and state[row][col+1] == ' ': # RIGHT
            moves.append((row, col+1))
        if col - 1 >= 0 and state[row][col-1] == ' ': # LEFT
            moves.append((row, col-1))

        if row - 1 >= 0 and col - 1 >= 0 and state[row-1][col-1] == ' ': # TOP-LEFT
            moves.append((row-1, col-1))
        if row - 1 >= 0 and col + 1 < dim and state[row-1][col+1] == ' ': # TOP-RIGHT
            moves.append((row-1, col+1))
        if row + 1 < dim and col - 1 >= 0 and state[row+1][col-1] == ' ': # BOT-LEFT
            moves.append((row+1, col-1))
        if row + 1 < dim and col + 1 < dim and state[row+1][col+1] == ' ': # BOT-RIGHT
            moves.append((row+1, col+1))

        return moves


    def get_drop_phase(self, state):
        count = 0
        for r in range(5):
            for c in range(5):
                if state[r][c] != ' ':
                    count += 1

        return count < 8 


    def opponent_move(self, move):
        """ Validates the opponent's next move against the internal board representation.
        You don't need to touch this code.

        Args:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.
        """
        # validate input
        if len(move) > 1:
            source_row = move[1][0]
            source_col = move[1][1]
            if source_row != None and self.board[source_row][source_col] != self.opp:
                self.print_board()
                print(move)
                raise Exception("You don't have a piece there!")
            if abs(source_row - move[0][0]) > 1 or abs(source_col - move[0][1]) > 1:
                self.print_board()
                print(move)
                raise Exception('Illegal move: Can only move to an adjacent space')
        if self.board[move[0][0]][move[0][1]] != ' ':
            raise Exception("Illegal move detected")
        # make move
        self.place_piece(move, self.opp)


    def place_piece(self, move, piece):
        """ Modifies the board representation using the specified move and piece

        Args:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.

                This argument is assumed to have been validated before this method
                is called.
            piece (str): the piece ('b' or 'r') to place on the board
        """
        if len(move) > 1:
            self.board[move[1][0]][move[1][1]] = ' '
        self.board[move[0][0]][move[0][1]] = piece


    def print_board(self):
        """ Formatted printing for the board """
        for row in range(len(self.board)):
            line = str(row) + ": "
            for cell in self.board[row]:
                line += cell + " "
            print(line)
        print("   A B C D E")


    def game_value(self, state):
        """ Checks the current board status for a win condition

        Args:
        state (list of lists): either the current state of the game as saved in
            this TeekoPlayer object, or a generated successor state.

        Returns:
            int: 1 if this TeekoPlayer wins, -1 if the opponent wins, 0 if no winner
        """
        # check horizontal wins
        for row in state:
            for i in range(2):
                if row[i] != ' ' and row[i] == row[i + 1] == row[i + 2] == row[i + 3]:
                    return 1 if row[i] == self.my_piece else -1

        # check vertical wins
        for col in range(5):
            for i in range(2):
                if state[i][col] != ' ' and state[i][col] == state[i + 1][col] == state[i + 2][col] == state[i + 3][col]:
                    return 1 if state[i][col] == self.my_piece else -1

        # check \ diagonal wins
        for col in range(2):
            for i in range(0,2):
                if state[i][col] != ' ' and state[i][col] == state[i+1][col+1] == state[i+2][col+2] == state[i+3][col+3]:
                    return 1 if state[i][col] == self.my_piece else -1
        
        # check / diagonal wins
        for col in range(2):
            for i in range(3,5):
                if state[i][col] != ' ' and state[i][col] == state[i-1][col+1] == state[i-2][col+2] == state[i-3][col+3]:
                    return 1 if state[i][col] == self.my_piece else -1
        
        # check box wins
        for row in range(4):
            for col in range(4):
                if state[row][col] != ' ' and state[row][col] == state[row+1][col] == state[row][col+1] == state[row+1][col+1]:
                    return 1 if state[row][col] == self.my_piece else -1

        return 0  # no winner yet


    def heuristic_game_value(self, state):
        weight = [
            [0.05, 0.1, 0.05, 0.1, 0.05],
            [ 0.1, 0.2,  0.2, 0.2, 0.1],
            [0.05, 0.2,  0.3, 0.2, 0.05],
            [ 0.1, 0.2,  0.2, 0.2, 0.1],
            [0.05, 0.1, 0.05, 0.1, 0.05]
        ]
        val = self.game_value(state)
        
        if val != 0: # terminal states have non-zero game_value
            return val
        
        player_score = 0
        opp_score = 0
        for row in range(5):
            for col in range(5):
                if state[row][col] == self.my_piece:
                    player_score += weight[row][col]
                elif state[row][col] == self.opp:
                    opp_score += weight[row][col]

        return player_score - opp_score


    def max_value(self, state, depth, alpha, beta):
        if abs(self.game_value(state)) == 1:
            return self.game_value(state)

        if depth == self.depth_limit:
            return self.heuristic_game_value(state)

        successors = self.succ(state, self.my_piece) # get all the possible successors
        for succ in successors:
            state_copy = copy.deepcopy(state)
            if self.get_drop_phase(state):
                state_copy[succ[0]][succ[1]] = self.my_piece # place piece
            else:
                state_copy[succ[0]][succ[1]] = ' '
                state_copy[succ[2]][succ[3]] = self.my_piece # move exisiting piece
            
            alpha = max(alpha, self.min_value(state_copy, depth+1, alpha, beta)) # update the alpha

        if alpha >= beta:
            return beta

        return alpha


    def min_value(self, state, depth, alpha, beta):
        if abs(self.game_value(state)) == 1:
            return self.game_value(state)

        if depth == self.depth_limit:
            return self.heuristic_game_value(state)

        successors = self.succ(state, self.opp) # get all the possible successors
        for succ in successors:
            state_copy = copy.deepcopy(state)
            if self.get_drop_phase(state):
                state_copy[succ[0]][succ[1]] = self.opp # place piece
            else:
                state_copy[succ[0]][succ[1]] = ' '
                state_copy[succ[2]][succ[3]] = self.opp # move exisiting piece
            
            beta = min(beta, self.max_value(state_copy, depth+1, alpha, beta)) # update the alpha

        if beta <= alpha:
            return alpha

        return beta


############################################################################
#
# THE FOLLOWING CODE IS FOR SAMPLE GAMEPLAY ONLY
#
############################################################################
def main():
    print('Hello, this is Samaritan')
    ai = TeekoPlayer()
    piece_count = 0
    turn = 0

    # drop phase
    while piece_count < 8 and ai.game_value(ai.board) == 0:

        # get the player or AI's move
        if ai.my_piece == ai.pieces[turn]:
            ai.print_board()
            move = ai.make_move(ai.board)
            ai.place_piece(move, ai.my_piece)
            print(ai.my_piece + " moved at " + chr(move[0][1] + ord("A")) + str(move[0][0]))
        else:
            move_made = False
            ai.print_board()
            print(ai.opp + "'s turn")
            while not move_made:
                player_move = input("Move (e.g. B3): ")
                while player_move[0] not in "ABCDE" or player_move[1] not in "01234":
                    player_move = input("Move (e.g. B3): ")
                try:
                    ai.opponent_move([(int(player_move[1]), ord(player_move[0]) - ord("A"))])
                    move_made = True
                except Exception as e:
                    print(e)

        # update the game variables
        piece_count += 1
        turn += 1
        turn %= 2

    # move phase - can't have a winner until all 8 pieces are on the board
    while ai.game_value(ai.board) == 0:

        # get the player or AI's move
        if ai.my_piece == ai.pieces[turn]:
            ai.print_board()
            move = ai.make_move(ai.board)
            ai.place_piece(move, ai.my_piece)
            print(ai.my_piece + " moved from " + chr(move[1][1] + ord("A")) + str(move[1][0]))
            print("  to " + chr(move[0][1] + ord("A")) + str(move[0][0]))
        else:
            move_made = False
            ai.print_board()
            print(ai.opp + "'s turn")
            while not move_made:
                move_from = input("Move from (e.g. B3): ")
                while move_from[0] not in "ABCDE" or move_from[1] not in "01234":
                    move_from = input("Move from (e.g. B3): ")
                move_to = input("Move to (e.g. B3): ")
                while move_to[0] not in "ABCDE" or move_to[1] not in "01234":
                    move_to = input("Move to (e.g. B3): ")
                try:
                    ai.opponent_move([(int(move_to[1]), ord(move_to[0]) - ord("A")),
                                      (int(move_from[1]), ord(move_from[0]) - ord("A"))])
                    move_made = True
                except Exception as e:
                    print(e)

        # update the game variables
        turn += 1
        turn %= 2

    ai.print_board()
    if ai.game_value(ai.board) == 1:
        print("AI wins! Game over.")
    else:
        print("You win! Game over.")


if __name__ == "__main__":
    main()
