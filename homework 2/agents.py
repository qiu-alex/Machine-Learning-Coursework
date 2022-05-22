import random
import math


BOT_NAME = "Bot Marius"


class RandomAgent:
    """Agent that picks a random available move.  You should be able to beat it."""
    def __init__(self, sd=None):
        if sd is None:
            self.st = None
        else:
            random.seed(sd)
            self.st = random.getstate()

    def get_move(self, state):
        if self.st is not None:
            random.setstate(self.st)
        return random.choice(state.successors())


class HumanAgent:
    """Prompts user to supply a valid move."""
    def get_move(self, state, depth=None):
        move__state = dict(state.successors())
        prompt = "Kindly enter your move {}: ".format(sorted(move__state.keys()))
        move = None
        while move not in move__state:
            try:
                move = int(input(prompt))
            except ValueError:
                continue
        return move, move__state[move]


class MinimaxAgent:
    """Artificially intelligent agent that uses minimax to optimally select the best move."""

    def get_move(self, state):
        """Select the best available move, based on minimax value."""
        nextp = state.next_player()
        best_util = -math.inf if nextp == 1 else math.inf
        best_move = None
        best_state = None

        for move, state in state.successors():
            util = self.minimax(state)
            if ((nextp == 1) and (util > best_util)) or ((nextp == -1) and (util < best_util)):
                best_util, best_move, best_state = util, move, state
        return best_move, best_state

    def minimax(self, state):
        """Determine the minimax utility value of the given state.

        Args:
            state: a connect383.GameState object representing the current board

        Returns: the exact minimax utility value of the state
        """
        # if terminal node reached return utility value
        if len(state.successors()) == 0:
            return state.utility()
        if (state.next_player() == 1):
            value = -math.inf
            for move, succs in state.successors():
                value = max(value, self.minimax(succs))
            return value
        else:
            value = math.inf
            for move, succs in state.successors():
                value = min(value, self.minimax(succs))
            return value


class MinimaxHeuristicAgent(MinimaxAgent):
    """Artificially intelligent agent that uses depth-limited minimax to select the best move.
    Hint: Consider what you did for MinimaxAgent. What do you need to change to get what you want? 
    """

    def __init__(self, depth_limit):
        self.depth_limit = depth_limit


    def minimax(self, state):
        """Determine the heuristically estimated minimax utility value of the given state.

        The depth data member (set in the constructor) determines the maximum depth of the game 
        tree that gets explored before estimating the state utilities using the evaluation() 
        function.  If depth is 0, no traversal is performed, and minimax returns the results of 
        a call to evaluation().  If depth is None, the entire game tree is traversed.

        Args:
            state: a connect383.GameState object representing the current board

        Returns: the minimax utility value of the state
        """
        #if there is no depth limit run regular minimax
        if (self.depth_limit == 0):
            player = MinimaxAgent()
            return player.minimax(state)
        else:
            return self.minimax_depth(state, self.depth_limit)

    def minimax_depth(self, state, depth):
        """This is just a helper method for minimax(). """
        #if self.depth_limit == 0 or terminal node reached return value of node
        if (depth == 0 or len(state.successors()) == 0):
            return self.evaluation(state)
        if (state.next_player() == 1):
            value = -math.inf
            for move, succs in state.successors():
                value = max(value, self.minimax_depth(succs, depth - 1))
            return value
        else:
            value = math.inf
            for move, succs in state.successors():
                value = min(value, self.minimax_depth(succs, depth - 1))
            return value


    def evaluation(self, state):
        """Estimate the utility value of the game state based on features.

        N.B.: This method must run in O(1) time!

        Args:
            state: a connect383.GameState object representing the current board

        Returns: a heuristic estimate of the utility value of the state
        """
        p1_score = 0
        p2_score = 0
        for run in state.get_rows() + state.get_cols() + state.get_diags():
            for player, length in all_streaks(run):
                if (player == 1) and (length >= 3):
                    p1_score += length**2
                elif (player == -1) and (length >= 3):
                    p2_score += length**2
        return p1_score - p2_score

#find lengths of all streaks: present and possible 
def all_streaks(lst):  
    """Get the lengths of all the streaks of the same element in a sequence."""
    rets = []  # list of (element, length) tuples
    prev = lst[0]
    curr_len = 1
    for curr in lst[1:]:
        if curr == prev:
            curr_len += 1
        else:
            rets.append((prev, curr_len))
            #if space is after streak
            if (curr == 0 and prev != 0):
                rets.append((prev, curr_len + 1))
            #if space is before streak
            if (prev == 0 and curr != 0):
                curr_len = 1
                for elem in lst[curr:]:
                    if (elem == curr):
                        curr_len += 1
                    else:
                        rets.append((prev, curr_len + 1))
            #rets.append((prev, curr_len))
            prev = curr
            curr_len = 1
    rets.append((prev, curr_len))
    return rets

class MinimaxPruneAgent(MinimaxAgent):
    """Smarter computer agent that uses minimax with alpha-beta pruning to select the best move.
    Hint: Consider what you did for MinimaxAgent. What do you need to change to get what you want? 
    """

    def minimax(self, state):
        """Determine the minimax utility value the given state using alpha-beta pruning.

        The value should be equal to the one determined by MinimaxAgent.minimax(), but the 
        algorithm should do less work.  You can check this by inspecting the value of the class 
        variable GameState.state_count, which keeps track of how many GameState objects have been 
        created over time.  This agent should also respect the depth limit like HeuristicAgent.

        N.B.: When exploring the game tree and expanding nodes, you must consider the child nodes
        in the order that they are returned by GameState.successors(). That is, you cannot prune
        the state reached by moving to column 4 before you've explored the state reached by a move
        to to column 1.

        Args: 
            state: a connect383.GameState object representing the current board

        Returns: the minimax utility value of the state
        """
        return self.alphabeta(state, -math.inf, math.inf)

    def alphabeta(self, state, alpha, beta):
        """ This is just a helper method for minimax(). Feel free to use it or not. """
        if len(state.successors()) == 0:
            return state.utility()
        if (state.next_player() == 1):
            value = -math.inf
            for move, succs in state.successors():
                value = max(value, self.alphabeta(succs, alpha, beta))
                if (value >= beta):
                    break
                alpha = max(value, alpha) 
            return value
        else:
            value = math.inf
            for move, succs in state.successors():
                value = min(value, self.alphabeta(succs, alpha, beta))
                if (value <= alpha):
                    break
                beta = min(value, beta) 
            return value


class OtherMinimaxHeuristicAgent(MinimaxAgent):
    """Alternative heursitic agent used for testing."""

    def __init__(self, depth_limit):
        self.depth_limit = depth_limit

    def minimax(self, state):
        """Determine the heuristically estimated minimax utility value of the given state."""
        #
        # Fill this in, if it pleases you.
        #
        return 26  # Change this line, unless you have something better to do.

