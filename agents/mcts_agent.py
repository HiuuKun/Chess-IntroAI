import math
import random
import time
import chess
from .agent import Agent


class Node:
    def __init__(self, board, parent=None, move=None):
        self.board = board
        self.parent = parent
        self.move = move
        self.children = []
        self.untried = list(board.legal_moves)
        self.visits = 0
        self.value = 0.0


class MCTSAgent(Agent):
    name = "MCTS"

    def __init__(self, simulations=200, rollout_depth=30, exploration=1.4):
        self.simulations = simulations
        self.rollout_depth = rollout_depth
        self.exploration = exploration

    def evaluate(self, board, color):
        outcome = board.outcome(claim_draw=True)
        if outcome is not None:
            if outcome.winner is None:              # Hòa = 0.5
                return 0.5                          
            return 1.0 if outcome.winner == color else 0.0      # Thắng = 1 or Thua = 0

        points = {chess.PAWN: 1,        # Tốt
                  chess.KNIGHT: 3,      # Mã
                  chess.BISHOP: 3,      # Tượng
                  chess.ROOK: 5,        # Xe
                  chess.QUEEN: 9}       # Hậu
        score = sum(points[p] * (len(board.pieces(p, color)) -
                                 len(board.pieces(p, not color))) for p in points)
        return max(0.0, min(1.0, 0.5 + score / 30))

    def choose_move(self, board: chess.Board, time_limit=None) -> chess.Move:
        moves = list(board.legal_moves)
        if not moves:
            return None
        if len(moves) == 1:
            return moves[0]

        color = board.turn
        root = Node(board.copy())
        start = time.perf_counter()
        count = 0

        while count == 0 or (count < self.simulations if time_limit is None
                             else time.perf_counter() - start < time_limit):
            node = root

            while not node.untried and node.children:
                parent = node
                node = max(parent.children, key=lambda child:
                    (child.value / child.visits if parent.board.turn == color
                     else 1 - child.value / child.visits)
                    + self.exploration * math.sqrt(math.log(parent.visits) / child.visits))

            if node.untried and not node.board.is_game_over(claim_draw=True):
                move = node.untried.pop(random.randrange(len(node.untried)))
                next_board = node.board.copy()
                next_board.push(move)
                child = Node(next_board, node, move)
                node.children.append(child)
                node = child

            simulation = node.board.copy()
            for _ in range(self.rollout_depth):
                if simulation.is_game_over(claim_draw=True):
                    break
                simulation.push(random.choice(list(simulation.legal_moves)))

            result = self.evaluate(simulation, color)
            while node is not None:
                node.visits += 1
                node.value += result
                node = node.parent
            count += 1

        return max(root.children, key=lambda child: child.visits).move
