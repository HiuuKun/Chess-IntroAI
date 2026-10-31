# agents/random_agent.py
import random
import chess
from .agent import Agent

class RandomAgent(Agent):
    name = "Random"

    def choose_move(self, board: chess.Board, time_limit=None) -> chess.Move:
        return random.choice(list(board.legal_moves))

