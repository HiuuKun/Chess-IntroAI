from abc import ABC, abstractmethod
import chess

class Agent(ABC):
    name = "Agent"
    interactive = False   

    @abstractmethod
    def choose_move(self, board: chess.Board, time_limit=None) -> chess.Move:
        """Implement hàm này, trả về một move hợp lệ từ list(board.legal_moves)."""

    def on_game_start(self, color): pass
    def on_game_end(self, result): pass
    def close(self): pass