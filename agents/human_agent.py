import chess
from .agent import Agent


class HumanAgent(Agent):
    """Người chơi: nước đi đến từ click chuột trong GUI, không qua choose_move.
    Tạo human_agent chủ yếu để đại diện cho player để chọn trong phần option."""
    name = "Human"
    interactive = True

    def choose_move(self, board: chess.Board, time_limit=None) -> chess.Move:
        raise NotImplementedError("Nước đi của người chơi đến từ GUI")
