import itertools
import random

import chess

from .agent import Agent

PROMO_NAMES = {
    chess.QUEEN: "Hậu",
    chess.KNIGHT: "Mã",
    chess.ROOK: "Xe",
    chess.BISHOP: "Tượng",
}


class PromotionTestAgent(Agent):
    """Agent dùng để test phong tốt.

    - Nếu có nước phong: phong lần lượt Hậu -> Mã -> Xe -> Tượng (xoay vòng),
      để mỗi loại phong đều được test.
    - Nếu không: ưu tiên đẩy tốt đang tiến xa nhất về phía hàng phong.
    - Không có tốt để đẩy: đi ngẫu nhiên.
    Mọi nước đều lấy từ board.legal_moves nên luôn hợp lệ.
    """

    name = "Promotion Test"

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self._cycle = itertools.cycle(
            [chess.QUEEN, chess.KNIGHT, chess.ROOK, chess.BISHOP]
        )
        self.promo_log: list[chess.Move] = []

    def on_game_start(self, color):
        self.promo_log.clear()

    def choose_move(self, board: chess.Board, time_limit=None) -> chess.Move:
        legal = list(board.legal_moves)

        # 1. Có nước phong -> chọn loại quân tiếp theo trong vòng xoay
        promos = [m for m in legal if m.promotion]
        if promos:
            want = next(self._cycle)
            move = next((m for m in promos if m.promotion == want), promos[0])
            self.promo_log.append(move)
            if self.verbose:
                print(f"[{self.name}] phong {PROMO_NAMES[move.promotion]}: {move.uci()}")
            return move

        # 2. Đẩy tốt đi xa nhất (ưu tiên nước đi thẳng, rồi nước ăn chéo)
        pawn_moves = [
            m for m in legal
            if board.piece_type_at(m.from_square) == chess.PAWN
        ]
        if pawn_moves:
            def progress(m):
                rank = chess.square_rank(m.to_square)
                return rank if board.turn == chess.WHITE else 7 - rank

            best = max(progress(m) for m in pawn_moves)
            return random.choice([m for m in pawn_moves if progress(m) == best])

        # 3. Fallback
        return random.choice(legal)
