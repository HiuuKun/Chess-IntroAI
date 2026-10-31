import chess
from agents.agent import Agent

class Game:
    def __init__(self, white: Agent, black: Agent):
        self.board = chess.Board()
        self.agents = {chess.WHITE: white, chess.BLACK: black}
        self.move_history: list[chess.Move] = []
        self._ended = False          
        for color, agent in self.agents.items():
            agent.on_game_start(color)

    @property
    def turn(self) -> chess.Color:
        return self.board.turn

    @property
    def current_agent(self) -> Agent:
        return self.agents[self.board.turn]

    # ---------- kết thúc ván & điều kiện hòa ----------
    def outcome(self) -> chess.Outcome | None:
        """game.board.outcome() xuất ra thành game.outcome()"""
        return self.board.outcome(claim_draw=True)

    def is_over(self) -> bool:
        "Từ game.outcome() xuất ra giá trị boolean. Check thử ván cờ đã kết thúc chưa?"
        return self.outcome() is not None

    def result(self) -> str | None:
        """Từ game.outcome() xuất ra string để check winner: '1-0', '0-1', '1/2-1/2' hoặc None nếu chưa kết thúc."""
        out = self.outcome()
        return out.result() if out else None

    def end_reason(self) -> str | None:
        """Lý do kết thúc, dùng để hiển thị trên GUI."""
        out = self.outcome()
        if out is None:
            return None
        names = {
            chess.Termination.CHECKMATE: "Chiếu hết",
            chess.Termination.STALEMATE: "Hòa do hết nước đi",
            chess.Termination.INSUFFICIENT_MATERIAL: "Hòa do thiếu quân",
            chess.Termination.THREEFOLD_REPETITION: "Hòa do lặp thế 3 lần",
            chess.Termination.FIVEFOLD_REPETITION: "Hòa do lặp thế 5 lần",
            chess.Termination.FIFTY_MOVES: "Hòa theo luật 50 nước",
            chess.Termination.SEVENTYFIVE_MOVES: "Hòa theo luật 75 nước",
        }
        return names.get(out.termination, out.termination.name)

    # ---------- thực hiện nước đi ----------
    def legal_moves_from(self, square: int) -> list[chess.Move]:
        "Danh sách các nước đi hợp lệ tại một vị trí nào đó. Lib tự check xem vị trí đó đang có quân nào."
        return [m for m in self.board.legal_moves if m.from_square == square]

    def build_move(self, from_sq: int, to_sq: int, promotion=chess.QUEEN) -> chess.Move:
        """Tạo Move hợp lệ từ 2 ô được chọn (from_sq -> to_sq); 
        tự gắn quân phong nếu là tốt tới hàng cuối.
        - Dành cho HumanAgent only. 
        - AI Agents sẽ tự chọn và trả về move từ legal_moves_from(sqle)"""""
        piece = self.board.piece_at(from_sq)
        is_promo = (
            piece is not None
            and piece.piece_type == chess.PAWN
            and chess.square_rank(to_sq) in (0, 7)
        )
        return chess.Move(from_sq, to_sq, promotion=promotion if is_promo else None)

    def make_move(self, move: chess.Move) -> bool:
        "Thực hiện move đó. Lưu lại move vào lịch sử."
        if move not in self.board.legal_moves:
            return False
        self.board.push(move)
        self.move_history.append(move)
        if self.is_over() and not self._ended:
            self._ended = True
            for agent in self.agents.values():
                agent.on_game_end(self.result())
        return True

    def step(self) -> chess.Move | None:
        """Vì các agent là non-interactive nên hàm này giúp các agent thực hiện move bằng cách gọi agent.choose_move().
        Đưa cho agent một bảng copy của bàn cờ để tránh làm hỏng bàn cờ gốc.
        GUI có sẵn update nên hàm này dùng cho env."""
        if self.is_over() or self.current_agent.interactive:
            return None
        move = self.current_agent.choose_move(self.board.copy())
        if not self.make_move(move):
            raise ValueError(f"{self.current_agent.name} trả về nước không hợp lệ: {move}")
        return move

    # ---------- undo ----------
    def undo(self, plies: int = 1) -> int:
        """Undo 1 nước cờ. 
        Nếu một bên vừa đi -> undo(1). 
        Nếu cả hai bên vừa đi và bạn muốn undo hoàn toàn nước của cả 2 -> undo(2)"""
        undone = 0
        while undone < plies and self.move_history:
            self.board.pop()
            self.move_history.pop()
            undone += 1
        if undone:
            self._ended = False      
        return undone

    def close(self):
        for agent in self.agents.values():
            agent.close()

    def reset(self):
        self.board.reset()
        self.move_history.clear()
        self._ended = False
        for color, agent in self.agents.items():
            agent.on_game_start(color)