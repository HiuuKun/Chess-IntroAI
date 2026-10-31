import glob
import os
import shutil
import chess
import chess.engine
from .agent import Agent


# thư mục gốc project = cha của thư mục agents/
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MATE_SCORE = 100_000  # quy đổi "chiếu hết" sang centipawn để so sánh các nước


def _looks_like_engine(path: str) -> bool:
    """File chạy được của Stockfish: .exe trên Windows, hoặc file không đuôi trên Linux/Mac."""
    if not os.path.isfile(path):
        return False
    ext = os.path.splitext(path)[1].lower()
    if os.name == "nt":
        return ext == ".exe"
    return ext == "" and os.access(path, os.X_OK)


def _fmt_score(score: chess.engine.PovScore, color: chess.Color) -> str:
    """Điểm theo góc nhìn của bên `color`: +0.35 (35 cp) hoặc 'Chiếu hết sau N nước'."""
    s = score.pov(color)
    if s.is_mate():
        m = s.mate()
        return f"Chiếu hết sau {abs(m)} nước ({'Stockfish thắng' if m > 0 else 'Stockfish thua'})"
    cp = s.score()
    return f"{cp / 100:+.2f} ({cp:+d} cp)"


def _score_value(info: dict, color: chess.Color) -> int:
    return info["score"].pov(color).score(mate_score=MATE_SCORE)


_METHOD_NOTE = """
================ CÁCH STOCKFISH TÍNH ĐIỂM VÀ CHỌN NƯỚC ================
Engine: {engine_name}
1) Hàm đánh giá: mạng nơ-ron NNUE chấm điểm từng thế cờ (vị trí quân, an toàn của vua,
   tốt thông, ...). Đơn vị centipawn (100 cp ~ hơn 1 con tốt); dương = bên đang đi có lợi.
   Nếu thấy chiếu hết cưỡng bức thì báo "Chiếu hết sau N nước" thay vì điểm số.
2) Tìm kiếm: alpha-beta (PVS) + iterative deepening: tìm độ sâu 1, 2, 3, ... và dùng kết quả
   tầng trước để sắp xếp nước cho tầng sau. Có cắt tỉa (null-move, late-move reductions, ...),
   bảng chuyển vị, và quiescence search ở lá (đi hết chuỗi ăn quân/chiếu rồi mới chấm điểm).
3) Điểm của một nước = điểm của thế cờ ở cuối PV (principal variation: dãy nước mà hai bên
   đều chơi tối ưu). Nước tốt nhất = nước có điểm cao nhất.
4) Skill Level = {level}/20: {level_note}
(Chi tiết con số có thể khác đôi chút giữa các phiên bản Stockfish.)
=======================================================================
"""


class StockfishAgentReport(Agent):
    """Agent bọc Stockfish qua giao thức UCI (python-chess).

    level     : 0..20, map vào option "Skill Level" của Stockfish (20 = mạnh nhất).
    move_time : giây suy nghĩ mỗi nước (mặc định 0.1s). Nếu truyền time_limit vào
                choose_move thì time_limit được ưu tiên.
    depth     : nếu set (và không có time_limit) thì giới hạn theo độ sâu thay vì thời gian.
    path      : đường dẫn tới stockfish. Nếu None sẽ tìm theo thứ tự:
                STOCKFISH_PATH -> lệnh `stockfish` trong PATH -> thư mục gốc project.
    verbose   : True -> mỗi nước in ra terminal: legal moves, các nước ứng viên + điểm,
                lý do chọn nước. Đặt False khi chạy arena hàng loạt.
    top_n     : số nước ứng viên (MultiPV) hiển thị khi verbose. Lưu ý: khi level < 20
                Stockfish tự tính ít nhất 4 ứng viên để chọn nước "yếu hơn".

    Engine được khởi động lười (lần choose_move đầu tiên) nên việc tạo agent
    trong AGENT_FACTORIES / GUI không sinh process Stockfish thừa.
    """
    interactive = False

    def __init__(self, level: int = 5, path: str | None = None,
                 move_time: float = 0.1, depth: int | None = None,
                 threads: int = 1, hash_mb: int = 16,
                 verbose: bool = True, top_n: int = 5):
        if not 0 <= level <= 20:
            raise ValueError("level phải nằm trong khoảng 0..20")
        self.level = level
        self.path = path
        self.move_time = move_time
        self.depth = depth
        self.threads = threads
        self.hash_mb = hash_mb
        self.verbose = verbose
        self.top_n = max(1, top_n)
        self.name = f"Stockfish Lv{level}"
        self._engine: chess.engine.SimpleEngine | None = None
        self._explained = False          # chỉ in phần giải thích phương pháp 1 lần
        self.last_info: dict = {}        # thống kê nước đi gần nhất (dùng cho arena)

    # ---------- quản lý engine ----------
    def _find_binary(self) -> str:
        # 1) path truyền vào / biến môi trường / PATH
        for c in (self.path, os.environ.get("STOCKFISH_PATH"), shutil.which("stockfish")):
            if c and os.path.isfile(c):
                return c
        # Không tìm thấy PATH hợp lệ thì gọi đệ quy folder project để tìm file engine
        roots = [PROJECT_ROOT, os.getcwd()]
        for root in dict.fromkeys(roots):
            for pattern in ("stockfish*", os.path.join("stockfish*", "**", "stockfish*")):
                for f in sorted(glob.glob(os.path.join(root, pattern), recursive=True)):
                    if _looks_like_engine(f):
                        return f

        raise FileNotFoundError(
            "Không tìm thấy Stockfish. Đã thử: path=..., biến STOCKFISH_PATH, PATH, "
            f"và thư mục {PROJECT_ROOT}. Hãy truyền path=... trỏ thẳng tới file chạy."
        )

    def _ensure_engine(self) -> chess.engine.SimpleEngine:
        if self._engine is None:
            engine = chess.engine.SimpleEngine.popen_uci(self._find_binary())
            options = {
                "Skill Level": self.level,
                "Threads": self.threads,
                "Hash": self.hash_mb,
            }
            # Bật thống kê thắng/hòa/thua (WDL) nếu bản Stockfish này hỗ trợ
            if "UCI_ShowWDL" in engine.options:
                options["UCI_ShowWDL"] = True
            engine.configure(options)
            self._engine = engine
        return self._engine

    def _limit(self, time_limit) -> chess.engine.Limit:
        if time_limit is not None:
            return chess.engine.Limit(time=time_limit)
        if self.depth is not None:
            return chess.engine.Limit(depth=self.depth)
        return chess.engine.Limit(time=self.move_time)

    # ---------- tìm kiếm ----------
    def _search(self, board: chess.Board, limit: chess.engine.Limit, n: int):
        """Trả về (nước Stockfish thực sự chọn, danh sách n dòng phân tích tốt nhất)."""
        engine = self._ensure_engine()
        with engine.analysis(board, limit, multipv=n, info=chess.engine.INFO_ALL) as analysis:
            best = analysis.wait()       # chờ engine trả `bestmove`
            lines = [dict(l) for l in analysis.multipv if "pv" in l and "score" in l]
        return best.move, lines

    # ---------- Agent API ----------
    def choose_move(self, board: chess.Board, time_limit=None) -> chess.Move:
        limit = self._limit(time_limit)
        n = max(1, min(self.top_n, board.legal_moves.count()))
        try:
            move, lines = self._search(board, limit, n)
        except chess.engine.EngineTerminatedError:
            self._engine = None
            move, lines = self._search(board, limit, n)

        if self.verbose:
            self._report(board, limit, move, lines)
        self._record(board, move, lines)
        return move

    def close(self):
        if self._engine is not None:
            try:
                self._engine.quit()
            except Exception:
                pass
            self._engine = None

    # ---------- thống kê & in ra terminal ----------
    def _record(self, board, move, lines):
        color = board.turn
        rank = next((i + 1 for i, l in enumerate(lines) if l["pv"][0] == move), None)
        top = lines[0] if lines else {}
        self.last_info = {
            "depth": top.get("depth"),
            "nodes": top.get("nodes"),
            "score_cp": _score_value(top, color) if top else None,
            "rank_chosen": rank,
        }

    def _explain_method_once(self):
        if self._explained:
            return
        self._explained = True
        if self.level >= 20:
            level_note = "đã tắt cơ chế làm yếu, luôn chọn nước có điểm cao nhất."
        else:
            level_note = (
                "đã BẬT cơ chế làm yếu. Stockfish ép tính ít nhất 4 nước ứng viên (MultiPV >= 4); "
                f"khi tìm tới độ sâu ~{self.level + 1} (= level + 1) thì chọn ngẫu nhiên có trọng số "
                "trong các ứng viên đó. Level càng thấp thì nhiễu ngẫu nhiên càng lớn, nước kém "
                "càng dễ được chọn; cho nghĩ lâu hơn gần như không làm nó mạnh hơn."
            )
        try:
            engine_name = self._ensure_engine().id.get("name", "Stockfish")
        except Exception:
            engine_name = "Stockfish"
        print(_METHOD_NOTE.format(engine_name=engine_name, level=self.level, level_note=level_note))

    def _describe_line(self, board: chess.Board, idx: int, info: dict) -> str:
        color = board.turn
        pv = info["pv"]
        try:
            pv_san = board.variation_san(pv[:6])
        except Exception:
            pv_san = " ".join(m.uci() for m in pv[:6])
        parts = [f"  {idx}. {board.san(pv[0]):<8} {_fmt_score(info['score'], color):<28}"]
        if "depth" in info:
            parts.append(f"depth {info['depth']}")
        if "nodes" in info:
            parts.append(f"nodes {info['nodes']:,}")
        wdl = info.get("wdl")
        if wdl is not None:
            w = wdl.pov(color)
            parts.append(f"W/D/L {w.wins / 10:.0f}%/{w.draws / 10:.0f}%/{w.losses / 10:.0f}%")
        return " | ".join(parts) + f"\n       PV: {pv_san}"

    def _report(self, board: chess.Board, limit: chess.engine.Limit, move: chess.Move, lines: list[dict]):
        self._explain_method_once()
        color = board.turn
        side = "Trắng" if color == chess.WHITE else "Đen"

        # --- (1) legal moves ---
        legal = sorted(board.san(m) for m in board.legal_moves)
        print(f"\n===== {self.name} ({side}) - nước thứ {board.fullmove_number} =====")
        print(f"Legal moves ({len(legal)}): {', '.join(legal)}")

        # --- (2) điểm các nước ứng viên ---
        if limit.time is not None:
            lim = f"{limit.time}s"
        elif limit.depth is not None:
            lim = f"depth {limit.depth}"
        else:
            lim = "không giới hạn"
        print(f"Giới hạn tìm kiếm: {lim}")
        print(f"Top {len(lines)} nước ứng viên (điểm theo góc nhìn của Stockfish):")
        for i, info in enumerate(lines, 1):
            print(self._describe_line(board, i, info))

        # --- (3) vì sao chọn nước này ---
        chosen = board.san(move)
        rank = next((i for i, l in enumerate(lines, 1) if l["pv"][0] == move), None)
        if not lines:
            print(f"-> Chọn: {chosen}")
        elif rank == 1:
            second = ""
            if len(lines) > 1:
                gap = _score_value(lines[0], color) - _score_value(lines[1], color)
                if abs(gap) < MATE_SCORE // 2:
                    second = f", hơn nước hạng 2 ({board.san(lines[1]['pv'][0])}) khoảng {gap} cp"
            print(f"-> Chọn: {chosen} - nước có điểm cao nhất sau khi tìm tới depth "
                  f"{lines[0].get('depth', '?')}{second}.")
        elif rank is not None:
            loss = _score_value(lines[0], color) - _score_value(lines[rank - 1], color)
            reason = (f"do Skill Level {self.level} < 20 nên Stockfish cố ý chọn nước không tối ưu"
                      if self.level < 20 else "do tìm kiếm kết thúc khi các nước còn sát điểm nhau")
            print(f"-> Chọn: {chosen} - hạng {rank}/{len(lines)}, kém nước tốt nhất "
                  f"({board.san(lines[0]['pv'][0])}) khoảng {loss} cp, {reason}.")
        else:
            print(f"-> Chọn: {chosen} - nằm NGOÀI top {len(lines)} đang hiển thị "
                  f"(Skill Level {self.level} chọn trong ít nhất 4 ứng viên; tăng top_n để thấy).")
