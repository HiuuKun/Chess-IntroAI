import glob
import os
import shutil
import chess
import chess.engine
from .agent import Agent


# thư mục gốc project = cha của thư mục agents/
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _looks_like_engine(path: str) -> bool:
    if not os.path.isfile(path):
        return False
    ext = os.path.splitext(path)[1].lower()
    if os.name == "nt":
        return ext == ".exe"
    return ext == "" and os.access(path, os.X_OK)


class StockfishAgent(Agent):
    interactive = False

    def __init__(self, level: int = 5, path: str | None = None,
                 move_time: float = 0.1, depth: int | None = None,
                 threads: int = 1, hash_mb: int = 16):
        if not 0 <= level <= 20:
            raise ValueError("level phải nằm trong khoảng 0..20")
        self.level = level
        self.path = path
        self.move_time = move_time
        self.depth = depth
        self.threads = threads
        self.hash_mb = hash_mb
        self.name = f"Stockfish Lv{level}"
        self._engine: chess.engine.SimpleEngine | None = None

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
            engine.configure({
                "Skill Level": self.level,
                "Threads": self.threads,
                "Hash": self.hash_mb,
            })
            self._engine = engine
        return self._engine

    def _limit(self, time_limit) -> chess.engine.Limit:
        if time_limit is not None:
            return chess.engine.Limit(time=time_limit)
        if self.depth is not None:
            return chess.engine.Limit(depth=self.depth)
        return chess.engine.Limit(time=self.move_time)

    # ---------- Agent API ----------
    def choose_move(self, board: chess.Board, time_limit=None) -> chess.Move:
        limit = self._limit(time_limit)
        try:
            result = self._ensure_engine().play(board, limit)
        except chess.engine.EngineTerminatedError:
            self._engine = None
            result = self._ensure_engine().play(board, limit)
        return result.move

    def close(self):
        if self._engine is not None:
            try:
                self._engine.quit()
            except Exception:
                pass
            self._engine = None