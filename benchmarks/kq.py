"""Validated positions and replay auditing for king-and-queen versus king."""

import hashlib
import json
from pathlib import Path

import chess

DEFAULT_DATASET = Path(__file__).parent / "data" / "king_queen_vs_king.json"


def board_for(position, color=chess.WHITE):
    board = chess.Board(position["fen"])
    return board if color == chess.WHITE else board.mirror()


def load_dataset(path=DEFAULT_DATASET):
    path = Path(path)
    raw = path.read_bytes()
    payload = json.loads(raw.decode("utf-8-sig"))
    if (not isinstance(payload, dict) or payload.get("schema_version") != 1
            or not isinstance(payload.get("positions"), list) or not payload["positions"]):
        raise ValueError("Expected schema_version 1 and nonempty positions")
    ids = set()
    for position in payload["positions"]:
        if not isinstance(position, dict):
            raise ValueError("Each position must be an object")
        name = position.get("id")
        if not isinstance(name, str) or not name or name in ids:
            raise ValueError("Position IDs must be nonempty and unique")
        ids.add(name)
        board = board_for(position)
        expected = {chess.Piece(chess.KING, chess.WHITE): 1,
                    chess.Piece(chess.QUEEN, chess.WHITE): 1,
                    chess.Piece(chess.KING, chess.BLACK): 1}
        pieces = list(board.piece_map().values())
        if (not board.is_valid() or len(pieces) != 3 or board.turn != chess.WHITE
                or any(pieces.count(piece) != count for piece, count in expected.items())
                or board.outcome(claim_draw=True) is not None):
            raise ValueError(f"{name}: expected legal nonterminal White KQ vs Black K, White to move")
        if position.get("category") not in ("general", "control"):
            raise ValueError(f"{name}: category must be general or control")
    return {**payload, "metadata": {"source": str(path.resolve()),
            "sha256": hashlib.sha256(raw).hexdigest(), "position_count": len(ids)}}


def audit_record(record):
    """Independently replay recorded moves and check terminal adjudication."""
    board = chess.Board(record["initial_fen"])
    color = board.turn
    for index, entry in enumerate(record["moves"], 1):
        if board.outcome(claim_draw=True) is not None:
            raise ValueError("Recorded moves continue after game termination")
        move = chess.Move.from_uci(entry["uci"])
        expected_actor = record["agent"] if board.turn == color else "Stockfish"
        if (entry["ply"] != index or entry["actor"] != expected_actor
                or entry["fen_before"] != board.fen() or move not in board.legal_moves
                or entry["san"] != board.san(move)):
            raise ValueError(f"Invalid move record at ply {index}")
        board.push(move)
        if entry["fen_after"] != board.fen():
            raise ValueError(f"FEN mismatch at ply {index}")
    if record["plies"] != len(record["moves"]) or board.fen() != record["final_fen"]:
        raise ValueError("Final position mismatch")
    outcome = board.outcome(claim_draw=True)
    if outcome is not None:
        result = "draw" if outcome.winner is None else "win" if outcome.winner == color else "loss"
        if (record["result"] != outcome.result() or record["agent_result"] != result
                or record["termination"] != outcome.termination.name):
            raise ValueError("Terminal result mismatch")
    elif record["result"] != "*" or record["agent_result"] != "unresolved":
        raise ValueError("Unfinished game counted as a result")
    return {"passed": True, "legal_plies_replayed": len(record["moves"])}
