"""
Arena: môi trường headless để test các agent.

Chạy từ thư mục gốc (cùng cấp với agents/ và game.py):

    # 1 ván, in legal moves + nước agent chọn ở mỗi lượt
    python -m env --white Random --black Stockfish

    # nhiều ván, in thống kê hiệu năng 2 phe
    python -m env --white Random --black Stockfish -n 200

    # nhiều ván nhưng vẫn muốn xem chi tiết từng nước
    python -m env --white Random --black Random -n 3 --verbose
    (Không khuyến khích vì số lượng lớn sẽ rất lâu) 
"""
import argparse
import math
import time
from collections import Counter
from dataclasses import dataclass, field

import chess

from core.game import Game
from agents import AGENT_FACTORIES


# ====================== kết quả 1 ván ======================
@dataclass
class GameRecord:
    result: str                      # '1-0', '0-1', '1/2-1/2'
    reason: str
    plies: int
    white_label: str
    black_label: str
    move_times: dict = field(default_factory=lambda: {chess.WHITE: [], chess.BLACK: []})
    moves_san: list = field(default_factory=list)


# ====================== chạy 1 ván ======================
def play_game(white, black, verbose=False, max_plies=1000,
              white_label=None, black_label=None) -> GameRecord:
    """Chạy 1 ván headless giữa 2 agent đã khởi tạo sẵn."""
    for ag in (white, black):
        if getattr(ag, "interactive", False):
            raise ValueError(f"{ag.name} là agent tương tác (cần GUI), không chạy headless được")

    white_label = white_label or white.name
    black_label = black_label or black.name
    game = Game(white, black)
    rec = GameRecord("1/2-1/2", "", 0, white_label, black_label)
    labels = {chess.WHITE: white_label, chess.BLACK: black_label}
    forfeit = None

    if verbose:
        print(f"\n===== {white_label} (Trắng) vs {black_label} (Đen) =====")

    while not game.is_over() and len(game.move_history) < max_plies:
        color = game.turn
        before = game.board.copy()
        legal_san = sorted(before.san(m) for m in before.legal_moves)

        t0 = time.perf_counter()
        try:
            move = game.step()
        except ValueError as e:          # agent trả về nước không hợp lệ -> xử thua
            print(f"[!] {e}")
            forfeit = color
            break
        dt = time.perf_counter() - t0
        rec.move_times[color].append(dt)

        san = before.san(move)
        rec.moves_san.append(san)

        if verbose:
            side = "Trắng" if color == chess.WHITE else "Đen"
            print(f"\nNước {before.fullmove_number} - {side} ({labels[color]})")
            print(f"  Legal moves ({len(legal_san)}): {', '.join(legal_san)}")
            print(f"  Agent chọn : {san} ({move.uci()})   [{dt * 1000:.1f} ms]")

    rec.plies = len(game.move_history)

    if forfeit is not None:
        rec.result = "0-1" if forfeit == chess.WHITE else "1-0"
        rec.reason = "Đi nước không hợp lệ"
        # on_game_end chưa được gọi vì ván chưa kết thúc theo luật
        for ag in game.agents.values():
            ag.on_game_end(rec.result)
    elif game.is_over():
        rec.result = game.result()
        rec.reason = game.end_reason()
    else:
        rec.result = "1/2-1/2"
        rec.reason = f"Vượt quá {max_plies} nước (tính hòa)"
        for ag in game.agents.values():
            ag.on_game_end(rec.result)

    if verbose:
        print(f"\n>>> Kết quả: {rec.result} - {rec.reason} ({rec.plies} ply)")
        print(f">>> Nước đi: {' '.join(rec.moves_san)}")
    return rec


# ====================== chạy nhiều ván ======================
@dataclass
class Tally:
    wins: int = 0
    draws: int = 0
    losses: int = 0

    @property
    def n(self): return self.wins + self.draws + self.losses

    @property
    def score(self): return (self.wins + 0.5 * self.draws) / self.n if self.n else 0.0


def play_match(agent_a, agent_b, n_games=100, swap_colors=True,
               max_plies=1000, verbose=False, label_a=None, label_b=None):
    """
    Đấu n_games ván giữa A và B. Nếu swap_colors, A cầm trắng ở ván chẵn,
    cầm đen ở ván lẻ để loại bỏ lợi thế đi trước.
    """
    label_a = label_a or agent_a.name
    label_b = label_b or agent_b.name
    if label_a == label_b:
        label_a, label_b = label_a + " (A)", label_b + " (B)"

    total = Tally()
    as_white, as_black = Tally(), Tally()
    reasons = Counter()
    plies_all = []
    times = {"A": [], "B": []}
    t_start = time.perf_counter()

    for i in range(n_games):
        a_is_white = (i % 2 == 0) or not swap_colors
        w, b = (agent_a, agent_b) if a_is_white else (agent_b, agent_a)
        wl, bl = (label_a, label_b) if a_is_white else (label_b, label_a)

        rec = play_game(w, b, verbose=verbose, max_plies=max_plies,
                        white_label=wl, black_label=bl)

        # quy kết quả về góc nhìn của A
        a_result = {"1-0": "win", "0-1": "loss", "1/2-1/2": "draw"}[rec.result]
        if not a_is_white and a_result != "draw":
            a_result = "loss" if a_result == "win" else "win"
        bucket = as_white if a_is_white else as_black
        for t in (total, bucket):
            if a_result == "win": t.wins += 1
            elif a_result == "loss": t.losses += 1
            else: t.draws += 1

        reasons[rec.reason] += 1
        plies_all.append(rec.plies)
        a_color = chess.WHITE if a_is_white else chess.BLACK
        times["A"] += rec.move_times[a_color]
        times["B"] += rec.move_times[not a_color]

        if not verbose:
            print(f"\r  Tiến độ: {i + 1}/{n_games} | A: +{total.wins} ={total.draws} -{total.losses}",
                  end="", flush=True)

    elapsed = time.perf_counter() - t_start
    if not verbose:
        print()
    print_report(label_a, label_b, total, as_white, as_black, reasons,
                 plies_all, times, elapsed)


# ====================== báo cáo ======================
def _elo_diff(score: float):
    if score <= 0: return "-inf"
    if score >= 1: return "+inf"
    return f"{-400 * math.log10(1 / score - 1):+.0f}"


def _fmt_time(ts):
    if not ts: return "n/a"
    return f"TB {sum(ts) / len(ts) * 1000:.1f} ms | max {max(ts) * 1000:.1f} ms"


def print_report(la, lb, total, a_white, a_black, reasons, plies, times, elapsed):
    n = total.n
    line = "=" * 64
    print(f"\n{line}\nKẾT QUẢ: {la}  vs  {lb}  ({n} ván, {elapsed:.1f}s)\n{line}")
    print(f"{'':22}{'Thắng':>8}{'Hòa':>8}{'Thua':>8}{'Điểm %':>10}")

    def row(name, t):
        if t.n == 0:
            return
        print(f"{name:22}{t.wins:>8}{t.draws:>8}{t.losses:>8}{t.score * 100:>9.1f}%")

    print(f"[{la}]")
    row("  Tổng", total)
    row("  Khi cầm Trắng", a_white)
    row("  Khi cầm Đen", a_black)
    print(f"[{lb}]")
    row("  Tổng", Tally(total.losses, total.draws, total.wins))
    row("  Khi cầm Trắng", Tally(a_black.losses, a_black.draws, a_black.wins))
    row("  Khi cầm Đen", Tally(a_white.losses, a_white.draws, a_white.wins))

    print(f"\nChênh lệch Elo ước tính ({la} so với {lb}): {_elo_diff(total.score)}")
    if n:
        print(f"Số ply TB mỗi ván: {sum(plies) / n:.1f} (min {min(plies)}, max {max(plies)})")
    print(f"Thời gian/nước {la}: {_fmt_time(times['A'])}")
    print(f"Thời gian/nước {lb}: {_fmt_time(times['B'])}")
    print("Lý do kết thúc:")
    for r, c in reasons.most_common():
        print(f"  - {r}: {c}")


# ====================== CLI ======================
def _make(name):
    if name not in AGENT_FACTORIES:
        raise SystemExit(f"Không có agent '{name}'. Có: {', '.join(AGENT_FACTORIES)}")
    return AGENT_FACTORIES[name]()


def main():
    p = argparse.ArgumentParser(description="Headless arena cho chess agents")
    p.add_argument("--white", "-w", required=True, help="Agent A (tên trong AGENT_FACTORIES)")
    p.add_argument("--black", "-b", required=True, help="Agent B")
    p.add_argument("-n", "--games", type=int, default=1, help="Số ván (mặc định 1)")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="In legal moves + nước chọn mỗi lượt (tự bật khi n=1)")
    p.add_argument("--no-swap", action="store_true", help="Không đổi phe giữa các ván")
    p.add_argument("--max-plies", type=int, default=1000)
    args = p.parse_args()

    a, b = _make(args.white), _make(args.black)
    try:
        if args.games == 1:
            # 1 ván: A cầm Trắng, B cầm Đen, luôn verbose
            rec = play_game(a, b, verbose=True, max_plies=args.max_plies)
        else:
            play_match(a, b, n_games=args.games, swap_colors=not args.no_swap,
                       max_plies=args.max_plies, verbose=args.verbose)
    finally:
        a.close()
        b.close()


if __name__ == "__main__":
    main()