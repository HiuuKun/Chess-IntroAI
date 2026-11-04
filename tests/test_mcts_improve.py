"""Regression tests for the separate improved agent; no engine is required."""

import math
import random
import time
import unittest
from unittest.mock import patch

import chess

from agents import AGENT_FACTORIES
from agents.mcts_agent import MCTSAgent
from agents.mcts_improve import MCTSImproveAgent, _Node, _SearchTimeout
from agents.random_agent import RandomAgent
from core.game import Game


MATE_TWO = "3r2k1/5ppp/3Rp3/1q2P3/1r3P2/7P/2P3P1/6RK w - - 1 39"
MATE_THREE = "6k1/5ppp/R2r4/p3r3/8/1P4P1/5P1P/6K1 w - - 0 25"
DRAW_TWO = "8/6pp/5ppk/6q1/5PPK/6P1/6PP/8 w - - 0 1"


class MCTSImproveTests(unittest.TestCase):
    def test_registered_separately(self):
        self.assertIsInstance(AGENT_FACTORIES["MCTS"](), MCTSAgent)
        self.assertIsInstance(AGENT_FACTORIES["mcts_improve"](), MCTSImproveAgent)

    def test_existing_game_loop_completes_the_mate_three(self):
        game = Game(MCTSImproveAgent(seed=0), RandomAgent())
        game.board = chess.Board(MATE_THREE)
        try:
            for _ in range(5):
                if game.is_over():
                    break
                self.assertIsNotNone(game.step())
            self.assertEqual(game.result(), "1-0")
            self.assertEqual(game.outcome().termination, chess.Termination.CHECKMATE)
        finally:
            game.close()

    def test_invalid_configuration(self):
        for options in (
            {"simulations": 0}, {"simulations": True}, {"simulations": 1.5},
            {"rollout_depth": -1}, {"rollout_depth": False},
            {"exploration": -1}, {"exploration": math.nan},
            {"seed": "7"}, {"rollout_randomness": 1.1},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                MCTSImproveAgent(**options)
        for budget in (-1, math.inf, math.nan):
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                MCTSImproveAgent().choose_move(chess.Board(), budget)

    def test_legal_moves_and_input_history_unchanged_for_both_colors(self):
        for black_to_move in (False, True):
            board = chess.Board()
            if black_to_move:
                board.push_uci("e2e4")
            before = (board.fen(), tuple(board.move_stack))
            agent = MCTSImproveAgent(simulations=24, rollout_depth=3, seed=7)
            move = agent.choose_move(board)
            self.assertIn(move, board.legal_moves)
            self.assertEqual(before, (board.fen(), tuple(board.move_stack)))
            stats = agent.last_stats
            self.assertEqual(stats["iterations"], 24)
            self.assertEqual(stats["root_visits"], stats["iterations"])
            self.assertEqual(stats["root_child_visits"], stats["iterations"])
            self.assertEqual(stats["terminal_rollouts"] + stats["heuristic_cutoffs"]
                             + stats["proven_leaves"], stats["iterations"])
            for child in agent.last_root_stats:
                self.assertTrue(0 <= child["mean"] <= 1)
                self.assertTrue(child["proven"] is None or child["proven"] in (0, 0.5, 1))

    def test_terminal_positions_with_and_without_legal_moves(self):
        for fen in (
            "7k/6Q1/6K1/8/8/8/8/8 b - - 0 1",  # Checkmate.
            "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1",  # Stalemate.
            "4k3/8/8/8/8/8/8/4K3 w - - 0 1",  # Insufficient material.
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 100 1",
        ):
            with self.subTest(fen=fen):
                board = chess.Board(fen)
                self.assertIsNotNone(board.outcome(claim_draw=True))
                agent = MCTSImproveAgent()
                self.assertIsNone(agent.choose_move(board, time_limit=0))
                self.assertEqual(agent.last_stats["stop_reason"], "terminal")

    def test_repetition_history_is_used(self):
        board = chess.Board()
        for move in ["g1f3", "g8f6", "f3g1", "f6g8"] * 2:
            board.push_uci(move)
        self.assertIsNotNone(board.outcome(claim_draw=True))
        self.assertIsNone(chess.Board(board.fen()).outcome(claim_draw=True))
        agent = MCTSImproveAgent()
        self.assertIsNone(agent.choose_move(board))
        self.assertEqual(len(board.move_stack), 8)

    def test_zero_budget_returns_deterministic_fallback(self):
        board = chess.Board()
        agent = MCTSImproveAgent(seed=4)
        self.assertEqual(agent.choose_move(board, time_limit=0).uci(), "a2a3")
        self.assertEqual(agent.last_stats["iterations"], 0)
        self.assertEqual(agent.last_stats["stop_reason"], "fallback")

    def test_deadline_is_checked_within_search(self):
        agent = MCTSImproveAgent(simulations=100000, rollout_depth=1000, seed=5)
        board = chess.Board()
        started = time.perf_counter()
        move = agent.choose_move(board, time_limit=0.02)
        elapsed = time.perf_counter() - started
        self.assertIn(move, board.legal_moves)
        # Allows scheduling noise; an unbounded 1000-ply rollout must not finish.
        self.assertLess(elapsed, 0.30)
        self.assertLess(agent.last_stats["iterations"], 100000)
        self.assertIn(agent.last_stats["stop_reason"], ("deadline", "fallback"))

    def test_interrupted_rollout_is_discarded_and_traversal_restored(self):
        board = chess.Board()
        traversals = []
        agent = MCTSImproveAgent(seed=3)

        def interrupt(private, *args):
            traversals.append(private)
            private.push(next(iter(private.legal_moves)))
            raise _SearchTimeout

        with patch.object(agent, "_rollout", side_effect=interrupt):
            self.assertIn(agent.choose_move(board), board.legal_moves)
        self.assertEqual(len(traversals), 1)
        self.assertEqual(traversals[0].fen(), board.fen())
        self.assertEqual(traversals[0].move_stack, board.move_stack)
        self.assertEqual(agent.last_stats["iterations"], 0)
        self.assertEqual(agent.last_stats["root_visits"], 0)
        self.assertEqual(agent.last_stats["root_child_visits"], 0)
        self.assertEqual(agent.last_stats["discarded_iterations"], 1)

    def test_time_budget_does_not_override_simulation_cap(self):
        agent = MCTSImproveAgent(simulations=3, rollout_depth=0, seed=3)
        agent.choose_move(chess.Board(), time_limit=1)
        self.assertEqual(agent.last_stats["iterations"], 3)
        self.assertEqual(agent.last_stats["stop_reason"], "simulations")

    def test_seed_is_repeatable_and_does_not_change_global_rng(self):
        random.seed(823)
        before = random.getstate()
        first = MCTSImproveAgent(simulations=30, rollout_depth=3, seed=12)
        second = MCTSImproveAgent(simulations=30, rollout_depth=3, seed=12)
        self.assertEqual(first.choose_move(chess.Board()), second.choose_move(chess.Board()))
        self.assertEqual(first.last_root_stats, second.last_root_stats)
        self.assertEqual(random.getstate(), before)

    def test_uct_is_adversarial_for_both_root_colors(self):
        for color in chess.COLORS:
            node = _Node(turn=color, visits=40)
            low = _Node(turn=not color, visits=20, value=2)
            high = _Node(turn=not color, visits=20, value=18)
            node.children = [low, high]
            agent = MCTSImproveAgent()
            self.assertIs(agent._select_child(node, color), high)
            node.turn = not color
            self.assertIs(agent._select_child(node, color), low)

    def test_proof_backup_requires_all_defenses_unless_a_win_is_known(self):
        for color in chess.COLORS:
            # The opponent can refute an apparently winning line.
            node = _Node(turn=not color)
            node.children = [_Node(turn=color, proven=1.0), _Node(turn=color)]
            MCTSImproveAgent._update_proven(node, color)
            self.assertIsNone(node.proven)
            node.children[1].proven = 0.0
            node.untried = [chess.Move.from_uci("a2a3")]
            MCTSImproveAgent._update_proven(node, color)
            self.assertEqual(node.proven, 0.0)
            # All opponent replies must be solved before proving a root win.
            node.proven = None
            node.children[1].proven = 1.0
            MCTSImproveAgent._update_proven(node, color)
            self.assertIsNone(node.proven)
            node.untried = []
            MCTSImproveAgent._update_proven(node, color)
            self.assertEqual(node.proven, 1.0)
            # A draw is certified only once every alternative is solved.
            parent = _Node(turn=color, children=[_Node(turn=not color, proven=0.0),
                                                _Node(turn=not color, proven=0.5)])
            MCTSImproveAgent._update_proven(parent, color)
            self.assertEqual(parent.proven, 0.5)

    def test_final_choice_uses_mean_for_tied_visits_and_prefers_proven_win(self):
        low = _Node(turn=False, move=chess.Move.from_uci("a2a3"), visits=5, value=2)
        high = _Node(turn=False, move=chess.Move.from_uci("b2b3"), visits=5, value=4)
        root = _Node(turn=True, children=[low, high])
        self.assertIs(MCTSImproveAgent._best_child(root), high)
        low.proven = 1.0
        low.visits = 1
        low.value = 1
        self.assertIs(MCTSImproveAgent._best_child(root), low)

    def test_proven_draw_beats_worse_or_equal_estimates_despite_visits(self):
        for color in chess.COLORS:
            for estimated in (0.0, 0.4946889732493848, 0.5):
                with self.subTest(color=color, estimated=estimated):
                    draw = _Node(turn=not color, move=chess.Move.from_uci("f4g5"),
                                 visits=1, value=0.5, proven=0.5)
                    uncertain = _Node(turn=not color, move=chess.Move.from_uci("h4h3"),
                                      visits=199, value=199 * estimated)
                    root = _Node(turn=color, children=[draw, uncertain])
                    self.assertIs(MCTSImproveAgent._best_child(root), draw)

    def test_proven_draw_is_kept_while_search_looks_for_a_proven_win(self):
        draw = _Node(turn=False, move=chess.Move.from_uci("a2a3"),
                     visits=1, value=0.5, proven=0.5)
        promising = _Node(turn=False, move=chess.Move.from_uci("b2b3"),
                          visits=199, value=199 * 0.75)
        root = _Node(turn=True, visits=200, children=[draw, promising])
        self.assertIs(MCTSImproveAgent()._select_child(root, True), promising)
        self.assertIs(MCTSImproveAgent._best_child(root), draw)
        promising.proven = 1.0
        self.assertIs(MCTSImproveAgent._best_child(root), promising)

    def test_proofs_override_stale_sample_means(self):
        draw = _Node(turn=False, move=chess.Move.from_uci("b2b3"),
                     visits=5, value=4.5, proven=0.5)
        other_draw = _Node(turn=False, move=chess.Move.from_uci("a2a3"),
                           visits=5, value=0.5, proven=0.5)
        uncertain = _Node(turn=False, move=chess.Move.from_uci("d2d3"),
                          visits=199, value=199 * 0.6)
        win = _Node(turn=False, move=chess.Move.from_uci("c2c3"),
                    visits=1, value=0, proven=1.0)
        root = _Node(turn=True, children=[draw, other_draw, uncertain])
        self.assertIs(MCTSImproveAgent._best_child(root), other_draw)
        root.children.append(win)
        self.assertIs(MCTSImproveAgent._best_child(root), win)

    def test_exact_draw_found_in_interrupted_iteration_remains_selectable(self):
        draw = _Node(turn=False, move=chess.Move.from_uci("a2a3"), proven=0.5)
        uncertain = _Node(turn=False, move=chess.Move.from_uci("b2b3"),
                          visits=199, value=0)
        root = _Node(turn=True, children=[draw, uncertain])
        self.assertIs(MCTSImproveAgent._best_child(root), draw)
        # An unvisited, unresolved node has no evidence and is not a candidate.
        unknown = _Node(turn=False, move=chess.Move.from_uci("c2c3"))
        self.assertIsNone(MCTSImproveAgent._best_child(_Node(turn=True, children=[unknown])))

    def test_forced_draw_is_completed_for_both_colors(self):
        white = chess.Board(DRAW_TWO)
        for mirrored in (False, True):
            for seed in (0, 2, 9):
                with self.subTest(mirrored=mirrored, seed=seed):
                    board = white.mirror() if mirrored else white.copy()
                    before = (board.fen(), tuple(board.move_stack))
                    move = MCTSImproveAgent(seed=seed).choose_move(board)
                    expected = "f5g4" if mirrored else "f4g5"
                    self.assertEqual(move.uci(), expected)
                    self.assertEqual(before, (board.fen(), tuple(board.move_stack)))
                    board.push(move)
                    self.assertEqual(board.legal_moves.count(), 1)
                    board.push(next(iter(board.legal_moves)))
                    final = MCTSImproveAgent(seed=seed).choose_move(board)
                    self.assertEqual(final.uci(), "h5h6" if mirrored else "h4h3")
                    board.push(final)
                    self.assertEqual(board.outcome().termination, chess.Termination.STALEMATE)

    def test_mate_in_one_for_both_colors(self):
        white = chess.Board("k7/8/1QK5/8/8/8/8/8 w - - 0 1")
        for board in (white, white.mirror()):
            agent = MCTSImproveAgent(seed=0)
            move = agent.choose_move(board)
            board.push(move)
            self.assertTrue(board.is_checkmate())
            self.assertEqual(agent.last_stats["stop_reason"], "proven")

    def test_mate_two_and_three_for_both_colors(self):
        for fen, expected in ((MATE_TWO, "d6d8"), (MATE_THREE, "a6a8")):
            white = chess.Board(fen)
            for mirrored in (False, True):
                board = white.mirror() if mirrored else white
                move = chess.Move.from_uci(expected)
                if mirrored:
                    move = chess.Move(chess.square_mirror(move.from_square),
                                      chess.square_mirror(move.to_square))
                for seed in (0, 1, 9):
                    with self.subTest(fen=fen, mirrored=mirrored, seed=seed):
                        agent = MCTSImproveAgent(seed=seed)
                        self.assertEqual(agent.choose_move(board), move)
                        self.assertEqual(agent.last_stats["stop_reason"], "proven")

    def test_only_legal_reply_is_returned(self):
        board = chess.Board(MATE_TWO)
        board.push_uci("d6d8")
        self.assertEqual(board.legal_moves.count(), 1)
        agent = MCTSImproveAgent(seed=0)
        self.assertEqual(agent.choose_move(board).uci(), "b5e8")
        self.assertEqual(agent.last_stats["stop_reason"], "forced")

    def test_castling_and_all_promotions_are_expanded_legally(self):
        fixtures = (
            ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", {"e1g1", "e1c1"}),
            ("4k3/P7/8/8/8/8/8/4K3 w - - 0 1", {"a7a8q", "a7a8r", "a7a8b", "a7a8n"}),
        )
        for fen, expected in fixtures:
            board = chess.Board(fen)
            agent = MCTSImproveAgent(simulations=40, rollout_depth=0, seed=0)
            self.assertIn(agent.choose_move(board), board.legal_moves)
            expanded = {item["move"] for item in agent.last_root_stats}
            self.assertTrue(expected <= expanded)

    def test_en_passant_is_scored_as_a_capture(self):
        board = chess.Board("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1")
        agent = MCTSImproveAgent(simulations=1, rollout_depth=0, seed=0)
        self.assertEqual(agent.choose_move(board).uci(), "e5d6")

    def test_nonterminal_evaluation_is_bounded_and_color_symmetric(self):
        board = chess.Board("4k3/8/8/8/8/8/8/4KQ2 w - - 0 1")
        agent = MCTSImproveAgent()
        white, black = agent.evaluate(board, True), agent.evaluate(board, False)
        self.assertTrue(0.5 < white < 1)
        self.assertTrue(0 < black < 0.5)
        self.assertAlmostEqual(white + black, 1)


if __name__ == "__main__":
    unittest.main()
